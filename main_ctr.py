"""Train and evaluate selectable career CTR experiments."""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import (
    CareerDatasetConfig,
    build_career_dataloaders,
    build_career_datasets,
)
from models import DCNv1, DCNv2, DIEN, DeepFM, DeepInterestNet
from utils import setup_seed, str2list


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TOP_K = (1, 3, 5, 10, 20, 50)


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _new_totals(ks: Sequence[int]) -> dict[str, Any]:
    return {
        "count": 0,
        "logloss_sum": 0.0,
        "auc_sum": 0.0,
        "recall_hits": {int(k): 0 for k in ks},
    }


def _update_totals(
    totals: dict[str, Any],
    logits: torch.Tensor,
    targets: torch.Tensor,
    ks: Sequence[int],
) -> None:
    batch_size, candidate_count = logits.shape
    rows = torch.arange(batch_size, device=logits.device)
    positive_scores = logits[rows, targets]
    positions = torch.arange(candidate_count, device=logits.device).unsqueeze(0)
    negative_mask = positions != targets.unsqueeze(1)
    lower = ((logits < positive_scores.unsqueeze(1)) & negative_mask).sum(dim=1)
    equal = ((logits == positive_scores.unsqueeze(1)) & negative_mask).sum(dim=1)
    auc = (lower + 0.5 * equal).float() / (candidate_count - 1)
    rank = 1 + ((logits > positive_scores.unsqueeze(1)) & negative_mask).sum(dim=1)

    totals["count"] += batch_size
    totals["logloss_sum"] += float(
        F.cross_entropy(logits, targets, reduction="sum")
    )
    totals["auc_sum"] += float(auc.sum())
    for k in ks:
        totals["recall_hits"][int(k)] += int((rank <= k).sum())


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ks: Sequence[int],
) -> dict[str, Any]:
    model.eval()
    totals = _new_totals(ks)
    task = model.prediction_task
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            output = model(batch)
            _update_totals(
                totals,
                output["logits"],
                batch[f"{task}_target_index"],
                ks,
            )
    count = totals["count"]
    return {
        "count": count,
        "auc": totals["auc_sum"] / count,
        "logloss": totals["logloss_sum"] / count,
        "recall": {
            int(k): totals["recall_hits"][int(k)] / count for k in ks
        },
    }


def print_metrics(split: str, metrics: Mapping[str, Any], ks: Sequence[int]) -> None:
    recalls = "  ".join(
        f"R@{k}={metrics['recall'][int(k)]:.4f}" for k in ks
    )
    print(
        f"[{split}] AUC={metrics['auc']:.4f}  "
        f"LogLoss={metrics['logloss']:.4f}"
    )
    print(f"       {recalls}")


def build_model(args: argparse.Namespace, train_dataset: Any) -> torch.nn.Module:
    model_types = {
        "DIN": DeepInterestNet,
        "DIEN": DIEN,
        "DCNv1": DCNv1,
        "DCNv2": DCNv2,
        "DeepFM": DeepFM,
    }
    try:
        model_type = model_types[args.model]
    except KeyError as error:
        raise ValueError(f"Unsupported model: {args.model!r}") from error
    task = args.prediction_task
    vocab_size = (
        train_dataset.role_vocab_size
        if task == "role"
        else train_dataset.company_vocab_size
    )
    model_kwargs = {
        "prediction_task": task,
        "adaptor_name": args.adaptor,
        "vocab_size": vocab_size,
        "factual_dim": train_dataset.embedding_dim,
        "embedding_dim": args.embed_dim,
        "tenure_hidden_dim": args.tenure_hidden_dim,
        "attention_architecture": args.attention_arch,
        "final_mlp_architecture": args.final_mlp_arch,
        "expert_architecture": args.convert_arch,
        "shared_expert_count": args.shared_expert_num,
        "specific_expert_count": args.specific_expert_num,
        "history_specific_expert_count": args.history_specific_expert_num,
        "residual_expert_hidden_dim": args.residual_expert_hidden_dim,
        "residual_initial_alpha": args.residual_initial_alpha,
        "history_decay_rate": args.history_decay_rate,
        "dropout": args.dropout,
        "convert_dropout": args.convert_dropout,
        "residual_dropout": args.residual_dropout,
    }
    if model_type is DIEN:
        model_kwargs["dien_gru"] = args.dien_gru
    elif model_type in {DCNv1, DCNv2}:
        model_kwargs.pop("attention_architecture")
        model_kwargs["deep_architecture"] = args.dcn_deep_arch
        model_kwargs["cross_layer_count"] = args.dcn_cross_num
    elif model_type is DeepFM:
        model_kwargs.pop("attention_architecture")
        model_kwargs.pop("final_mlp_architecture")
        model_kwargs["deep_architecture"] = args.deepfm_deep_arch
    model = model_type(**model_kwargs)
    nn_init = torch.nn.init
    nn_init.xavier_normal_(model.item_embedding.weight)
    with torch.no_grad():
        model.item_embedding.weight[0].zero_()
    return model


def train(args: argparse.Namespace) -> None:
    setup_seed(args.seed, args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config = CareerDatasetConfig(
        num_candidates=args.num_candidates,
        max_history_length=args.max_history_length,
        candidate_position_seed=args.candidate_position_seed,
    )
    datasets = build_career_datasets(args.data_dir, config)
    loaders = build_career_dataloaders(
        datasets,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    model = build_model(args, datasets.train).to(device)
    task = args.prediction_task
    target_key = f"{task}_target_index"
    checkpoint_path = (
        Path(args.save_dir) / f"{args.model}_{args.adaptor}_{task}.pt"
    )

    if args.eval_only:
        checkpoint = torch.load(args.reload_path or checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        best_epoch = 0
        stale_evaluations = 0

        for epoch in range(1, args.epoch_num + 1):
            model.train()
            loss_sum = 0.0
            example_count = 0
            progress = tqdm(
                loaders["train"],
                desc=f"Epoch {epoch}/{args.epoch_num}",
            )
            for raw_batch in progress:
                batch = move_batch_to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss = F.cross_entropy(output["logits"], batch[target_key])
                loss.backward()
                if args.grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                batch_size = batch[target_key].shape[0]
                loss_sum += float(loss.detach()) * batch_size
                example_count += batch_size
                progress.set_postfix(
                    loss=f"{loss_sum / example_count:.4f}"
                )

            print(f"Epoch {epoch}: train loss={loss_sum / example_count:.4f}")
            test_metrics = evaluate(model, loaders["test"], device, args.top_k)
            ood_metrics = evaluate(model, loaders["ood"], device, args.top_k)
            print_metrics(f"Epoch {epoch} Test", test_metrics, args.top_k)
            print_metrics(f"Epoch {epoch} OOD", ood_metrics, args.top_k)

            if test_metrics["logloss"] < best_loss:
                best_loss = test_metrics["logloss"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_evaluations = 0
                if args.save_checkpoint:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": best_state,
                            "epoch": epoch,
                            "prediction_task": task,
                            "model": args.model,
                            "adaptor": args.adaptor,
                            "test_metrics": test_metrics,
                            "ood_metrics": ood_metrics,
                        },
                        checkpoint_path,
                    )
            else:
                stale_evaluations += 1
                if stale_evaluations >= args.patience:
                    print(
                        "Early stopping: test logloss did not improve for "
                        f"{args.patience} epochs."
                    )
                    break

        if best_state is None:
            raise RuntimeError("Training completed without a valid model state")
        model.load_state_dict(best_state)
        print(f"Restored epoch {best_epoch}, best test logloss={best_loss:.4f}")

    print("Final evaluation")
    print_metrics("Test", evaluate(model, loaders["test"], device, args.top_k), args.top_k)
    print_metrics("OOD", evaluate(model, loaders["ood"], device, args.top_k), args.top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save_dir", type=Path, default=Path("./model"))
    parser.add_argument("--reload_path", type=Path, default=None)
    parser.add_argument("--prediction_task", choices=("role", "company"), default="role")
    parser.add_argument(
        "--model",
        choices=("DIN", "DIEN", "DCNv1", "DCNv2", "DeepFM"),
        default="DIN",
        help="CTR backbone model.",
    )
    parser.add_argument(
        "--adaptor",
        choices=("none", "kar_base", "kar_ours"),
        default="kar_base",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_norm", type=float, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--max_history_length", type=int, default=20)
    parser.add_argument("--candidate_position_seed", type=int, default=42)
    parser.add_argument("--top_k", type=str2list, default=list(DEFAULT_TOP_K))
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--tenure_hidden_dim", type=int, default=16)
    parser.add_argument("--attention_arch", type=str2list, default=[32, 16])
    parser.add_argument(
        "--dien_gru",
        choices=("GRU", "AIGRU", "AGRU", "AUGRU"),
        default="AIGRU",
        help="DIEN interest-evolution unit; KAR's run_ctr.py uses AIGRU.",
    )
    parser.add_argument("--dcn_deep_arch", type=str2list, default=[200, 80])
    parser.add_argument("--deepfm_deep_arch", type=str2list, default=[200, 80])
    parser.add_argument("--dcn_cross_num", type=int, default=3)
    parser.add_argument("--final_mlp_arch", type=str2list, default=[200, 80])
    parser.add_argument("--convert_arch", type=str2list, default=[128, 32])
    parser.add_argument("--shared_expert_num", type=int, default=2)
    parser.add_argument("--specific_expert_num", type=int, default=6)
    parser.add_argument("--history_specific_expert_num", type=int, default=2)
    parser.add_argument("--residual_expert_hidden_dim", type=int, default=32)
    parser.add_argument("--residual_initial_alpha", type=float, default=0.01)
    parser.add_argument("--history_decay_rate", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--convert_dropout", type=float, default=0.0)
    parser.add_argument("--residual_dropout", type=float, default=0.1)
    args = parser.parse_args()
    args.top_k = tuple(sorted(set(args.top_k)))
    return args


if __name__ == "__main__":
    train(parse_args())
