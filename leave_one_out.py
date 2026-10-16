"""Leave-one-out behavior attribution for the three DIN adaptor variants.

For each selected evaluation row, the script first scores the original
history and then removes every occurrence of each unique behavior item in
turn.  It records the
resulting change in the correct candidate's logit, candidate-set probability,
margin over the strongest negative, loss, and rank.

By default this analyzes company-task OOD rows whose visible company history
contains at least one true OOD company.  Existing datasets, models, and
checkpoints are only read; this script does not train or modify them.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    DEFAULT_DATA_DIR,
    CareerDatasetConfig,
    build_career_datasets,
    career_collate_fn,
)
from main_ctr import build_model, move_batch_to_device
from utils import setup_seed, str2list


ROOT = Path(__file__).resolve().parent
ADAPTORS = ("none", "kar_base", "kar_ours")
DEFAULT_OOD_ITEMS = ROOT / "data" / "true_ood_companies.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "leave_one_out"

OUTPUT_FIELDS = (
    "adaptor",
    "split",
    "row_index",
    "user_id",
    "target_item",
    "history_items",
    "history_length",
    "removed_positions",
    "removed_occurrence_count",
    "removed_item",
    "removed_item_is_ood",
    "baseline_predicted_item",
    "ablated_predicted_item",
    "baseline_top1_correct",
    "ablated_top1_correct",
    "top1_success_lost",
    "baseline_target_logit",
    "ablated_target_logit",
    "target_logit_drop",
    "baseline_target_probability",
    "ablated_target_probability",
    "target_probability_drop",
    "baseline_target_margin",
    "ablated_target_margin",
    "target_margin_drop",
    "baseline_logloss",
    "ablated_logloss",
    "logloss_increase",
    "baseline_rank",
    "ablated_rank",
    "rank_increase",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_array(values: Any) -> str:
    return json.dumps(_json_value(values), ensure_ascii=False, separators=(",", ":"))


def _checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    overrides = {
        "none": args.none_checkpoint,
        "kar_base": args.kar_base_checkpoint,
        "kar_ours": args.kar_ours_checkpoint,
    }
    return {
        adaptor: (
            overrides[adaptor]
            or args.checkpoint_dir / f"DIN_{adaptor}_{args.prediction_task}.pt"
        ).expanduser().resolve()
        for adaptor in ADAPTORS
    }


def _load_models(
    args: argparse.Namespace,
    train_dataset: Any,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for adaptor, path in _checkpoint_paths(args).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {adaptor} checkpoint: {path}")
        model_args = argparse.Namespace(**vars(args))
        model_args.model = "DIN"
        model_args.adaptor = adaptor
        model = build_model(model_args, train_dataset).to(device)
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            raise RuntimeError(
                f"Checkpoint {path} does not match the supplied architecture "
                "options. Use the same settings used during training."
            ) from error
        model.eval()
        models[adaptor] = model
    return models


def _select_batch(batch: Mapping[str, Any], indices: list[int]) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return {
        key: value.index_select(0, index_tensor)
        if isinstance(value, torch.Tensor)
        else [value[index] for index in indices]
        for key, value in batch.items()
    }


def _remove_history_items(
    batch: Mapping[str, Any], task: str, removal_mask: torch.Tensor
) -> dict[str, Any]:
    """Remove selected behavior slots and repack every affected sequence.

    ``removal_mask`` has shape ``[B,H]``. Repacking (instead of merely setting
    masks false) preserves the chronological sequence convention and
    KAR-Ours' recency-distance calculation.
    """

    history_mask_key = f"{task}_history_mask"
    valid_after_removal = batch[history_mask_key].bool() & ~removal_mask.bool()
    output = dict(batch)
    keys = (
        f"{task}_history_indices",
        f"{task}_history_sequence",
        f"{task}_history_tenures",
        f"{task}_history_mask",
    )
    for key in keys:
        source = batch[key]
        packed = torch.zeros_like(source)
        for row_index in range(source.shape[0]):
            kept = source[row_index][valid_after_removal[row_index]]
            packed[row_index, : kept.shape[0]] = kept
        output[key] = packed
    return output


def _score_details(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    rows = torch.arange(logits.shape[0], device=logits.device)
    target_logits = logits[rows, targets]
    negative_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=logits.shape[1]).bool(), float("-inf")
    )
    ranks = 1 + (logits > target_logits.unsqueeze(1)).sum(dim=1)
    return {
        "target_logit": target_logits,
        "target_probability": torch.softmax(logits, dim=1)[rows, targets],
        "target_margin": target_logits - negative_logits.max(dim=1).values,
        "logloss": F.cross_entropy(logits, targets, reduction="none"),
        "rank": ranks,
        "predicted_index": logits.argmax(dim=1),
    }


def _load_ood_items(args: argparse.Namespace) -> set[Any]:
    path = args.ood_items.expanduser().resolve()
    frame = pd.read_csv(path, keep_default_na=False)
    if args.ood_item_column not in frame.columns:
        raise ValueError(f"OOD item file lacks {args.ood_item_column!r} column: {path}")
    return set(frame[args.ood_item_column].tolist())


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def run(args: argparse.Namespace) -> None:
    setup_seed(args.seed, args.deterministic)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    ood_items = _load_ood_items(args)
    config = CareerDatasetConfig(
        num_candidates=args.num_candidates,
        max_history_length=args.max_history_length,
        candidate_position_seed=args.candidate_position_seed,
    )
    datasets = build_career_datasets(args.data_dir, config)
    dataset = getattr(datasets, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=career_collate_fn,
    )
    models = _load_models(args, datasets.train, device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        adaptor: output_dir
        / f"leave_one_out_{adaptor}_{args.prediction_task}_{args.split}.csv"
        for adaptor in ADAPTORS
    }
    handles: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}
    summaries: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    selected_samples = 0
    attribution_rows = 0
    task = args.prediction_task
    history_ids_key = f"history_{task}_ids"
    target_key = f"{task}_target_index"
    candidate_ids_key = f"{task}_candidate_ids"

    try:
        for adaptor, path in output_paths.items():
            handle = path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            handles[adaptor] = handle
            writers[adaptor] = writer

        progress = tqdm(loader, desc=f"DIN leave-one-out ({args.split}/{task})")
        for raw_batch in progress:
            selected_indices = []
            for index, history in enumerate(raw_batch[history_ids_key]):
                if not args.ood_history_only or any(item in ood_items for item in history):
                    selected_indices.append(index)
            if not selected_indices:
                continue
            if args.max_samples is not None:
                remaining = args.max_samples - selected_samples
                if remaining <= 0:
                    break
                selected_indices = selected_indices[:remaining]

            selected = _select_batch(raw_batch, selected_indices)
            batch = move_batch_to_device(selected, device)
            targets = batch[target_key]
            histories = selected[history_ids_key]
            candidates = selected[candidate_ids_key]
            lengths = [len(history) for history in histories]
            unique_history_items = [list(dict.fromkeys(history)) for history in histories]
            maximum_unique_items = max(len(items) for items in unique_history_items)

            with torch.inference_mode():
                baseline = {
                    adaptor: _score_details(model(batch)["logits"], targets)
                    for adaptor, model in models.items()
                }
                for item_number in range(maximum_unique_items):
                    active_rows = [
                        row_index
                        for row_index, items in enumerate(unique_history_items)
                        if item_number < len(items)
                    ]
                    if not active_rows:
                        continue
                    removal_mask = torch.zeros_like(
                        batch[f"{task}_history_mask"], dtype=torch.bool
                    )
                    removed_items: dict[int, Any] = {}
                    removed_positions: dict[int, list[int]] = {}
                    for row_number in active_rows:
                        removed_item = unique_history_items[row_number][item_number]
                        positions = [
                            position
                            for position, history_item in enumerate(histories[row_number])
                            if history_item == removed_item
                        ]
                        removed_items[row_number] = removed_item
                        removed_positions[row_number] = positions
                        removal_mask[row_number, positions] = True
                    ablated_batch = _remove_history_items(batch, task, removal_mask)
                    for adaptor, model in models.items():
                        ablated = _score_details(model(ablated_batch)["logits"], targets)
                        base = baseline[adaptor]
                        for row_number in active_rows:
                            removed_item = removed_items[row_number]
                            positions = removed_positions[row_number]
                            is_ood = removed_item in ood_items
                            target_index = int(targets[row_number].item())
                            base_pred_index = int(base["predicted_index"][row_number].item())
                            ablated_pred_index = int(
                                ablated["predicted_index"][row_number].item()
                            )
                            base_correct = base_pred_index == target_index
                            ablated_correct = ablated_pred_index == target_index

                            def scalar(name: str, source: Mapping[str, torch.Tensor]) -> float:
                                return float(source[name][row_number].item())

                            base_logit = scalar("target_logit", base)
                            ablated_logit = scalar("target_logit", ablated)
                            base_probability = scalar("target_probability", base)
                            ablated_probability = scalar("target_probability", ablated)
                            base_margin = scalar("target_margin", base)
                            ablated_margin = scalar("target_margin", ablated)
                            base_loss = scalar("logloss", base)
                            ablated_loss = scalar("logloss", ablated)
                            base_rank = int(base["rank"][row_number].item())
                            ablated_rank = int(ablated["rank"][row_number].item())
                            values = {
                                "target_logit_drop": base_logit - ablated_logit,
                                "target_probability_drop": (
                                    base_probability - ablated_probability
                                ),
                                "target_margin_drop": base_margin - ablated_margin,
                                "logloss_increase": ablated_loss - base_loss,
                                "rank_increase": ablated_rank - base_rank,
                            }
                            group = "ood" if is_ood else "non_ood"
                            for name, value in values.items():
                                summaries[(adaptor, group)][name].append(float(value))
                            record = {
                                "adaptor": adaptor,
                                "split": args.split,
                                "row_index": int(selected["row_index"][row_number].item()),
                                "user_id": _json_value(selected["user_id"][row_number]),
                                "target_item": _json_value(
                                    selected[f"pos_{task}_id"][row_number]
                                ),
                                "history_items": _json_array(histories[row_number]),
                                "history_length": lengths[row_number],
                                "removed_positions": _json_array(positions),
                                "removed_occurrence_count": len(positions),
                                "removed_item": _json_value(removed_item),
                                "removed_item_is_ood": is_ood,
                                "baseline_predicted_item": _json_value(
                                    candidates[row_number][base_pred_index]
                                ),
                                "ablated_predicted_item": _json_value(
                                    candidates[row_number][ablated_pred_index]
                                ),
                                "baseline_top1_correct": base_correct,
                                "ablated_top1_correct": ablated_correct,
                                "top1_success_lost": base_correct and not ablated_correct,
                                "baseline_target_logit": base_logit,
                                "ablated_target_logit": ablated_logit,
                                "baseline_target_probability": base_probability,
                                "ablated_target_probability": ablated_probability,
                                "baseline_target_margin": base_margin,
                                "ablated_target_margin": ablated_margin,
                                "baseline_logloss": base_loss,
                                "ablated_logloss": ablated_loss,
                                "baseline_rank": base_rank,
                                "ablated_rank": ablated_rank,
                                **values,
                            }
                            writers[adaptor].writerow(record)
                            attribution_rows += 1

            selected_samples += len(selected_indices)
            progress.set_postfix(samples=selected_samples)
            if args.max_samples is not None and selected_samples >= args.max_samples:
                break
    finally:
        for handle in handles.values():
            handle.close()

    summary_path = output_dir / f"leave_one_out_summary_{task}_{args.split}.csv"
    summary_fields = (
        "adaptor",
        "removed_item_group",
        "count",
        "mean_target_logit_drop",
        "mean_target_probability_drop",
        "mean_target_margin_drop",
        "mean_logloss_increase",
        "mean_rank_increase",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for adaptor in ADAPTORS:
            for group in ("ood", "non_ood"):
                values = summaries[(adaptor, group)]
                writer.writerow(
                    {
                        "adaptor": adaptor,
                        "removed_item_group": group,
                        "count": len(values["target_logit_drop"]),
                        "mean_target_logit_drop": _mean(values["target_logit_drop"]),
                        "mean_target_probability_drop": _mean(
                            values["target_probability_drop"]
                        ),
                        "mean_target_margin_drop": _mean(values["target_margin_drop"]),
                        "mean_logloss_increase": _mean(values["logloss_increase"]),
                        "mean_rank_increase": _mean(values["rank_increase"]),
                    }
                )

    print(f"Device: {device}")
    print(f"Selected samples: {selected_samples:,}")
    print(f"Attribution rows across all adaptors: {attribution_rows:,}")
    for adaptor, path in output_paths.items():
        print(f"{adaptor}: {path}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leave-one-out behavior attribution for DIN adaptor variants."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=("train", "test", "ood"), default="ood")
    parser.add_argument(
        "--prediction_task", choices=("role", "company"), default="company"
    )
    parser.add_argument("--ood_items", type=Path, default=DEFAULT_OOD_ITEMS)
    parser.add_argument("--ood_item_column", default="company")
    parser.add_argument("--ood_history_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "model")
    parser.add_argument("--none_checkpoint", type=Path, default=None)
    parser.add_argument("--kar_base_checkpoint", type=Path, default=None)
    parser.add_argument("--kar_ours_checkpoint", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--max_history_length", type=int, default=20)
    parser.add_argument("--candidate_position_seed", type=int, default=42)

    # These settings must match the checkpoints' training architecture.
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--tenure_hidden_dim", type=int, default=16)
    parser.add_argument("--attention_arch", type=str2list, default=[32, 16])
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
