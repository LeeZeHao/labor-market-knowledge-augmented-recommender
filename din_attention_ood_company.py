"""Export DIN attention distributions for histories containing OOD companies.

For every OOD evaluation sample whose model-visible company history contains
at least one company from ``true_ood_companies.csv``, this script runs the
company-task DIN ``none``, ``kar_base``, and ``kar_ours`` checkpoints.

DIN itself uses raw, unnormalized activation scores.  This analysis selects
the scores produced while evaluating the correct target candidate, masks
padding, and applies softmax over the valid history positions.  If multiple
history positions contain OOD companies, their names and individual softmax
weights are stored in corresponding arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
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
DEFAULT_OOD_COMPANIES = ROOT / "data" / "true_ood_companies.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "din_attention_ood_company"
OUTPUT_FIELDS = (
    "ood_row_index",
    "user_id",
    "target_company",
    "predicted_company",
    "history_companies",
    "ood_history_companies",
    "ood_company_names",
    "ood_history_positions",
    "source_sample",
    "attention_softmax",
    "ood_company_attention_softmax",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def json_array(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))


def checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    overrides = {
        "none": args.none_checkpoint,
        "kar_base": args.kar_base_checkpoint,
        "kar_ours": args.kar_ours_checkpoint,
    }
    return {
        adaptor: (
            overrides[adaptor]
            or args.checkpoint_dir / f"DIN_{adaptor}_company.pt"
        ).expanduser().resolve()
        for adaptor in ADAPTORS
    }


def load_models(
    args: argparse.Namespace,
    train_dataset: Any,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    models: dict[str, torch.nn.Module] = {}
    for adaptor, path in checkpoint_paths(args).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {adaptor} checkpoint: {path}")
        model_args = argparse.Namespace(**vars(args))
        model_args.model = "DIN"
        model_args.adaptor = adaptor
        model_args.prediction_task = "company"
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


def select_batch(batch: Mapping[str, Any], indices: list[int]) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return {
        key: value.index_select(0, index_tensor)
        if isinstance(value, torch.Tensor)
        else [value[index] for index in indices]
        for key, value in batch.items()
    }


def softmax_attention(
    raw_scores: torch.Tensor, history_mask: torch.Tensor
) -> torch.Tensor:
    """Softmax candidate-specific DIN scores over valid history positions."""

    if raw_scores.ndim != 3:
        raise ValueError(
            f"Expected attention scores [B,C,H], got {tuple(raw_scores.shape)}"
        )
    valid = history_mask.bool().unsqueeze(1)
    masked_scores = raw_scores.masked_fill(~valid, float("-inf"))
    return torch.softmax(masked_scores, dim=-1)


def source_sample(row: pd.Series) -> dict[str, Any]:
    """Keep source data while excluding the two large 1024d vector columns."""

    excluded = {"prompt_emb", "textual_reasoning_emb"}
    return {
        column: jsonable(value)
        for column, value in row.items()
        if column not in excluded
    }


def run(args: argparse.Namespace) -> None:
    setup_seed(args.seed, args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ood_frame = pd.read_csv(
        args.ood_companies.expanduser().resolve(), keep_default_na=False
    )
    if args.ood_company_column not in ood_frame.columns:
        raise ValueError(
            f"OOD company file lacks {args.ood_company_column!r} column"
        )
    ood_companies = set(ood_frame[args.ood_company_column].tolist())

    config = CareerDatasetConfig(
        num_candidates=args.num_candidates,
        max_history_length=args.max_history_length,
        candidate_position_seed=args.candidate_position_seed,
    )
    datasets = build_career_datasets(args.data_dir, config)
    loader = DataLoader(
        datasets.ood,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=career_collate_fn,
    )
    models = load_models(args, datasets.train, device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}
    paths: dict[str, Path] = {}
    try:
        for adaptor in ADAPTORS:
            path = output_dir / f"din_attention_ood_company_{adaptor}.csv"
            handle = path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            files[adaptor] = handle
            writers[adaptor] = writer
            paths[adaptor] = path

        selected_count = 0
        for raw_batch in tqdm(loader, desc="OOD histories with OOD companies"):
            selected_indices: list[int] = []
            selected_ood_positions: list[list[int]] = []
            for batch_index, history in enumerate(raw_batch["history_company_ids"]):
                positions = [
                    position
                    for position, company in enumerate(history)
                    if company in ood_companies
                ]
                if positions:
                    selected_indices.append(batch_index)
                    selected_ood_positions.append(positions)
            if not selected_indices:
                continue

            selected = select_batch(raw_batch, selected_indices)
            batch = move_batch_to_device(selected, device)
            model_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            with torch.inference_mode():
                for adaptor, model in models.items():
                    output = model(batch)
                    weights = softmax_attention(
                        output["attention_scores"],
                        batch["company_history_mask"],
                    )
                    batch_indices = torch.arange(weights.size(0), device=device)
                    target_weights = weights[
                        batch_indices, batch["company_target_index"]
                    ]
                    predicted_indices = output["logits"].argmax(dim=-1)
                    model_outputs[adaptor] = (
                        target_weights.detach().cpu(),
                        predicted_indices.detach().cpu(),
                    )

            for row_number, positions in enumerate(selected_ood_positions):
                row_index = int(selected["row_index"][row_number].item())
                history = selected["history_company_ids"][row_number]
                history_length = len(history)
                candidates = selected["company_candidate_ids"][row_number]
                ood_history = [history[position] for position in positions]
                common = {
                    "ood_row_index": row_index,
                    "user_id": jsonable(selected["user_id"][row_number]),
                    "target_company": jsonable(selected["pos_company_id"][row_number]),
                    "history_companies": json_array(history),
                    "ood_history_companies": json_array(ood_history),
                    "ood_company_names": json_array(ood_history),
                    "ood_history_positions": json_array(positions),
                    "source_sample": json_array(
                        source_sample(datasets.ood.user_frame.iloc[row_index])
                    ),
                }
                for adaptor in ADAPTORS:
                    target_weights, predicted_indices = model_outputs[adaptor]
                    record = dict(common)
                    record["predicted_company"] = candidates[
                        int(predicted_indices[row_number].item())
                    ]
                    record["attention_softmax"] = json_array(
                        target_weights[row_number, :history_length]
                    )
                    record["ood_company_attention_softmax"] = json_array(
                        target_weights[row_number, positions]
                    )
                    writers[adaptor].writerow(record)
                selected_count += 1
    finally:
        for handle in files.values():
            handle.close()

    print(f"OOD companies loaded: {len(ood_companies):,}")
    print(f"OOD samples processed: {len(datasets.ood):,}")
    print(f"Samples with model-visible OOD history: {selected_count:,}")
    for adaptor in ADAPTORS:
        print(f"{adaptor}: {paths[adaptor]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export softmax-normalized DIN history attention for company-task "
            "OOD samples containing true OOD companies in their histories."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--ood_companies", type=Path, default=DEFAULT_OOD_COMPANIES
    )
    parser.add_argument("--ood_company_column", default="company")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "model")
    parser.add_argument("--none_checkpoint", type=Path, default=None)
    parser.add_argument("--kar_base_checkpoint", type=Path, default=None)
    parser.add_argument("--kar_ours_checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--max_history_length", type=int, default=20)
    parser.add_argument("--candidate_position_seed", type=int, default=42)

    # Architecture settings must match training.
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
