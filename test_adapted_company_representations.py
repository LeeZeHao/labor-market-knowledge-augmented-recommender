"""Directly evaluate learned company representations against career transitions.

The test asks whether companies people actually move to are ranked near an
OOD source company in representation space.  For each OOD source, observed
destinations receive their empirical transition shares as relevance weights
and are mixed with hard semantic non-destination companies.  The script then
reports transition-weighted NDCG@K, weighted pairwise ranking accuracy, and a
positive-versus-negative cosine-similarity margin.

Representations compared:
  * raw_factual: the unadapted factual-text embedding;
  * none_id: the DIN-none trainable company-ID embedding;
  * kar_base: the DIN KAR-base item-side HEA output;
  * kar_ours: the DIN KAR-Ours item-side residual HEA output.

The existing datasets, model code, and checkpoints are read only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import DEFAULT_DATA_DIR, CareerDatasetConfig, build_career_datasets
from main_ctr import build_model
from utils import setup_seed, str2list


ROOT = Path(__file__).resolve().parent
DEFAULT_OOD_COMPANIES = ROOT / "data" / "true_ood_companies.csv"
DEFAULT_OOD_DATA = ROOT / "data" / "ood_df_seq_reasoning.parquet"
DEFAULT_COMPANY_CATALOGUE = (
    ROOT / "data" / "company_factual_text_neg_samples.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "adapted_company_representation_test"
REPRESENTATIONS = ("raw_factual", "none_id", "kar_base", "kar_ours")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        return list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def stack_vectors(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(
            f"Company catalogue lacks {column!r}; available: {frame.columns.tolist()}"
        )
    vectors = [np.asarray(value, dtype=np.float32).reshape(-1) for value in frame[column]]
    if not vectors:
        raise ValueError("Company catalogue is empty")
    dimension = vectors[0].shape[0]
    for row_index, vector in enumerate(vectors):
        if vector.shape[0] != dimension:
            raise ValueError(
                f"Embedding row {row_index} has dimension {vector.shape[0]}; "
                f"expected {dimension}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Embedding row {row_index} contains NaN or infinity")
    return np.stack(vectors)


def longest_trajectory_per_user(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the earlier transition analysis: longest row, first on a tie."""

    working = frame.reset_index(drop=True).copy()
    working["_input_order"] = np.arange(len(working), dtype=np.int64)
    working["_company_count"] = working["companies"].map(lambda x: len(as_list(x)))
    selected = working.groupby("user_id", sort=False)["_company_count"].idxmax()
    return (
        working.loc[selected]
        .sort_values("_input_order")
        .drop(columns=["_input_order", "_company_count"])
        .reset_index(drop=True)
    )


def observed_transitions(
    frame: pd.DataFrame,
    source_companies: set[Any],
    *,
    exclude_self_transitions: bool,
) -> dict[Any, Counter[Any]]:
    """Count adjacent transitions inside each canonical company trajectory."""

    transitions: dict[Any, Counter[Any]] = defaultdict(Counter)
    for raw_history in frame["companies"]:
        history = as_list(raw_history)
        for source, destination in zip(history, history[1:]):
            if source not in source_companies or destination is None:
                continue
            if exclude_self_transitions and source == destination:
                continue
            transitions[source][destination] += 1
    return transitions


def load_model(
    args: argparse.Namespace,
    adaptor: str,
    train_dataset: Any,
    device: torch.device,
) -> torch.nn.Module:
    path = (
        getattr(args, f"{adaptor}_checkpoint")
        or args.checkpoint_dir / f"DIN_{adaptor}_company.pt"
    ).expanduser().resolve()
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
            f"Checkpoint {path} does not match the supplied architecture options."
        ) from error
    model.eval()
    return model


def adapt_factual_vectors(
    model: torch.nn.Module,
    factual_vectors: np.ndarray,
    adaptor: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(factual_vectors), batch_size),
            desc=f"Encoding {adaptor}",
        ):
            inputs = torch.from_numpy(
                factual_vectors[start : start + batch_size]
            ).to(device)
            if adaptor == "kar_base":
                converted, _ = model.convert_module._adapt(inputs, "item")
            elif adaptor == "kar_ours":
                converted, _ = model.convert_module._adapt_items(inputs)
            else:
                raise ValueError(f"Cannot factually adapt {adaptor!r}")
            outputs.append(converted.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def extract_representations(
    args: argparse.Namespace,
    catalogue: pd.DataFrame,
    factual_vectors: np.ndarray,
    train_dataset: Any,
    device: torch.device,
) -> dict[str, np.ndarray]:
    models = {
        adaptor: load_model(args, adaptor, train_dataset, device)
        for adaptor in ("none", "kar_base", "kar_ours")
    }
    company_ids = catalogue[args.company_column].tolist()
    catalogue_indices = np.asarray(
        [train_dataset.company_id_to_index[item] for item in company_ids],
        dtype=np.int64,
    )
    none_id = (
        models["none"]
        .item_embedding.weight.detach()
        .cpu()
        .numpy()[catalogue_indices]
    )
    vectors = {
        "raw_factual": factual_vectors,
        "none_id": none_id,
        "kar_base": adapt_factual_vectors(
            models["kar_base"], factual_vectors, "kar_base", args.batch_size, device
        ),
        "kar_ours": adapt_factual_vectors(
            models["kar_ours"], factual_vectors, "kar_ours", args.batch_size, device
        ),
    }
    return {name: normalize_rows(value) for name, value in vectors.items()}


def hard_negatives(
    source: Any,
    destinations: set[Any],
    negative_lookup: Mapping[Any, Any],
    valid_companies: set[Any],
    count: int,
) -> list[Any]:
    selected: list[Any] = []
    forbidden = set(destinations) | {source}
    for candidate in as_list(negative_lookup.get(source)):
        if candidate in forbidden or candidate not in valid_companies:
            continue
        selected.append(candidate)
        forbidden.add(candidate)
        if len(selected) == count:
            break
    return selected


def weighted_ndcg(
    similarities: np.ndarray,
    relevance: np.ndarray,
    k: int,
) -> float:
    cutoff = min(k, len(similarities))
    if cutoff == 0 or relevance.sum() <= 0:
        return float("nan")
    order = np.argsort(-similarities, kind="stable")[:cutoff]
    discount = np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    dcg = float(np.sum(relevance[order] / discount))
    ideal = np.sort(relevance)[::-1][:cutoff]
    ideal_dcg = float(np.sum(ideal / discount))
    return dcg / ideal_dcg if ideal_dcg > 0 else float("nan")


def representation_metrics(
    source_index: int,
    destination_indices: np.ndarray,
    destination_weights: np.ndarray,
    negative_indices: np.ndarray,
    vectors: np.ndarray,
    ks: Sequence[int],
) -> dict[str, float]:
    source = vectors[source_index]
    positive_similarity = vectors[destination_indices] @ source
    negative_similarity = vectors[negative_indices] @ source
    candidate_similarity = np.concatenate([positive_similarity, negative_similarity])
    relevance = np.concatenate(
        [destination_weights, np.zeros(len(negative_similarity), dtype=np.float64)]
    )
    comparisons = positive_similarity[:, None] - negative_similarity[None, :]
    pair_scores = (comparisons > 0).astype(np.float64) + 0.5 * (
        comparisons == 0
    )
    per_destination_accuracy = pair_scores.mean(axis=1)
    weighted_positive_similarity = float(
        np.sum(destination_weights * positive_similarity)
    )
    metrics = {
        "weighted_pairwise_accuracy": float(
            np.sum(destination_weights * per_destination_accuracy)
        ),
        "weighted_destination_similarity": weighted_positive_similarity,
        "mean_negative_similarity": float(negative_similarity.mean()),
        "weighted_similarity_margin": (
            weighted_positive_similarity - float(negative_similarity.mean())
        ),
    }
    for k in ks:
        metrics[f"ndcg_at_{k}"] = weighted_ndcg(
            candidate_similarity, relevance, int(k)
        )
    return metrics


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    if len(values) == 0 or samples <= 0:
        return float("nan"), float("nan")
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start : start + size] = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def summarize(
    detail: pd.DataFrame,
    metric_columns: Sequence[str],
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for representation in REPRESENTATIONS:
        subset = detail[detail["representation"] == representation]
        for metric in metric_columns:
            values = subset[metric].to_numpy(dtype=np.float64)
            lower, upper = bootstrap_interval(values, rng, bootstrap_samples)
            weights = subset["transition_count"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "representation": representation,
                    "metric": metric,
                    "source_company_count": len(values),
                    "macro_mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "transition_weighted_mean": float(np.average(values, weights=weights)),
                    "macro_mean_ci95_lower": lower,
                    "macro_mean_ci95_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def paired_permutation_tests(
    detail: pd.DataFrame,
    metric_columns: Sequence[str],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    wide = detail.pivot(index="source_company", columns="representation", values=metric_columns)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for comparison in ("kar_base", "none_id", "raw_factual"):
        for metric in metric_columns:
            paired = wide.loc[:, [(metric, "kar_ours"), (metric, comparison)]].dropna()
            differences = (
                paired[(metric, "kar_ours")] - paired[(metric, comparison)]
            ).to_numpy(dtype=np.float64)
            observed = float(differences.mean())
            if samples > 0 and len(differences):
                extreme = 0
                for start in range(0, samples, 256):
                    size = min(256, samples - start)
                    signs = rng.choice(
                        np.asarray([-1.0, 1.0]), size=(size, len(differences))
                    )
                    permuted = (signs * differences).mean(axis=1)
                    extreme += int((np.abs(permuted) >= abs(observed)).sum())
                p_value = (extreme + 1) / (samples + 1)
            else:
                p_value = float("nan")
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"kar_ours - {comparison}",
                    "paired_source_count": len(differences),
                    "mean_difference": observed,
                    "median_difference": float(np.median(differences)),
                    "fraction_kar_ours_higher": float((differences > 0).mean()),
                    "two_sided_sign_permutation_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    setup_seed(args.seed, args.deterministic)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    catalogue_path = args.company_catalogue.expanduser().resolve()
    catalogue = pd.read_parquet(catalogue_path)
    required_columns = {
        args.company_column,
        args.embedding_column,
        args.negative_samples_column,
    }
    missing = required_columns.difference(catalogue.columns)
    if missing:
        raise ValueError(f"Company catalogue is missing columns: {sorted(missing)}")
    if catalogue[args.company_column].duplicated().any():
        raise ValueError("Company catalogue contains duplicate company IDs")

    ood_frame = pd.read_csv(args.ood_companies.expanduser().resolve(), keep_default_na=False)
    if args.ood_company_column not in ood_frame.columns:
        raise ValueError(
            f"OOD company file lacks {args.ood_company_column!r} column"
        )
    ordered_ood = ood_frame[args.ood_company_column].tolist()
    ood_companies = set(ordered_ood)
    ood_data = pd.read_parquet(args.ood_data.expanduser().resolve())
    trajectories = longest_trajectory_per_user(ood_data)
    transitions = observed_transitions(
        trajectories,
        ood_companies,
        exclude_self_transitions=args.exclude_self_transitions,
    )

    dataset_config = CareerDatasetConfig(
        num_candidates=args.num_candidates,
        max_history_length=args.max_history_length,
        candidate_position_seed=args.candidate_position_seed,
        item_embedding_column=args.embedding_column,
    )
    datasets = build_career_datasets(args.data_dir, dataset_config)
    factual_vectors = stack_vectors(catalogue, args.embedding_column)
    vectors = extract_representations(
        args, catalogue, factual_vectors, datasets.train, device
    )

    company_ids = catalogue[args.company_column].tolist()
    company_to_index = {company: index for index, company in enumerate(company_ids)}
    valid_companies = set(company_ids)
    negative_lookup = dict(
        zip(company_ids, catalogue[args.negative_samples_column])
    )
    ks = tuple(sorted(set(int(k) for k in args.top_k)))
    detail_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source in tqdm(ordered_ood, desc="Evaluating OOD company representations"):
        if source not in company_to_index:
            skipped["source_not_in_catalogue"] += 1
            continue
        counts = Counter(
            {
                destination: count
                for destination, count in transitions.get(source, {}).items()
                if destination in company_to_index
                and (not args.exclude_self_transitions or destination != source)
            }
        )
        if not counts:
            skipped["no_valid_observed_destination"] += 1
            continue
        destinations = list(counts)
        negatives = hard_negatives(
            source,
            set(destinations),
            negative_lookup,
            valid_companies,
            args.negative_count,
        )
        if len(negatives) < args.negative_count:
            skipped["insufficient_hard_negatives"] += 1
            continue
        destination_counts = np.asarray(
            [counts[destination] for destination in destinations], dtype=np.float64
        )
        destination_weights = destination_counts / destination_counts.sum()
        destination_indices = np.asarray(
            [company_to_index[destination] for destination in destinations], dtype=np.int64
        )
        negative_indices = np.asarray(
            [company_to_index[negative] for negative in negatives], dtype=np.int64
        )
        common = {
            "source_company": source,
            "transition_count": int(destination_counts.sum()),
            "destination_company_count": len(destinations),
            "hard_negative_count": len(negatives),
            "destination_companies": json.dumps(destinations, ensure_ascii=False),
            "destination_transition_counts": json.dumps(
                destination_counts.astype(int).tolist()
            ),
            "destination_transition_shares": json.dumps(
                destination_weights.tolist()
            ),
            "hard_negative_companies": json.dumps(negatives, ensure_ascii=False),
        }
        for representation in REPRESENTATIONS:
            metrics = representation_metrics(
                company_to_index[source],
                destination_indices,
                destination_weights,
                negative_indices,
                vectors[representation],
                ks,
            )
            detail_rows.append(
                {**common, "representation": representation, **metrics}
            )

    if not detail_rows:
        raise RuntimeError(
            "No OOD source company could be evaluated. Check transition and "
            "catalogue inputs."
        )
    detail = pd.DataFrame(detail_rows)
    metric_columns = [
        *(f"ndcg_at_{k}" for k in ks),
        "weighted_pairwise_accuracy",
        "weighted_similarity_margin",
        "weighted_destination_similarity",
        "mean_negative_similarity",
    ]
    summary = summarize(
        detail, metric_columns, args.bootstrap_samples, args.seed
    )
    paired = paired_permutation_tests(
        detail, metric_columns, args.permutation_samples, args.seed + 1
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "company_representation_per_source.csv"
    summary_path = output_dir / "company_representation_summary.csv"
    paired_path = output_dir / "company_representation_paired_tests.csv"
    detail.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)

    evaluated_sources = detail["source_company"].nunique()
    print(f"Device: {device}")
    print(f"Canonical OOD trajectories: {len(trajectories):,}")
    print(f"OOD source companies evaluated: {evaluated_sources:,}")
    print(f"Self-transitions excluded: {args.exclude_self_transitions}")
    print(f"Hard negatives per source: {args.negative_count}")
    for reason, count in skipped.items():
        print(f"Skipped ({reason}): {count:,}")
    print("\nMacro means:")
    display_columns = ["representation", "metric", "macro_mean"]
    print(summary[display_columns].pivot(
        index="representation", columns="metric", values="macro_mean"
    ).round(4).to_string())
    print(f"\nPer-source results: {detail_path}")
    print(f"Summary with source-bootstrap CIs: {summary_path}")
    print(f"Paired permutation tests: {paired_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test trained DIN company representations against observed transitions."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--company_catalogue", type=Path, default=DEFAULT_COMPANY_CATALOGUE)
    parser.add_argument("--ood_companies", type=Path, default=DEFAULT_OOD_COMPANIES)
    parser.add_argument("--ood_data", type=Path, default=DEFAULT_OOD_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "model")
    parser.add_argument("--none_checkpoint", type=Path, default=None)
    parser.add_argument("--kar_base_checkpoint", type=Path, default=None)
    parser.add_argument("--kar_ours_checkpoint", type=Path, default=None)
    parser.add_argument("--company_column", default="company")
    parser.add_argument("--embedding_column", default="textual_desc_emb")
    parser.add_argument("--negative_samples_column", default="negative_samples")
    parser.add_argument("--ood_company_column", default="company")
    parser.add_argument("--negative_count", type=int, default=49)
    parser.add_argument("--top_k", type=str2list, default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--exclude_self_transitions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--permutation_samples", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
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
    args = parser.parse_args()
    if args.negative_count < 1:
        parser.error("--negative_count must be positive")
    if not args.top_k or any(k < 1 for k in args.top_k):
        parser.error("--top_k must contain positive integers")
    return args


if __name__ == "__main__":
    run(parse_args())
