"""Regenerate company hard negatives without leaking OOD companies.

Cosine similarity over ``textual_desc_emb`` determines the nearest companies.
For a non-OOD source company, OOD companies are excluded from its negative
pool. For an OOD source company, every other catalogue company is eligible.
The input parquet is replaced atomically by default.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = ROOT / "data" / "company_factual_text_neg_samples.parquet"
DEFAULT_OOD_COMPANIES = ROOT / "data" / "true_ood_companies.csv"


def load_embeddings(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        embeddings = np.stack(
            [np.asarray(value, dtype=np.float32) for value in frame[column]]
        )
    except ValueError as error:
        raise ValueError(
            f"{column!r} must contain equal-length embedding vectors"
        ) from error
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError(
            f"{column!r} must form a non-empty 2D matrix, got {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{column!r} contains NaN or infinite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        bad_rows = np.flatnonzero(norms[:, 0] <= 0)[:10].tolist()
        raise ValueError(f"Zero-norm embeddings found at rows {bad_rows}")
    return embeddings / norms


def nearest_companies(
    *,
    source_indices: np.ndarray,
    candidate_indices: np.ndarray,
    embeddings: np.ndarray,
    company_names: np.ndarray,
    negative_count: int,
    chunk_size: int,
    description: str,
    output: list[np.ndarray | None],
) -> None:
    """Find exact cosine nearest neighbours for one eligibility group."""

    if source_indices.size == 0:
        return
    if candidate_indices.size - 1 < negative_count:
        raise ValueError(
            f"{description} has only {candidate_indices.size - 1} eligible "
            f"negatives, but {negative_count} were requested"
        )

    candidate_embeddings = embeddings[candidate_indices]
    candidate_position = np.full(len(company_names), -1, dtype=np.int64)
    candidate_position[candidate_indices] = np.arange(candidate_indices.size)

    starts = range(0, source_indices.size, chunk_size)
    for start in tqdm(
        starts,
        total=(source_indices.size + chunk_size - 1) // chunk_size,
        desc=description,
    ):
        sources = source_indices[start : start + chunk_size]
        similarities = embeddings[sources] @ candidate_embeddings.T

        self_positions = candidate_position[sources]
        valid_self = self_positions >= 0
        similarities[
            np.arange(sources.size)[valid_self], self_positions[valid_self]
        ] = -np.inf

        partition_start = similarities.shape[1] - negative_count
        nearest_positions = np.argpartition(
            similarities, partition_start, axis=1
        )[:, partition_start:]
        nearest_scores = np.take_along_axis(
            similarities, nearest_positions, axis=1
        )
        descending_order = np.argsort(
            -nearest_scores, axis=1, kind="stable"
        )
        nearest_positions = np.take_along_axis(
            nearest_positions, descending_order, axis=1
        )
        nearest_global_indices = candidate_indices[nearest_positions]

        for source_index, negative_indices in zip(
            sources.tolist(), nearest_global_indices
        ):
            output[source_index] = company_names[negative_indices].copy()


def write_parquet_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    ood_path = args.ood_companies.expanduser().resolve()

    frame = pd.read_parquet(input_path)
    required_columns = {args.company_column, args.embedding_column}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input parquet is missing columns: {sorted(missing)}")
    if frame[args.company_column].isna().any():
        raise ValueError(f"{args.company_column!r} contains null values")
    if frame[args.company_column].duplicated().any():
        raise ValueError(f"{args.company_column!r} contains duplicate values")
    if args.negative_count < 1:
        raise ValueError("negative_count must be positive")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    ood_frame = pd.read_csv(ood_path, keep_default_na=False)
    if args.ood_company_column not in ood_frame.columns:
        raise ValueError(
            f"OOD CSV is missing column {args.ood_company_column!r}"
        )
    requested_ood = set(ood_frame[args.ood_company_column])
    company_names = frame[args.company_column].to_numpy(dtype=object)
    catalogue_companies = set(company_names.tolist())
    missing_ood = requested_ood.difference(catalogue_companies)
    if missing_ood:
        examples = sorted(map(str, missing_ood))[:10]
        raise ValueError(
            f"{len(missing_ood)} OOD companies are absent from the catalogue; "
            f"examples: {examples}"
        )

    embeddings = load_embeddings(frame, args.embedding_column)
    is_ood = np.fromiter(
        (company in requested_ood for company in company_names),
        dtype=np.bool_,
        count=len(company_names),
    )
    all_indices = np.arange(len(frame), dtype=np.int64)
    in_distribution_indices = all_indices[~is_ood]
    ood_indices = all_indices[is_ood]
    negatives: list[np.ndarray | None] = [None] * len(frame)

    nearest_companies(
        source_indices=in_distribution_indices,
        candidate_indices=in_distribution_indices,
        embeddings=embeddings,
        company_names=company_names,
        negative_count=args.negative_count,
        chunk_size=args.chunk_size,
        description="Non-OOD sources",
        output=negatives,
    )
    nearest_companies(
        source_indices=ood_indices,
        candidate_indices=all_indices,
        embeddings=embeddings,
        company_names=company_names,
        negative_count=args.negative_count,
        chunk_size=args.chunk_size,
        description="OOD sources",
        output=negatives,
    )
    if any(value is None for value in negatives):
        raise RuntimeError("Negative generation did not produce every row")

    generated = [value for value in negatives if value is not None]
    ood_violations = sum(
        any(negative in requested_ood for negative in generated[index])
        for index in in_distribution_indices
    )
    self_violations = sum(
        company_names[index] in generated[index] for index in all_indices
    )
    if ood_violations or self_violations:
        raise RuntimeError(
            f"Generated invalid negatives: {ood_violations} non-OOD rows contain "
            f"OOD negatives; {self_violations} rows contain themselves"
        )

    frame[args.negative_column] = pd.Series(generated, dtype=object)
    write_parquet_atomically(frame, output_path)
    print(f"Companies: {len(frame):,}")
    print(f"OOD companies: {is_ood.sum():,}")
    print(f"Negatives per company: {args.negative_count:,}")
    print("Non-OOD source rows containing OOD negatives: 0")
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--ood-companies", type=Path, default=DEFAULT_OOD_COMPANIES
    )
    parser.add_argument("--company-column", default="company")
    parser.add_argument("--ood-company-column", default="company")
    parser.add_argument("--embedding-column", default="textual_desc_emb")
    parser.add_argument("--negative-column", default="negative_samples")
    parser.add_argument("--negative-count", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
