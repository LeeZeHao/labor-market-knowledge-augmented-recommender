"""Verify that true-OOD companies are isolated from non-OOD experiment data.

This script is read-only. It checks that true-OOD companies do not occur as
history companies or targets in the train/test splits, and that non-OOD source
companies never use an OOD company as a negative sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OOD_COMPANIES = DATA_DIR / "true_ood_companies.csv"
DEFAULT_TRAIN = DATA_DIR / "train_df_no_ood_seq_reasoning.parquet"
DEFAULT_TEST = DATA_DIR / "test_df_no_ood_seq_reasoning.parquet"
DEFAULT_NEGATIVES = DATA_DIR / "company_factual_text_neg_samples.parquet"


def as_list(value: Any) -> list[Any]:
    """Convert a parquet sequence cell to a list without treating strings as sequences."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def report_leaks(
    *,
    label: str,
    leaks: list[tuple[int, list[Any]]],
    errors: list[str],
    example_limit: int,
) -> None:
    if not leaks:
        print(f"PASS [{label}]: no OOD companies found")
        return

    occurrence_count = sum(len(companies) for _, companies in leaks)
    distinct_companies = sorted(
        {str(company) for _, companies in leaks for company in companies}
    )
    message = (
        f"ERROR [{label}]: {len(leaks):,} rows contain {occurrence_count:,} "
        f"OOD occurrences across {len(distinct_companies):,} distinct companies"
    )
    print(message)
    errors.append(message)
    for row_index, companies in leaks[:example_limit]:
        print(f"  row={row_index}: {companies}")


def verify_split(
    *,
    path: Path,
    label: str,
    ood_companies: set[Any],
    history_column: str,
    target_column: str,
    errors: list[str],
    example_limit: int,
) -> None:
    try:
        frame = pd.read_parquet(path, columns=[history_column, target_column])
    except Exception as error:
        message = f"ERROR [{label} schema/read]: {error}"
        print(message)
        errors.append(message)
        return

    target_leaks = [
        (int(row_index), [target])
        for row_index, target in frame[target_column].items()
        if target in ood_companies
    ]
    history_leaks: list[tuple[int, list[Any]]] = []
    for row_index, history in frame[history_column].items():
        matches = [company for company in as_list(history) if company in ood_companies]
        if matches:
            history_leaks.append((int(row_index), matches))

    report_leaks(
        label=f"{label} targets",
        leaks=target_leaks,
        errors=errors,
        example_limit=example_limit,
    )
    report_leaks(
        label=f"{label} histories",
        leaks=history_leaks,
        errors=errors,
        example_limit=example_limit,
    )


def verify_negative_samples(
    *,
    path: Path,
    ood_companies: set[Any],
    company_column: str,
    negative_column: str,
    errors: list[str],
    example_limit: int,
) -> None:
    try:
        frame = pd.read_parquet(path, columns=[company_column, negative_column])
    except Exception as error:
        message = f"ERROR [negative-sample schema/read]: {error}"
        print(message)
        errors.append(message)
        return

    leaks: list[tuple[int, list[Any]]] = []
    for row_index, row in frame.iterrows():
        source_company = row[company_column]
        if source_company in ood_companies:
            continue
        matches = [
            company
            for company in as_list(row[negative_column])
            if company in ood_companies
        ]
        if matches:
            leaks.append((int(row_index), matches))

    report_leaks(
        label="non-OOD source negative samples",
        leaks=leaks,
        errors=errors,
        example_limit=example_limit,
    )


def run(args: argparse.Namespace) -> int:
    errors: list[str] = []
    required_paths = {
        "OOD company CSV": args.ood_companies,
        "train parquet": args.train,
        "test parquet": args.test,
        "company-negative parquet": args.negatives,
    }
    for label, path in required_paths.items():
        if not path.is_file():
            message = f"ERROR [{label}]: file does not exist: {path}"
            print(message)
            errors.append(message)
    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} error(s)")
        return 1

    try:
        ood_frame = pd.read_csv(args.ood_companies, keep_default_na=False)
    except Exception as error:
        print(f"ERROR [OOD company CSV read]: {error}")
        return 1
    if args.ood_company_column not in ood_frame.columns:
        print(
            f"ERROR [OOD company CSV schema]: missing column "
            f"{args.ood_company_column!r}"
        )
        return 1

    ood_companies = set(ood_frame[args.ood_company_column])
    if not ood_companies:
        print("ERROR [OOD company CSV]: the OOD company set is empty")
        return 1
    print(f"Loaded {len(ood_companies):,} distinct OOD companies")

    verify_split(
        path=args.train,
        label="train",
        ood_companies=ood_companies,
        history_column=args.history_column,
        target_column=args.target_column,
        errors=errors,
        example_limit=args.example_limit,
    )
    verify_split(
        path=args.test,
        label="test",
        ood_companies=ood_companies,
        history_column=args.history_column,
        target_column=args.target_column,
        errors=errors,
        example_limit=args.example_limit,
    )
    verify_negative_samples(
        path=args.negatives,
        ood_companies=ood_companies,
        company_column=args.company_column,
        negative_column=args.negative_column,
        errors=errors,
        example_limit=args.example_limit,
    )

    if errors:
        print(f"\nVERIFICATION FAILED: {len(errors)} check(s) failed")
        return 1
    print("\nVERIFICATION PASSED: no OOD leakage was found")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood-companies", type=Path, default=DEFAULT_OOD_COMPANIES)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--ood-company-column", default="company")
    parser.add_argument("--history-column", default="companies")
    parser.add_argument("--target-column", default="target_company")
    parser.add_argument("--company-column", default="company")
    parser.add_argument("--negative-column", default="negative_samples")
    parser.add_argument("--example-limit", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
