"""Shared datasets for career next-role and next-company experiments.

Each row is evaluated against a deterministic 50-item candidate set
containing the ground truth and its 49 closest valid semantic negatives. The
ground truth is moved to a reproducibly randomized position for every row and
task.

The semantic-negative parquet files already store negatives from most to least
similar. This module preserves that order apart from swapping one negative
with the randomly positioned positive.

The dataset exposes both catalogue indices and factual-text embeddings so the
same input pipeline can support ID-only baselines, KAR-style models, DIN,
DIEN, DCN, and later backbones.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class CareerDatasetConfig:
    """Configuration shared by every dataset split."""

    num_candidates: int = 50
    max_history_length: int = 20
    candidate_position_seed: int = 42
    role_id_column: str = "role_k1500"
    company_id_column: str = "company"
    item_embedding_column: str = "textual_desc_emb"
    negative_samples_column: str = "negative_samples"
    target_role_column: str = "target_role"
    target_company_column: str = "target_company"
    history_roles_column: str = "roles"
    history_companies_column: str = "companies"
    history_tenures_column: str = "tenures"

    def __post_init__(self) -> None:
        if self.num_candidates < 2:
            raise ValueError(
                "num_candidates must contain one positive and at least one "
                "negative"
            )
        if self.max_history_length < 1:
            raise ValueError("max_history_length must be positive")


@dataclass(frozen=True)
class CareerDataFrames:
    """Raw frames loaded from ``career_experiments_2/data``."""

    train: pd.DataFrame
    test: pd.DataFrame
    ood: pd.DataFrame
    roles: pd.DataFrame
    companies: pd.DataFrame


@dataclass(frozen=True)
class CareerDatasets:
    """Train/test/OOD datasets sharing identical catalogue mappings."""

    train: "CareerDataset"
    test: "CareerDataset"
    ood: "CareerDataset"


def _read_parquet(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        available = "\n".join(
            f"  - {candidate.name}"
            for candidate in sorted(data_dir.glob("*.parquet"))
        )
        raise FileNotFoundError(
            f"Could not find {path}.\nAvailable parquet files:\n{available}"
        )
    return pd.read_parquet(path)


def load_career_dataframes(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> CareerDataFrames:
    """Load the three user splits and the two candidate catalogues."""

    data_dir = Path(data_dir).expanduser().resolve()
    return CareerDataFrames(
        train=_read_parquet(
            data_dir, "train_df_no_ood_seq_reasoning.parquet"
        ),
        test=_read_parquet(
            data_dir, "test_df_no_ood_seq_reasoning.parquet"
        ),
        ood=_read_parquet(data_dir, "ood_df_seq_reasoning.parquet"),
        roles=_read_parquet(
            data_dir, "role_factual_text_neg_samples.parquet"
        ),
        companies=_read_parquet(
            data_dir, "company_factual_text_neg_samples.parquet"
        ),
    )


def _as_list(value: Any) -> list[Any]:
    """Normalize parquet/list/string sequence representations."""

    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)
        return [parsed]
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


def _to_tenure(value: Any) -> float:
    """Convert tenure-in-years to float; unavailable values become zero."""

    try:
        tenure = float(value)
    except (TypeError, ValueError):
        return 0.0
    return tenure if np.isfinite(tenure) else 0.0


class CareerDataset(Dataset):
    """Candidate-grouped pointwise CTR data for role and company prediction.

    The positive receives a reproducibly randomized position for each row and
    task. Every candidate receives a binary label, allowing BCE/log-loss and
    AUC computation, while ``*_target_index`` supports grouped cross-entropy
    and Recall@K.

    Histories are chronological (oldest to newest) and left-padded: valid
    entries occupy the beginning of each fixed-length tensor.  If a history is
    longer than ``max_history_length``, only its most recent entries are kept.
    Tenures remain aligned with their role/company after catalogue filtering.
    """

    def __init__(
        self,
        user_frame: pd.DataFrame,
        role_catalogue: pd.DataFrame,
        company_catalogue: pd.DataFrame,
        config: CareerDatasetConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CareerDatasetConfig()
        self.user_frame = user_frame.reset_index(drop=True)

        self._validate_user_frame()
        self._validate_catalogue(
            role_catalogue,
            self.config.role_id_column,
            "role",
        )
        self._validate_catalogue(
            company_catalogue,
            self.config.company_id_column,
            "company",
        )

        role_ids = role_catalogue[self.config.role_id_column].tolist()
        company_ids = company_catalogue[
            self.config.company_id_column
        ].tolist()

        # Index zero is reserved for sequence padding.
        self.role_id_to_index = {
            item_id: index + 1 for index, item_id in enumerate(role_ids)
        }
        self.company_id_to_index = {
            item_id: index + 1
            for index, item_id in enumerate(company_ids)
        }
        self.role_index_to_id = {
            index: item_id
            for item_id, index in self.role_id_to_index.items()
        }
        self.company_index_to_id = {
            index: item_id
            for item_id, index in self.company_id_to_index.items()
        }
        self.role_vocab_size = len(self.role_id_to_index) + 1
        self.company_vocab_size = len(self.company_id_to_index) + 1

        embedding_column = self.config.item_embedding_column
        negative_column = self.config.negative_samples_column
        self.role_embedding_lookup = dict(
            zip(role_ids, role_catalogue[embedding_column])
        )
        self.company_embedding_lookup = dict(
            zip(company_ids, company_catalogue[embedding_column])
        )
        self.role_negatives_lookup = dict(
            zip(role_ids, role_catalogue[negative_column])
        )
        self.company_negatives_lookup = dict(
            zip(company_ids, company_catalogue[negative_column])
        )

        self.role_embedding_dim = self._infer_embedding_dim(
            self.role_embedding_lookup, "role"
        )
        self.company_embedding_dim = self._infer_embedding_dim(
            self.company_embedding_lookup, "company"
        )
        if self.role_embedding_dim != self.company_embedding_dim:
            raise ValueError(
                "Role and company factual embeddings must have the same "
                f"dimension; got {self.role_embedding_dim} and "
                f"{self.company_embedding_dim}"
            )
        self.embedding_dim = self.role_embedding_dim

        self._role_candidate_cache: dict[
            Any, tuple[list[Any], np.ndarray]
        ] = {}
        self._company_candidate_cache: dict[
            Any, tuple[list[Any], np.ndarray]
        ] = {}

    def _validate_user_frame(self) -> None:
        required = {
            self.config.target_role_column,
            self.config.target_company_column,
            self.config.history_roles_column,
            self.config.history_companies_column,
            self.config.history_tenures_column,
        }
        missing = required.difference(self.user_frame.columns)
        if missing:
            raise ValueError(
                f"User dataframe is missing columns: {sorted(missing)}"
            )

    def _validate_catalogue(
        self,
        frame: pd.DataFrame,
        id_column: str,
        entity: str,
    ) -> None:
        required = {
            id_column,
            self.config.item_embedding_column,
            self.config.negative_samples_column,
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{entity.capitalize()} catalogue is missing columns: "
                f"{sorted(missing)}"
            )
        if frame[id_column].isna().any():
            raise ValueError(
                f"{entity.capitalize()} catalogue contains null IDs"
            )
        duplicates = frame.loc[
            frame[id_column].duplicated(), id_column
        ].tolist()
        if duplicates:
            raise ValueError(
                f"{entity.capitalize()} catalogue contains duplicate IDs; "
                f"examples: {duplicates[:5]}"
            )

    @staticmethod
    def _infer_embedding_dim(
        lookup: Mapping[Any, Any], entity: str
    ) -> int:
        if not lookup:
            raise ValueError(f"{entity.capitalize()} catalogue is empty")
        first_id, first_value = next(iter(lookup.items()))
        dimension = np.asarray(first_value).reshape(-1).shape[0]
        if dimension < 1:
            raise ValueError(
                f"{entity.capitalize()} embedding for {first_id!r} is empty"
            )
        return dimension

    def __len__(self) -> int:
        return len(self.user_frame)

    def _candidate_data(
        self,
        positive_id: Any,
        negative_lookup: Mapping[Any, Any],
        embedding_lookup: Mapping[Any, Any],
        id_to_index: Mapping[Any, int],
        cache: dict[Any, tuple[list[Any], np.ndarray]],
        entity: str,
    ) -> tuple[list[Any], np.ndarray]:
        cached = cache.get(positive_id)
        if cached is not None:
            return cached
        if positive_id not in id_to_index:
            raise KeyError(
                f"{entity.capitalize()} target {positive_id!r} is absent "
                "from its catalogue"
            )

        candidate_ids = [positive_id]
        seen = {positive_id}
        for negative_id in _as_list(negative_lookup.get(positive_id)):
            if negative_id in seen or negative_id not in embedding_lookup:
                continue
            candidate_ids.append(negative_id)
            seen.add(negative_id)
            if len(candidate_ids) == self.config.num_candidates:
                break

        if len(candidate_ids) != self.config.num_candidates:
            raise ValueError(
                f"{entity.capitalize()} target {positive_id!r} has only "
                f"{len(candidate_ids) - 1} valid distinct negatives; "
                f"{self.config.num_candidates - 1} are required"
            )
        candidate_indices = np.asarray(
            [id_to_index[item_id] for item_id in candidate_ids],
            dtype=np.int64,
        )
        cache[positive_id] = (candidate_ids, candidate_indices)
        return cache[positive_id]

    def _history_data(
        self,
        history_ids: Any,
        raw_tenures: Sequence[Any],
        id_to_index: Mapping[Any, int],
        embedding_lookup: Mapping[Any, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]:
        ids = _as_list(history_ids)
        tenures = list(raw_tenures)
        if len(ids) != len(tenures):
            raise ValueError(
                "History items and tenures must have identical lengths; "
                f"got {len(ids)} and {len(tenures)}"
            )

        valid_pairs = [
            (item_id, _to_tenure(tenure))
            for item_id, tenure in zip(ids, tenures)
            if item_id in id_to_index
        ][-self.config.max_history_length :]

        length = self.config.max_history_length
        indices = np.zeros(length, dtype=np.int64)
        embeddings = np.zeros(
            (length, self.embedding_dim), dtype=np.float32
        )
        history_tenures = np.zeros(length, dtype=np.float32)
        mask = np.zeros(length, dtype=np.bool_)
        valid_ids = [item_id for item_id, _ in valid_pairs]

        if valid_pairs:
            valid_length = len(valid_pairs)
            indices[:valid_length] = [
                id_to_index[item_id] for item_id, _ in valid_pairs
            ]
            embeddings[:valid_length] = np.stack(
                [
                    np.asarray(
                        embedding_lookup[item_id], dtype=np.float32
                    )
                    for item_id, _ in valid_pairs
                ]
            )
            history_tenures[:valid_length] = [
                tenure for _, tenure in valid_pairs
            ]
            mask[:valid_length] = True

        return indices, embeddings, history_tenures, mask, valid_ids

    def _raw_tenure_history(
        self, raw_tenures: Sequence[Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        tenure_values = [
            _to_tenure(value) for value in raw_tenures
        ][-self.config.max_history_length :]
        output = np.zeros(
            self.config.max_history_length, dtype=np.float32
        )
        mask = np.zeros(self.config.max_history_length, dtype=np.bool_)
        if tenure_values:
            output[: len(tenure_values)] = tenure_values
            mask[: len(tenure_values)] = True
        return output, mask

    @staticmethod
    def _candidate_embeddings(
        candidate_ids: Sequence[Any],
        embedding_lookup: Mapping[Any, Any],
    ) -> np.ndarray:
        return np.stack(
            [
                np.asarray(
                    embedding_lookup[candidate_id], dtype=np.float32
                )
                for candidate_id in candidate_ids
            ]
        )

    def _randomize_positive_position(
        self,
        candidate_ids: Sequence[Any],
        candidate_indices: np.ndarray,
        row_index: int,
        task_offset: int,
    ) -> tuple[list[Any], np.ndarray, int]:
        """Move the positive from index zero to a reproducible random slot."""

        seed = np.random.SeedSequence(
            [
                self.config.candidate_position_seed,
                row_index,
                task_offset,
            ]
        )
        random_position = int(
            np.random.default_rng(seed).integers(
                self.config.num_candidates
            )
        )
        order = np.arange(
            self.config.num_candidates, dtype=np.int64
        )
        if random_position:
            order[0], order[random_position] = (
                order[random_position],
                order[0],
            )
        randomized_ids = [candidate_ids[i] for i in order]
        return (
            randomized_ids,
            candidate_indices[order].copy(),
            random_position,
        )

    @staticmethod
    def _optional_vector(row: pd.Series, column: str) -> torch.Tensor:
        if column not in row or row[column] is None:
            return torch.empty(0, dtype=torch.float32)
        return torch.from_numpy(
            np.array(row[column], dtype=np.float32, copy=True)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.user_frame.iloc[index]
        positive_role = row[self.config.target_role_column]
        positive_company = row[self.config.target_company_column]
        raw_tenures = _as_list(row[self.config.history_tenures_column])

        role_candidate_ids, role_candidate_indices = (
            self._candidate_data(
                positive_role,
                self.role_negatives_lookup,
                self.role_embedding_lookup,
                self.role_id_to_index,
                self._role_candidate_cache,
                "role",
            )
        )
        company_candidate_ids, company_candidate_indices = (
            self._candidate_data(
                positive_company,
                self.company_negatives_lookup,
                self.company_embedding_lookup,
                self.company_id_to_index,
                self._company_candidate_cache,
                "company",
            )
        )
        (
            role_candidate_ids,
            role_candidate_indices,
            role_target_index,
        ) = self._randomize_positive_position(
            role_candidate_ids,
            role_candidate_indices,
            index,
            task_offset=0,
        )
        (
            company_candidate_ids,
            company_candidate_indices,
            company_target_index,
        ) = self._randomize_positive_position(
            company_candidate_ids,
            company_candidate_indices,
            index,
            task_offset=1,
        )

        (
            role_history_indices,
            role_history_embeddings,
            role_history_tenures,
            role_history_mask,
            history_role_ids,
        ) = self._history_data(
            row[self.config.history_roles_column],
            raw_tenures,
            self.role_id_to_index,
            self.role_embedding_lookup,
        )
        (
            company_history_indices,
            company_history_embeddings,
            company_history_tenures,
            company_history_mask,
            history_company_ids,
        ) = self._history_data(
            row[self.config.history_companies_column],
            raw_tenures,
            self.company_id_to_index,
            self.company_embedding_lookup,
        )
        history_tenures, history_mask = self._raw_tenure_history(
            raw_tenures
        )

        role_labels = torch.zeros(
            self.config.num_candidates, dtype=torch.float32
        )
        role_labels[role_target_index] = 1.0
        company_labels = torch.zeros(
            self.config.num_candidates, dtype=torch.float32
        )
        company_labels[company_target_index] = 1.0

        return {
            "row_index": torch.tensor(index, dtype=torch.long),
            "user_id": row.get("user_id", index),
            "user_prompt_emb": self._optional_vector(row, "prompt_emb"),
            "user_reasoning_emb": self._optional_vector(
                row, "textual_reasoning_emb"
            ),
            "role_history_indices": torch.from_numpy(
                role_history_indices
            ),
            "role_history_sequence": torch.from_numpy(
                role_history_embeddings
            ),
            "role_history_tenures": torch.from_numpy(
                role_history_tenures
            ),
            "role_history_mask": torch.from_numpy(role_history_mask),
            "company_history_indices": torch.from_numpy(
                company_history_indices
            ),
            "company_history_sequence": torch.from_numpy(
                company_history_embeddings
            ),
            "company_history_tenures": torch.from_numpy(
                company_history_tenures
            ),
            "company_history_mask": torch.from_numpy(
                company_history_mask
            ),
            # Raw career-level tenure sequence, before branch catalogue
            # filtering. This is useful to future models that encode a joint
            # role-company-tenure event.
            "history_tenures": torch.from_numpy(history_tenures),
            "history_mask": torch.from_numpy(history_mask),
            "role_candidate_indices": torch.from_numpy(
                role_candidate_indices.copy()
            ),
            "role_candidate_embs": torch.from_numpy(
                self._candidate_embeddings(
                    role_candidate_ids, self.role_embedding_lookup
                )
            ),
            "role_candidate_labels": role_labels,
            "role_target_index": torch.tensor(
                role_target_index, dtype=torch.long
            ),
            "company_candidate_indices": torch.from_numpy(
                company_candidate_indices.copy()
            ),
            "company_candidate_embs": torch.from_numpy(
                self._candidate_embeddings(
                    company_candidate_ids,
                    self.company_embedding_lookup,
                )
            ),
            "company_candidate_labels": company_labels,
            "company_target_index": torch.tensor(
                company_target_index, dtype=torch.long
            ),
            # Non-tensor metadata is kept for subgroup analysis and debugging.
            "pos_role_id": positive_role,
            "pos_company_id": positive_company,
            "role_candidate_ids": list(role_candidate_ids),
            "company_candidate_ids": list(company_candidate_ids),
            "history_role_ids": history_role_ids,
            "history_company_ids": history_company_ids,
        }


def build_career_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    config: CareerDatasetConfig | None = None,
) -> CareerDatasets:
    """Build train/test/OOD datasets from the standard parquet filenames."""

    frames = load_career_dataframes(data_dir)
    config = config or CareerDatasetConfig()
    kwargs = {
        "role_catalogue": frames.roles,
        "company_catalogue": frames.companies,
        "config": config,
    }
    datasets = CareerDatasets(
        train=CareerDataset(frames.train, **kwargs),
        test=CareerDataset(frames.test, **kwargs),
        ood=CareerDataset(frames.ood, **kwargs),
    )

    # Building each split from the same catalogue should make the mappings
    # identical. Keep this explicit because differing mappings silently break
    # shared train/evaluation embedding tables.
    for evaluation_dataset in (datasets.test, datasets.ood):
        if (
            datasets.train.role_id_to_index
            != evaluation_dataset.role_id_to_index
            or datasets.train.company_id_to_index
            != evaluation_dataset.company_id_to_index
        ):
            raise RuntimeError(
                "Catalogue index mappings differ between dataset splits"
            )
    return datasets


def career_collate_fn(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack tensor fields and preserve ID/debug metadata as Python lists."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    collated: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        collated[key] = (
            torch.stack(values, dim=0)
            if isinstance(values[0], torch.Tensor)
            else values
        )
    return collated


def build_career_dataloaders(
    datasets: CareerDatasets,
    batch_size: int = 32,
    *,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create reproducible train and deterministic test/OOD loaders."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": career_collate_fn,
    }
    return {
        "train": DataLoader(
            datasets.train,
            shuffle=True,
            generator=generator,
            **common,
        ),
        "test": DataLoader(datasets.test, shuffle=False, **common),
        "ood": DataLoader(datasets.ood, shuffle=False, **common),
    }


__all__ = [
    "DEFAULT_DATA_DIR",
    "CareerDataset",
    "CareerDataFrames",
    "CareerDatasetConfig",
    "CareerDatasets",
    "build_career_dataloaders",
    "build_career_datasets",
    "career_collate_fn",
    "load_career_dataframes",
]
