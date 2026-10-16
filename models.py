"""DIN backbone with selectable career knowledge adaptors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from layers import (
    AttentionPoolingLayer,
    CrossNetV1,
    CrossNetV2,
    HEA,
    InterestEvolving,
    InterestExtractor,
    KAROursAdaptor,
    MLP,
)


class DeepInterestNet(nn.Module):
    """One task-specific DIN model with ``none``, KAR-base, or KAR-Ours."""

    def __init__(
        self,
        *,
        prediction_task: str,
        adaptor_name: str,
        vocab_size: int,
        factual_dim: int,
        embedding_dim: int = 32,
        tenure_hidden_dim: int = 16,
        attention_architecture: Sequence[int] = (32, 16),
        final_mlp_architecture: Sequence[int] = (200, 80),
        expert_architecture: Sequence[int] = (128, 32),
        shared_expert_count: int = 2,
        specific_expert_count: int = 6,
        history_specific_expert_count: int = 2,
        residual_expert_hidden_dim: int = 32,
        residual_initial_alpha: float = 0.01,
        history_decay_rate: float = 0.3,
        dropout: float = 0.0,
        convert_dropout: float = 0.0,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if prediction_task not in {"role", "company"}:
            raise ValueError("prediction_task must be 'role' or 'company'")
        if adaptor_name not in {"none", "kar_base", "kar_ours"}:
            raise ValueError("adaptor_name must be none, kar_base, or kar_ours")
        if expert_architecture[-1] != embedding_dim:
            raise ValueError(
                "The KAR-base HEA output must equal embedding_dim: "
                f"{expert_architecture[-1]} != {embedding_dim}"
            )
        self.prediction_task = prediction_task
        self.adaptor_name = adaptor_name
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if adaptor_name == "kar_base":
            self.convert_module: nn.Module | None = HEA(
                input_dim=factual_dim,
                expert_architecture=expert_architecture,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                dropout=convert_dropout,
            )
        elif adaptor_name == "kar_ours":
            self.convert_module = KAROursAdaptor(
                input_dim=factual_dim,
                output_dim=embedding_dim,
                expert_hidden_dim=residual_expert_hidden_dim,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                history_specific_expert_count=history_specific_expert_count,
                dropout=residual_dropout,
                initial_alpha=residual_initial_alpha,
                history_decay_rate=history_decay_rate,
            )
        else:
            self.convert_module = None

        self.attention_embedding_dim = (
            embedding_dim * 2 if adaptor_name == "kar_ours" else embedding_dim
        )
        self.tenure_encoder = nn.Sequential(
            nn.Linear(1, tenure_hidden_dim),
            nn.ReLU(),
            nn.Linear(tenure_hidden_dim, embedding_dim),
        )
        self.history_map_layer = nn.Linear(
            self.attention_embedding_dim + embedding_dim,
            self.attention_embedding_dim,
        )
        self.attention_net = AttentionPoolingLayer(
            self.attention_embedding_dim, dropout, attention_architecture
        )
        final_input_dim = self.attention_embedding_dim * 2
        if adaptor_name == "kar_base":
            final_input_dim += embedding_dim * 2
        elif adaptor_name == "kar_ours":
            final_input_dim += embedding_dim * 3
        self.final_mlp = MLP(final_mlp_architecture, final_input_dim, dropout)
        self.final_fc = nn.Linear(self.final_mlp.output_dim, 1)

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        task = self.prediction_task
        candidate_indices = inputs[f"{task}_candidate_indices"]
        history_indices = inputs[f"{task}_history_indices"]
        history_mask = inputs[f"{task}_history_mask"]
        history_tenures = inputs[f"{task}_history_tenures"]
        query_ids = self.item_embedding(candidate_indices)
        history_ids = self.item_embedding(history_indices)
        result: dict[str, Any] = {}

        user_knowledge: torch.Tensor | None = None
        item_knowledge: torch.Tensor | None = None
        if self.adaptor_name == "kar_base":
            if not isinstance(self.convert_module, HEA):
                raise RuntimeError("KAR-base HEA is not initialized")
            user_knowledge, item_knowledge, user_gates, item_gates = (
                self.convert_module(
                    inputs["user_reasoning_emb"],
                    inputs[f"{task}_candidate_embs"],
                )
            )
            query = query_ids
            history_items = history_ids
            result.update(
                user_gate_weights=user_gates,
                item_gate_weights=item_gates,
            )
        elif self.adaptor_name == "kar_ours":
            if not isinstance(self.convert_module, KAROursAdaptor):
                raise RuntimeError("KAR-Ours adaptor is not initialized")
            knowledge = self.convert_module(
                reasoning=inputs["user_reasoning_emb"],
                role_history=inputs["role_history_sequence"],
                role_history_mask=inputs["role_history_mask"],
                company_history=inputs["company_history_sequence"],
                company_history_mask=inputs["company_history_mask"],
                candidates=inputs[f"{task}_candidate_embs"],
                behaviors=inputs[f"{task}_history_sequence"],
            )
            user_knowledge = knowledge["user"]
            item_knowledge = knowledge["candidate"]
            query = torch.cat([item_knowledge, query_ids], dim=-1)
            history_items = torch.cat(
                [knowledge["behavior"], history_ids], dim=-1
            )
            result.update(
                user_gate_weights=knowledge["user_gate_weights"],
                item_gate_weights=knowledge["item_gate_weights"],
                behavior_gate_weights=knowledge["behavior_gate_weights"],
            )
        else:
            query = query_ids
            history_items = history_ids

        tenure_features = torch.log1p(history_tenures.abs()).unsqueeze(-1)
        tenure_embeddings = self.tenure_encoder(tenure_features)
        tenure_embeddings = tenure_embeddings * history_mask.unsqueeze(-1).to(
            tenure_embeddings.dtype
        )
        history = self.history_map_layer(
            torch.cat([history_items, tenure_embeddings], dim=-1)
        )
        user_interest, attention_scores = self.attention_net(
            query, history, history_mask
        )
        final_features = [user_interest, query]
        if self.adaptor_name == "kar_base":
            if user_knowledge is None or item_knowledge is None:
                raise RuntimeError("KAR-base knowledge is missing")
            final_features.extend(
                [
                    user_knowledge.unsqueeze(1).expand(
                        -1, candidate_indices.shape[1], -1
                    ),
                    item_knowledge,
                ]
            )
        elif self.adaptor_name == "kar_ours":
            if user_knowledge is None or item_knowledge is None:
                raise RuntimeError("KAR-Ours knowledge is missing")
            final_features.append(
                user_knowledge.unsqueeze(1).expand(
                    -1, candidate_indices.shape[1], -1
                )
            )
        logits = self.final_fc(
            self.final_mlp(torch.cat(final_features, dim=-1))
        ).squeeze(-1)
        result.update(logits=logits, attention_scores=attention_scores)
        return result


class DeepFM(nn.Module):
    """KAR-style DeepFM with selectable career knowledge adaptors."""

    def __init__(
        self,
        *,
        prediction_task: str,
        adaptor_name: str,
        vocab_size: int,
        factual_dim: int,
        embedding_dim: int = 32,
        tenure_hidden_dim: int = 16,
        deep_architecture: Sequence[int] = (200, 80),
        expert_architecture: Sequence[int] = (128, 32),
        shared_expert_count: int = 2,
        specific_expert_count: int = 6,
        history_specific_expert_count: int = 2,
        residual_expert_hidden_dim: int = 32,
        residual_initial_alpha: float = 0.01,
        history_decay_rate: float = 0.3,
        dropout: float = 0.0,
        convert_dropout: float = 0.0,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if prediction_task not in {"role", "company"}:
            raise ValueError("prediction_task must be 'role' or 'company'")
        if adaptor_name not in {"none", "kar_base", "kar_ours"}:
            raise ValueError("adaptor_name must be none, kar_base, or kar_ours")
        if expert_architecture[-1] != embedding_dim:
            raise ValueError(
                "The HEA output must equal embedding_dim: "
                f"{expert_architecture[-1]} != {embedding_dim}"
            )
        self.prediction_task = prediction_task
        self.adaptor_name = adaptor_name
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.fm_first_item_embedding = nn.Embedding(
            vocab_size, 1, padding_idx=0
        )
        nn.init.xavier_normal_(self.fm_first_item_embedding.weight)
        with torch.no_grad():
            self.fm_first_item_embedding.weight[0].zero_()

        if adaptor_name == "kar_base":
            self.convert_module: nn.Module | None = HEA(
                input_dim=factual_dim,
                expert_architecture=expert_architecture,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                dropout=convert_dropout,
            )
        elif adaptor_name == "kar_ours":
            self.convert_module = KAROursAdaptor(
                input_dim=factual_dim,
                output_dim=embedding_dim,
                expert_hidden_dim=residual_expert_hidden_dim,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                history_specific_expert_count=history_specific_expert_count,
                dropout=residual_dropout,
                initial_alpha=residual_initial_alpha,
                history_decay_rate=history_decay_rate,
            )
        else:
            self.convert_module = None

        self.tenure_encoder = nn.Sequential(
            nn.Linear(1, tenure_hidden_dim),
            nn.ReLU(),
            nn.Linear(tenure_hidden_dim, embedding_dim),
        )
        if adaptor_name == "none":
            self.deep_input_dim = embedding_dim * 3
            self.fm_field_count = 3
            dense_feature_dim = 0
        elif adaptor_name == "kar_base":
            self.deep_input_dim = embedding_dim * 5
            self.fm_field_count = 3
            dense_feature_dim = embedding_dim * 2
        else:
            self.deep_input_dim = embedding_dim * 8
            self.fm_field_count = 5
            dense_feature_dim = embedding_dim * 4
        self.dense_feature_dim = dense_feature_dim
        self.fm_first_dense_weight = nn.Parameter(
            torch.rand(dense_feature_dim, 1)
        )
        self.deep_part = MLP(
            deep_architecture, self.deep_input_dim, dropout
        )
        self.deep_output = nn.Linear(self.deep_part.output_dim, 1)

    @staticmethod
    def _fixed_history_mean(
        field: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        masked = field * mask.unsqueeze(-1).to(field.dtype)
        return masked.mean(dim=1)

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        task = self.prediction_task
        candidate_indices = inputs[f"{task}_candidate_indices"]
        history_indices = inputs[f"{task}_history_indices"]
        history_mask = inputs[f"{task}_history_mask"]
        history_tenures = inputs[f"{task}_history_tenures"]
        candidate_ids = self.item_embedding(candidate_indices)
        history_ids = self.item_embedding(history_indices)
        result: dict[str, Any] = {}

        user_knowledge: torch.Tensor | None = None
        item_knowledge: torch.Tensor | None = None
        behavior_knowledge: torch.Tensor | None = None
        if self.adaptor_name == "kar_base":
            if not isinstance(self.convert_module, HEA):
                raise RuntimeError("KAR-base HEA is not initialized")
            user_knowledge, item_knowledge, user_gates, item_gates = (
                self.convert_module(
                    inputs["user_reasoning_emb"],
                    inputs[f"{task}_candidate_embs"],
                )
            )
            result.update(
                user_gate_weights=user_gates,
                item_gate_weights=item_gates,
            )
        elif self.adaptor_name == "kar_ours":
            if not isinstance(self.convert_module, KAROursAdaptor):
                raise RuntimeError("KAR-Ours adaptor is not initialized")
            knowledge = self.convert_module(
                reasoning=inputs["user_reasoning_emb"],
                role_history=inputs["role_history_sequence"],
                role_history_mask=inputs["role_history_mask"],
                company_history=inputs["company_history_sequence"],
                company_history_mask=inputs["company_history_mask"],
                candidates=inputs[f"{task}_candidate_embs"],
                behaviors=inputs[f"{task}_history_sequence"],
            )
            user_knowledge = knowledge["user"]
            item_knowledge = knowledge["candidate"]
            behavior_knowledge = knowledge["behavior"]
            result.update(
                user_gate_weights=knowledge["user_gate_weights"],
                item_gate_weights=knowledge["item_gate_weights"],
                behavior_gate_weights=knowledge["behavior_gate_weights"],
            )

        tenure_features = torch.log1p(history_tenures.abs()).unsqueeze(-1)
        tenure_embeddings = self.tenure_encoder(tenure_features)
        history_id_mean = self._fixed_history_mean(
            history_ids, history_mask
        )
        tenure_mean = self._fixed_history_mean(
            tenure_embeddings, history_mask
        )
        candidate_count = candidate_indices.shape[1]

        def repeat_user(field: torch.Tensor) -> torch.Tensor:
            return field.unsqueeze(1).expand(-1, candidate_count, -1)

        fm_fields = [
            candidate_ids,
            repeat_user(history_id_mean),
            repeat_user(tenure_mean),
        ]
        deep_fields = list(fm_fields)
        dense_features: torch.Tensor | None = None

        if self.adaptor_name == "kar_base":
            if user_knowledge is None or item_knowledge is None:
                raise RuntimeError("KAR-base knowledge is missing")
            repeated_user = repeat_user(user_knowledge)
            dense_features = torch.cat(
                [repeated_user, item_knowledge], dim=-1
            )
            deep_fields.extend([repeated_user, item_knowledge])
        elif self.adaptor_name == "kar_ours":
            if (
                user_knowledge is None
                or item_knowledge is None
                or behavior_knowledge is None
            ):
                raise RuntimeError("KAR-Ours knowledge is missing")
            behavior_mean = self._fixed_history_mean(
                behavior_knowledge, history_mask
            )
            repeated_behavior = repeat_user(behavior_mean)
            repeated_user = repeat_user(user_knowledge)
            fm_fields = [
                candidate_ids,
                item_knowledge,
                repeat_user(history_id_mean),
                repeated_behavior,
                repeat_user(tenure_mean),
            ]
            deep_fields = [*fm_fields, repeated_user]
            dense_features = torch.cat(
                [repeated_user, item_knowledge], dim=-1
            )

        deep_input = torch.cat(deep_fields, dim=-1)
        if deep_input.shape[-1] != self.deep_input_dim:
            raise RuntimeError(
                f"DeepFM input width {deep_input.shape[-1]} does not match "
                f"configured width {self.deep_input_dim}"
            )
        field_tensor = torch.stack(fm_fields, dim=-2)
        if field_tensor.shape[-2] != self.fm_field_count:
            raise RuntimeError(
                f"DeepFM field count {field_tensor.shape[-2]} does not match "
                f"configured count {self.fm_field_count}"
            )

        first_order = self.fm_first_item_embedding(
            candidate_indices
        ).squeeze(-1)
        if dense_features is not None:
            if dense_features.shape[-1] != self.dense_feature_dim:
                raise RuntimeError("DeepFM dense feature width is inconsistent")
            first_order = first_order + torch.matmul(
                dense_features, self.fm_first_dense_weight
            ).squeeze(-1)

        square_of_sum = field_tensor.sum(dim=-2).pow(2)
        sum_of_square = field_tensor.pow(2).sum(dim=-2)
        second_order = 0.5 * (
            square_of_sum - sum_of_square
        ).sum(dim=-1)

        flattened_deep = deep_input.reshape(-1, self.deep_input_dim)
        deep_logit = self.deep_output(
            self.deep_part(flattened_deep)
        ).reshape(candidate_indices.shape[0], candidate_count)
        result["logits"] = first_order + second_order + deep_logit
        return result


class DCNv1(nn.Module):
    """KAR-style DCNv1 with selectable career knowledge adaptors."""

    cross_network_type: type[nn.Module] = CrossNetV1

    def __init__(
        self,
        *,
        prediction_task: str,
        adaptor_name: str,
        vocab_size: int,
        factual_dim: int,
        embedding_dim: int = 32,
        tenure_hidden_dim: int = 16,
        deep_architecture: Sequence[int] = (200, 80),
        cross_layer_count: int = 3,
        final_mlp_architecture: Sequence[int] = (200, 80),
        expert_architecture: Sequence[int] = (128, 32),
        shared_expert_count: int = 2,
        specific_expert_count: int = 6,
        history_specific_expert_count: int = 2,
        residual_expert_hidden_dim: int = 32,
        residual_initial_alpha: float = 0.01,
        history_decay_rate: float = 0.3,
        dropout: float = 0.0,
        convert_dropout: float = 0.0,
        residual_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if prediction_task not in {"role", "company"}:
            raise ValueError("prediction_task must be 'role' or 'company'")
        if adaptor_name not in {"none", "kar_base", "kar_ours"}:
            raise ValueError("adaptor_name must be none, kar_base, or kar_ours")
        if expert_architecture[-1] != embedding_dim:
            raise ValueError(
                "The HEA output must equal embedding_dim: "
                f"{expert_architecture[-1]} != {embedding_dim}"
            )
        self.prediction_task = prediction_task
        self.adaptor_name = adaptor_name
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        if adaptor_name == "kar_base":
            self.convert_module: nn.Module | None = HEA(
                input_dim=factual_dim,
                expert_architecture=expert_architecture,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                dropout=convert_dropout,
            )
        elif adaptor_name == "kar_ours":
            self.convert_module = KAROursAdaptor(
                input_dim=factual_dim,
                output_dim=embedding_dim,
                expert_hidden_dim=residual_expert_hidden_dim,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                history_specific_expert_count=history_specific_expert_count,
                dropout=residual_dropout,
                initial_alpha=residual_initial_alpha,
                history_decay_rate=history_decay_rate,
            )
        else:
            self.convert_module = None

        self.tenure_encoder = nn.Sequential(
            nn.Linear(1, tenure_hidden_dim),
            nn.ReLU(),
            nn.Linear(tenure_hidden_dim, embedding_dim),
        )
        if adaptor_name == "none":
            self.model_input_dim = embedding_dim * 3
        elif adaptor_name == "kar_base":
            self.model_input_dim = embedding_dim * 5
        else:
            # candidate64 + mean behavior96 + adapted user96
            self.model_input_dim = embedding_dim * 8

        self.cross_net = self.cross_network_type(
            self.model_input_dim, cross_layer_count
        )
        self.deep_net = MLP(
            deep_architecture, self.model_input_dim, dropout
        )
        self.final_mlp = MLP(
            final_mlp_architecture,
            self.model_input_dim + self.deep_net.output_dim,
            dropout,
        )
        self.final_fc = nn.Linear(self.final_mlp.output_dim, 1)

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        task = self.prediction_task
        candidate_indices = inputs[f"{task}_candidate_indices"]
        history_indices = inputs[f"{task}_history_indices"]
        history_mask = inputs[f"{task}_history_mask"]
        history_tenures = inputs[f"{task}_history_tenures"]
        candidate_ids = self.item_embedding(candidate_indices)
        history_ids = self.item_embedding(history_indices)
        result: dict[str, Any] = {}

        user_knowledge: torch.Tensor | None = None
        item_knowledge: torch.Tensor | None = None
        if self.adaptor_name == "kar_base":
            if not isinstance(self.convert_module, HEA):
                raise RuntimeError("KAR-base HEA is not initialized")
            user_knowledge, item_knowledge, user_gates, item_gates = (
                self.convert_module(
                    inputs["user_reasoning_emb"],
                    inputs[f"{task}_candidate_embs"],
                )
            )
            candidate_features = candidate_ids
            history_items = history_ids
            result.update(
                user_gate_weights=user_gates,
                item_gate_weights=item_gates,
            )
        elif self.adaptor_name == "kar_ours":
            if not isinstance(self.convert_module, KAROursAdaptor):
                raise RuntimeError("KAR-Ours adaptor is not initialized")
            knowledge = self.convert_module(
                reasoning=inputs["user_reasoning_emb"],
                role_history=inputs["role_history_sequence"],
                role_history_mask=inputs["role_history_mask"],
                company_history=inputs["company_history_sequence"],
                company_history_mask=inputs["company_history_mask"],
                candidates=inputs[f"{task}_candidate_embs"],
                behaviors=inputs[f"{task}_history_sequence"],
            )
            user_knowledge = knowledge["user"]
            item_knowledge = knowledge["candidate"]
            candidate_features = torch.cat(
                [item_knowledge, candidate_ids], dim=-1
            )
            history_items = torch.cat(
                [knowledge["behavior"], history_ids], dim=-1
            )
            result.update(
                user_gate_weights=knowledge["user_gate_weights"],
                item_gate_weights=knowledge["item_gate_weights"],
                behavior_gate_weights=knowledge["behavior_gate_weights"],
            )
        else:
            candidate_features = candidate_ids
            history_items = history_ids

        tenure_features = torch.log1p(history_tenures.abs()).unsqueeze(-1)
        tenure_embeddings = self.tenure_encoder(tenure_features)
        history_features = torch.cat(
            [history_items, tenure_embeddings], dim=-1
        )
        history_features = history_features * history_mask.unsqueeze(-1).to(
            history_features.dtype
        )
        # Deliberately match KAR DCNv1's mean over the fixed history width.
        history_mean = history_features.mean(dim=1)
        candidate_count = candidate_indices.shape[1]
        repeated_history = history_mean.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        feature_parts = [candidate_features, repeated_history]

        if self.adaptor_name == "kar_base":
            if user_knowledge is None or item_knowledge is None:
                raise RuntimeError("KAR-base knowledge is missing")
            feature_parts.extend(
                [
                    user_knowledge.unsqueeze(1).expand(
                        -1, candidate_count, -1
                    ),
                    item_knowledge,
                ]
            )
        elif self.adaptor_name == "kar_ours":
            if user_knowledge is None:
                raise RuntimeError("KAR-Ours user knowledge is missing")
            feature_parts.append(
                user_knowledge.unsqueeze(1).expand(
                    -1, candidate_count, -1
                )
            )

        features = torch.cat(feature_parts, dim=-1)
        if features.shape[-1] != self.model_input_dim:
            raise RuntimeError(
                f"DCNv1 feature width {features.shape[-1]} does not match "
                f"configured width {self.model_input_dim}"
            )
        flattened = features.reshape(-1, self.model_input_dim)
        deep_part = self.deep_net(flattened)
        cross_part = self.cross_net(flattened)
        combined = torch.cat([deep_part, cross_part], dim=-1)
        logits = self.final_fc(self.final_mlp(combined)).reshape(
            candidate_indices.shape[0], candidate_count
        )
        result["logits"] = logits
        return result


class DCNv2(DCNv1):
    """KAR DCNv2: DCNv1 feature pipeline with full-matrix crosses."""

    cross_network_type: type[nn.Module] = CrossNetV2


class DIEN(nn.Module):
    """KAR-repository DIEN adapted to grouped career candidates.

    The first GRU extracts an interest sequence. The second recurrent stage
    evolves that sequence relative to every candidate using GRU, AIGRU, AGRU,
    or AUGRU. Knowledge insertion for each adaptor mirrors ``DeepInterestNet``.
    """

    def __init__(
        self,
        *,
        prediction_task: str,
        adaptor_name: str,
        vocab_size: int,
        factual_dim: int,
        embedding_dim: int = 32,
        tenure_hidden_dim: int = 16,
        attention_architecture: Sequence[int] = (32, 16),
        final_mlp_architecture: Sequence[int] = (200, 80),
        expert_architecture: Sequence[int] = (128, 32),
        shared_expert_count: int = 2,
        specific_expert_count: int = 6,
        history_specific_expert_count: int = 2,
        residual_expert_hidden_dim: int = 32,
        residual_initial_alpha: float = 0.01,
        history_decay_rate: float = 0.3,
        dropout: float = 0.0,
        convert_dropout: float = 0.0,
        residual_dropout: float = 0.1,
        dien_gru: str = "AUGRU",
    ) -> None:
        super().__init__()
        if prediction_task not in {"role", "company"}:
            raise ValueError("prediction_task must be 'role' or 'company'")
        if adaptor_name not in {"none", "kar_base", "kar_ours"}:
            raise ValueError("adaptor_name must be none, kar_base, or kar_ours")
        if expert_architecture[-1] != embedding_dim:
            raise ValueError(
                "The HEA output must equal embedding_dim: "
                f"{expert_architecture[-1]} != {embedding_dim}"
            )
        self.prediction_task = prediction_task
        self.adaptor_name = adaptor_name
        self.embedding_dim = embedding_dim
        self.dien_gru = dien_gru.upper()
        self.item_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        if adaptor_name == "kar_base":
            self.convert_module: nn.Module | None = HEA(
                input_dim=factual_dim,
                expert_architecture=expert_architecture,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                dropout=convert_dropout,
            )
        elif adaptor_name == "kar_ours":
            self.convert_module = KAROursAdaptor(
                input_dim=factual_dim,
                output_dim=embedding_dim,
                expert_hidden_dim=residual_expert_hidden_dim,
                shared_expert_count=shared_expert_count,
                specific_expert_count=specific_expert_count,
                history_specific_expert_count=history_specific_expert_count,
                dropout=residual_dropout,
                initial_alpha=residual_initial_alpha,
                history_decay_rate=history_decay_rate,
            )
        else:
            self.convert_module = None

        self.attention_embedding_dim = (
            embedding_dim * 2 if adaptor_name == "kar_ours" else embedding_dim
        )
        self.tenure_encoder = nn.Sequential(
            nn.Linear(1, tenure_hidden_dim),
            nn.ReLU(),
            nn.Linear(tenure_hidden_dim, embedding_dim),
        )
        history_input_dim = self.attention_embedding_dim + embedding_dim
        self.interest_extractor = InterestExtractor(
            history_input_dim, self.attention_embedding_dim
        )
        self.interest_evolution = InterestEvolving(
            self.attention_embedding_dim,
            gru_type=self.dien_gru,
            dropout=dropout,
            attention_architecture=attention_architecture,
        )

        final_input_dim = self.attention_embedding_dim * 2
        if adaptor_name == "kar_base":
            final_input_dim += embedding_dim * 2
        elif adaptor_name == "kar_ours":
            final_input_dim += embedding_dim * 3
        self.final_mlp = MLP(
            final_mlp_architecture, final_input_dim, dropout
        )
        self.final_fc = nn.Linear(self.final_mlp.output_dim, 1)

    def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        task = self.prediction_task
        candidate_indices = inputs[f"{task}_candidate_indices"]
        history_indices = inputs[f"{task}_history_indices"]
        history_mask = inputs[f"{task}_history_mask"]
        history_tenures = inputs[f"{task}_history_tenures"]
        query_ids = self.item_embedding(candidate_indices)
        history_ids = self.item_embedding(history_indices)
        result: dict[str, Any] = {}

        user_knowledge: torch.Tensor | None = None
        item_knowledge: torch.Tensor | None = None
        if self.adaptor_name == "kar_base":
            if not isinstance(self.convert_module, HEA):
                raise RuntimeError("KAR-base HEA is not initialized")
            user_knowledge, item_knowledge, user_gates, item_gates = (
                self.convert_module(
                    inputs["user_reasoning_emb"],
                    inputs[f"{task}_candidate_embs"],
                )
            )
            query = query_ids
            history_items = history_ids
            result.update(
                user_gate_weights=user_gates,
                item_gate_weights=item_gates,
            )
        elif self.adaptor_name == "kar_ours":
            if not isinstance(self.convert_module, KAROursAdaptor):
                raise RuntimeError("KAR-Ours adaptor is not initialized")
            knowledge = self.convert_module(
                reasoning=inputs["user_reasoning_emb"],
                role_history=inputs["role_history_sequence"],
                role_history_mask=inputs["role_history_mask"],
                company_history=inputs["company_history_sequence"],
                company_history_mask=inputs["company_history_mask"],
                candidates=inputs[f"{task}_candidate_embs"],
                behaviors=inputs[f"{task}_history_sequence"],
            )
            user_knowledge = knowledge["user"]
            item_knowledge = knowledge["candidate"]
            query = torch.cat([item_knowledge, query_ids], dim=-1)
            history_items = torch.cat(
                [knowledge["behavior"], history_ids], dim=-1
            )
            result.update(
                user_gate_weights=knowledge["user_gate_weights"],
                item_gate_weights=knowledge["item_gate_weights"],
                behavior_gate_weights=knowledge["behavior_gate_weights"],
            )
        else:
            query = query_ids
            history_items = history_ids

        tenure_features = torch.log1p(history_tenures.abs()).unsqueeze(-1)
        tenure_embeddings = self.tenure_encoder(tenure_features)
        tenure_embeddings = tenure_embeddings * history_mask.unsqueeze(-1).to(
            tenure_embeddings.dtype
        )
        recurrent_inputs = torch.cat(
            [history_items, tenure_embeddings], dim=-1
        )
        extracted_interests = self.interest_extractor(
            recurrent_inputs, history_mask
        )
        user_interest, attention_scores = self.interest_evolution(
            query, extracted_interests, history_mask
        )

        final_features = [user_interest, query]
        if self.adaptor_name == "kar_base":
            if user_knowledge is None or item_knowledge is None:
                raise RuntimeError("KAR-base knowledge is missing")
            final_features.extend(
                [
                    user_knowledge.unsqueeze(1).expand(
                        -1, candidate_indices.shape[1], -1
                    ),
                    item_knowledge,
                ]
            )
        elif self.adaptor_name == "kar_ours":
            if user_knowledge is None or item_knowledge is None:
                raise RuntimeError("KAR-Ours knowledge is missing")
            final_features.append(
                user_knowledge.unsqueeze(1).expand(
                    -1, candidate_indices.shape[1], -1
                )
            )

        logits = self.final_fc(
            self.final_mlp(torch.cat(final_features, dim=-1))
        ).squeeze(-1)
        result.update(logits=logits, attention_scores=attention_scores)
        return result


__all__ = ["DCNv1", "DCNv2", "DIEN", "DeepFM", "DeepInterestNet"]
