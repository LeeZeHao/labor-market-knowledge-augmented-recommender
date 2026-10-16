"""Neural layers required by the career KAR-base DIN experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Dice(nn.Module):
    """Data-adaptive activation used by DIN."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.var(dim=0, unbiased=False)
        normalized = (inputs - inputs.mean(dim=0)) / torch.sqrt(
            variance + self.epsilon
        )
        probability = torch.sigmoid(normalized)
        return self.alpha * inputs * (1.0 - probability) + inputs * probability


class MLP(nn.Module):
    """Linear-activation-dropout stack."""

    def __init__(
        self,
        architecture: Sequence[int],
        input_dim: int,
        dropout: float,
        *,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for output_dim in architecture:
            layers.extend(
                [nn.Linear(current_dim, output_dim), activation(), nn.Dropout(dropout)]
            )
            current_dim = output_dim
        self.fc = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


class HEA(nn.Module):
    """Original-style two-input Hybrid-Expert Adaptor.

    The two inputs are user reasoning knowledge and candidate factual
    knowledge. Two experts are shared and six are specific to each input by
    default. Each input has its own linear-softmax gate.
    """

    def __init__(
        self,
        input_dim: int,
        expert_architecture: Sequence[int] = (128, 32),
        shared_expert_count: int = 2,
        specific_expert_count: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        architecture = tuple(expert_architecture)
        self.output_dim = architecture[-1]
        self.shared_experts = nn.ModuleList(
            [MLP(architecture, input_dim, dropout) for _ in range(shared_expert_count)]
        )
        self.specific_experts = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [MLP(architecture, input_dim, dropout) for _ in range(specific_expert_count)]
                )
                for name in ("user", "item")
            }
        )
        expert_count = shared_expert_count + specific_expert_count
        self.gates = nn.ModuleDict(
            {
                "user": nn.Linear(input_dim, expert_count),
                "item": nn.Linear(input_dim, expert_count),
            }
        )

    def _adapt(
        self, inputs: torch.Tensor, branch: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.stack(
            [expert(inputs) for expert in self.shared_experts]
            + [expert(inputs) for expert in self.specific_experts[branch]],
            dim=-2,
        )
        weights = F.softmax(self.gates[branch](inputs), dim=-1)
        adapted = torch.sum(weights.unsqueeze(-1) * outputs, dim=-2)
        return adapted, weights

    def forward(
        self,
        user_reasoning: torch.Tensor,
        candidate_facts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user, user_weights = self._adapt(user_reasoning, "user")
        item, item_weights = self._adapt(candidate_facts, "item")
        return user, item, user_weights, item_weights


class AttentionPoolingLayer(nn.Module):
    """KAR-repository DIN local activation with masked, unnormalized pooling."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        architecture: Sequence[int] = (32, 16),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = embedding_dim * 4
        for output_dim in architecture:
            layers.extend([nn.Linear(current_dim, output_dim), Dice(), nn.Dropout(dropout)])
            current_dim = output_dim
        layers.append(nn.Linear(current_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        user_behavior: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or user_behavior.ndim != 3:
            raise ValueError("query and behavior must have shapes [B,C,E] and [B,H,E]")
        history_length = user_behavior.shape[1]
        queries = query.unsqueeze(2).expand(-1, -1, history_length, -1)
        behaviors = user_behavior.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        attention_input = torch.cat(
            [queries, behaviors, queries - behaviors, queries * behaviors], dim=-1
        )
        scores = self.fc(attention_input).squeeze(-1)
        scores = scores * mask.unsqueeze(1).to(scores.dtype)
        interest = torch.sum(behaviors * scores.unsqueeze(-1), dim=2)
        return interest, scores


class CrossNetV1(nn.Module):
    """Rank-one DCNv1 cross network from the KAR repository.

    For each layer: ``x_(l+1) = x_0 (x_l^T w_l) + b_l + x_l``.
    """

    def __init__(self, input_dim: int, layer_count: int = 3) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        self.input_dim = input_dim
        self.layer_count = layer_count
        self.kernels = nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(torch.empty(input_dim, 1))
                )
                for _ in range(layer_count)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(input_dim, 1))
                for _ in range(layer_count)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [N,{self.input_dim}], got "
                f"{tuple(inputs.shape)}"
            )
        x0 = inputs.unsqueeze(2)
        xl = x0
        for kernel, bias in zip(self.kernels, self.biases):
            xl_w = torch.tensordot(xl, kernel, dims=([1], [0]))
            xl = torch.matmul(x0, xl_w) + bias + xl
        return xl.squeeze(2)


class CrossNetV2(nn.Module):
    """Full-matrix DCNv2 cross network from the KAR repository.

    For each layer: ``x_(l+1) = x_0 * (W_l x_l + b_l) + x_l``.
    """

    def __init__(self, input_dim: int, layer_count: int = 3) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        self.input_dim = input_dim
        self.layer_count = layer_count
        self.kernels = nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.xavier_normal_(
                        torch.empty(input_dim, input_dim)
                    )
                )
                for _ in range(layer_count)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(input_dim, 1))
                for _ in range(layer_count)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [N,{self.input_dim}], got "
                f"{tuple(inputs.shape)}"
            )
        x0 = inputs.unsqueeze(2)
        xl = x0
        for kernel, bias in zip(self.kernels, self.biases):
            transformed = torch.matmul(kernel, xl) + bias
            xl = x0 * transformed + xl
        return xl.squeeze(2)


class InterestExtractor(nn.Module):
    """First DIEN GRU: extract a latent interest at every history step.

    This follows the KAR repository's ``InterestExtractor`` while accepting a
    boolean mask and preserving zero-length rows in a grouped candidate batch.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(
        self, behaviors: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if behaviors.ndim != 3 or mask.ndim != 2:
            raise ValueError("behaviors and mask must have shapes [B,H,E] and [B,H]")
        batch_size, history_length, _ = behaviors.shape
        lengths = mask.long().sum(dim=1)
        valid_indices = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
        output = behaviors.new_zeros(
            batch_size, history_length, self.hidden_size
        )
        if valid_indices.numel() == 0:
            return output

        valid_behaviors = behaviors.index_select(0, valid_indices)
        valid_lengths = lengths.index_select(0, valid_indices)
        packed = pack_padded_sequence(
            valid_behaviors,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_interests, _ = self.gru(packed)
        interests, _ = pad_packed_sequence(
            packed_interests,
            batch_first=True,
            padding_value=0.0,
            total_length=history_length,
        )
        return output.index_copy(0, valid_indices, interests)


class AGRUCell(nn.Module):
    """Attention-based GRU cell used by the KAR DIEN implementation."""

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.empty(3 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size)
        )
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.hidden_size ** -0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        attention_score: torch.Tensor,
    ) -> torch.Tensor:
        input_gates = F.linear(inputs, self.weight_ih, self.bias_ih)
        hidden_gates = F.linear(hidden, self.weight_hh, self.bias_hh)
        input_reset, _, input_new = input_gates.chunk(3, dim=1)
        hidden_reset, _, hidden_new = hidden_gates.chunk(3, dim=1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        new_state = torch.tanh(input_new + reset_gate * hidden_new)
        attention_score = attention_score.reshape(-1, 1)
        return (1.0 - attention_score) * hidden + attention_score * new_state


class AUGRUCell(nn.Module):
    """GRU whose update gate is scaled by candidate attention."""

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.empty(3 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size)
        )
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.hidden_size ** -0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        attention_score: torch.Tensor,
    ) -> torch.Tensor:
        input_gates = F.linear(inputs, self.weight_ih, self.bias_ih)
        hidden_gates = F.linear(hidden, self.weight_hh, self.bias_hh)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_state = torch.tanh(input_new + reset_gate * hidden_new)
        update_gate = attention_score.reshape(-1, 1) * update_gate
        return (1.0 - update_gate) * hidden + update_gate * new_state


class InterestEvolving(nn.Module):
    """Second DIEN stage from KAR: GRU, AIGRU, AGRU, or AUGRU.

    The upstream implementation scores one candidate per row. Here the same
    equations are vectorized over the 50 candidates associated with each
    career record.
    """

    supported_gru_types = ("GRU", "AIGRU", "AGRU", "AUGRU")

    def __init__(
        self,
        input_size: int,
        gru_type: str = "GRU",
        dropout: float = 0.0,
        attention_architecture: Sequence[int] = (32, 16),
    ) -> None:
        super().__init__()
        gru_type = gru_type.upper()
        if gru_type not in self.supported_gru_types:
            raise ValueError(
                f"Unsupported DIEN GRU type {gru_type!r}; expected one of "
                f"{self.supported_gru_types}"
            )
        self.input_size = input_size
        self.gru_type = gru_type
        self.attention = AttentionPoolingLayer(
            input_size, dropout, attention_architecture
        )
        if gru_type in {"GRU", "AIGRU"}:
            self.interest_evolution: nn.Module = nn.GRU(
                input_size=input_size,
                hidden_size=input_size,
                batch_first=True,
            )
        elif gru_type == "AGRU":
            self.interest_evolution = AGRUCell(input_size, input_size)
        else:
            self.interest_evolution = AUGRUCell(input_size, input_size)

    @staticmethod
    def _run_packed_gru(
        gru: nn.GRU,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_sequence: bool,
    ) -> torch.Tensor:
        count, sequence_length, hidden_size = sequences.shape
        valid_indices = torch.nonzero(lengths > 0, as_tuple=False).squeeze(1)
        if return_sequence:
            output = sequences.new_zeros(count, sequence_length, hidden_size)
        else:
            output = sequences.new_zeros(count, hidden_size)
        if valid_indices.numel() == 0:
            return output

        valid_sequences = sequences.index_select(0, valid_indices)
        valid_lengths = lengths.index_select(0, valid_indices)
        packed = pack_padded_sequence(
            valid_sequences,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = gru(packed)
        unpacked, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            padding_value=0.0,
            total_length=sequence_length,
        )
        if return_sequence:
            valid_output = unpacked
        else:
            rows = torch.arange(valid_indices.numel(), device=sequences.device)
            valid_output = unpacked[rows, valid_lengths - 1]
        return output.index_copy(0, valid_indices, valid_output)

    def _candidate_attention(
        self,
        query: torch.Tensor,
        interests: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply upstream single-candidate attention over grouped candidates."""

        batch_size, candidate_count, hidden_size = query.shape
        history_length = interests.shape[1]
        flattened_query = query.reshape(
            batch_size * candidate_count, 1, hidden_size
        )
        flattened_interests = interests.unsqueeze(1).expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, history_length, hidden_size)
        flattened_mask = mask.unsqueeze(1).expand(
            -1, candidate_count, -1
        ).reshape(batch_size * candidate_count, history_length)
        pooled, scores = self.attention(
            flattened_query, flattened_interests, flattened_mask
        )
        return (
            pooled.reshape(batch_size, candidate_count, hidden_size),
            scores.reshape(batch_size, candidate_count, history_length),
        )

    def forward(
        self,
        query: torch.Tensor,
        interests: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or interests.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "query, interests, and mask must have shapes [B,C,E], "
                "[B,H,E], and [B,H]"
            )
        batch_size, candidate_count, hidden_size = query.shape
        history_length = interests.shape[1]
        lengths = mask.long().sum(dim=1)

        if self.gru_type == "GRU":
            if not isinstance(self.interest_evolution, nn.GRU):
                raise RuntimeError("GRU interest evolution is not initialized")
            evolved = self._run_packed_gru(
                self.interest_evolution,
                interests,
                lengths,
                return_sequence=True,
            )
            return self._candidate_attention(query, evolved, mask)

        _, attention_scores = self._candidate_attention(
            query, interests, mask
        )
        candidate_interests = interests.unsqueeze(1).expand(
            -1, candidate_count, -1, -1
        )

        if self.gru_type == "AIGRU":
            if not isinstance(self.interest_evolution, nn.GRU):
                raise RuntimeError("AIGRU interest evolution is not initialized")
            weighted = candidate_interests * attention_scores.unsqueeze(-1)
            flattened = weighted.reshape(
                batch_size * candidate_count, history_length, hidden_size
            )
            candidate_lengths = lengths.unsqueeze(1).expand(
                -1, candidate_count
            ).reshape(-1)
            evolved = self._run_packed_gru(
                self.interest_evolution,
                flattened,
                candidate_lengths,
                return_sequence=False,
            )
            return evolved.reshape(batch_size, candidate_count, hidden_size), attention_scores

        if not isinstance(self.interest_evolution, (AGRUCell, AUGRUCell)):
            raise RuntimeError("Attention-gated interest evolution is not initialized")
        flattened = candidate_interests.reshape(
            batch_size * candidate_count, history_length, hidden_size
        )
        flattened_scores = attention_scores.reshape(
            batch_size * candidate_count, history_length
        )
        flattened_mask = mask.unsqueeze(1).expand(
            -1, candidate_count, -1
        ).reshape(batch_size * candidate_count, history_length)
        hidden = interests.new_zeros(
            batch_size * candidate_count, hidden_size
        )
        for position in range(history_length):
            proposed = self.interest_evolution(
                flattened[:, position],
                hidden,
                flattened_scores[:, position],
            )
            active = flattened_mask[:, position].unsqueeze(1)
            hidden = torch.where(active, proposed, hidden)
        return hidden.reshape(batch_size, candidate_count, hidden_size), attention_scores


class ResidualExpertMLP(nn.Module):
    """Residual 1024→bottleneck→1024 expert used by KAR-Ours."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        initial_alpha: float,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.normalize(inputs, p=2, dim=-1)
        return F.normalize(base + self.alpha * self.adapter(base), p=2, dim=-1)


class KAROursAdaptor(nn.Module):
    """Current residual HEA used by the career ``kar_ours`` experiment."""

    user_modalities = ("reasoning", "role_history", "company_history")

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 32,
        expert_hidden_dim: int = 32,
        shared_expert_count: int = 2,
        specific_expert_count: int = 6,
        history_specific_expert_count: int = 2,
        dropout: float = 0.1,
        initial_alpha: float = 0.01,
        history_decay_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shared_expert_count = shared_expert_count
        self.specific_expert_count = specific_expert_count
        self.history_specific_expert_count = history_specific_expert_count
        self.history_decay_rate = history_decay_rate

        def make_expert() -> ResidualExpertMLP:
            return ResidualExpertMLP(
                input_dim, expert_hidden_dim, dropout, initial_alpha
            )

        self.shared_experts = nn.ModuleList(
            [make_expert() for _ in range(shared_expert_count)]
        )
        self.reasoning_experts = nn.ModuleList(
            [make_expert() for _ in range(specific_expert_count)]
        )
        self.history_experts = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [make_expert() for _ in range(history_specific_expert_count)]
                )
                for name in ("role_history", "company_history")
            }
        )
        self.item_experts = nn.ModuleList(
            [make_expert() for _ in range(specific_expert_count)]
        )
        gate_modules: dict[str, nn.Module] = {}
        for modality in self.user_modalities:
            specific_count = (
                specific_expert_count
                if modality == "reasoning"
                else history_specific_expert_count
            )
            gate_modules[f"user_{modality}"] = nn.Linear(
                input_dim, shared_expert_count + specific_count
            )
        gate_modules["item"] = nn.Linear(
            input_dim, shared_expert_count + specific_expert_count
        )
        self.gates = nn.ModuleDict(gate_modules)
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.user_output_projections = nn.ModuleDict(
            {
                modality: nn.Linear(input_dim, output_dim)
                for modality in self.user_modalities
            }
        )
        self.item_output_projection = nn.Linear(input_dim, output_dim)

    @staticmethod
    def _mix(
        expert_outputs: torch.Tensor,
        gate_logits: torch.Tensor,
        valid_experts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_experts is not None:
            gate_logits = gate_logits.masked_fill(~valid_experts, -1e9)
        weights = F.softmax(gate_logits, dim=-1)
        adapted = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=-2)
        return F.normalize(adapted, p=2, dim=-1), weights

    def _recency_pool(
        self, history: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        valid = mask.bool()
        lengths = valid.sum(dim=1)
        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0)
        distances = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
        weights = torch.exp(-self.history_decay_rate * distances.to(history.dtype))
        weights = weights * valid.to(history.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(history * weights.unsqueeze(-1), dim=1)

    def _adapt_user_stream(
        self, inputs: torch.Tensor, modality: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        specific_experts = (
            self.reasoning_experts
            if modality == "reasoning"
            else self.history_experts[modality]
        )
        outputs = torch.stack(
            [expert(inputs) for expert in self.shared_experts]
            + [expert(inputs) for expert in specific_experts],
            dim=1,
        )
        valid = inputs.norm(dim=-1) > 1e-8
        valid_experts = torch.cat(
            [
                torch.ones(
                    inputs.shape[0],
                    self.shared_expert_count,
                    dtype=torch.bool,
                    device=inputs.device,
                ),
                valid.unsqueeze(1).expand(-1, len(specific_experts)),
            ],
            dim=1,
        )
        adapted, gates = self._mix(
            outputs,
            self.gates[f"user_{modality}"](
                F.normalize(inputs, p=2, dim=-1)
            ),
            valid_experts,
        )
        return self.user_output_projections[modality](adapted), gates

    def _adapt_items(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch.stack(
            [expert(inputs) for expert in self.shared_experts]
            + [expert(inputs) for expert in self.item_experts],
            dim=-2,
        )
        adapted, gates = self._mix(
            outputs,
            self.gates["item"](F.normalize(inputs, p=2, dim=-1)),
        )
        return self.item_output_projection(adapted), gates

    def forward(
        self,
        *,
        reasoning: torch.Tensor,
        role_history: torch.Tensor,
        role_history_mask: torch.Tensor,
        company_history: torch.Tensor,
        company_history_mask: torch.Tensor,
        candidates: torch.Tensor,
        behaviors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pooled = {
            "reasoning": reasoning,
            "role_history": self._recency_pool(
                role_history, role_history_mask
            ),
            "company_history": self._recency_pool(
                company_history, company_history_mask
            ),
        }
        user_streams: list[torch.Tensor] = []
        user_gates: list[torch.Tensor] = []
        for modality in self.user_modalities:
            adapted, gates = self._adapt_user_stream(pooled[modality], modality)
            user_streams.append(adapted)
            user_gates.append(gates)
        candidate, item_gates = self._adapt_items(candidates)
        behavior, behavior_gates = self._adapt_items(behaviors)
        return {
            "user": torch.cat(user_streams, dim=-1),
            "candidate": candidate,
            "behavior": behavior,
            "user_gate_weights": torch.cat(user_gates, dim=-1),
            "item_gate_weights": item_gates,
            "behavior_gate_weights": behavior_gates,
        }


__all__ = [
    "AttentionPoolingLayer",
    "Dice",
    "AGRUCell",
    "AUGRUCell",
    "CrossNetV1",
    "CrossNetV2",
    "HEA",
    "InterestEvolving",
    "InterestExtractor",
    "KAROursAdaptor",
    "MLP",
    "ResidualExpertMLP",
]
