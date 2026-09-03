"""Opt-in profile soft-prompt adapter for Qwen2.5-Omni A/B questions.

The base Qwen model is frozen.  A small trainable module converts a cached
profile embedding into a short sequence of continuous (soft) tokens.  Those
tokens continue the real multimodal prompt after Qwen has encoded causal audio,
the matching causal transcript, and one A/B question.  The answer is scored by
Qwen's original frozen vocabulary head.

When ``adapter_enabled=False`` the wrapper delegates directly to Qwen and does
not alter its inputs or outputs.  This makes the turn-taking adapter opt-in and
keeps ordinary Qwen generation available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


PAPER_TASKS = ("turn_change", "backchannel", "interruption", "floor_taking")

PAPER_AB_OPTIONS = {
    "turn_change": (
        "The current floor holder continues; no natural turn change begins.",
        "The other participant begins a natural turn after the current speaker yields.",
    ),
    "backchannel": (
        "No new listener backchannel begins.",
        "The listener begins a short acknowledgement while the current speaker keeps the floor.",
    ),
    "interruption": (
        "No substantive interruption by the other participant begins.",
        "The other participant begins a substantive contribution before the current speaker yields.",
    ),
    "floor_taking": (
        "The overlapping participant does not take the floor; the original speaker continues.",
        "The overlapping participant takes the floor and becomes the next floor holder.",
    ),
}


def build_paper_ab_prompt(record: Mapping[str, Any], task: str) -> str:
    """Build one causal natural-language A/B question without profile or label text."""

    if task not in PAPER_AB_OPTIONS:
        raise ValueError(f"Unknown A/B task: {task}")
    transcript = record.get("transcript_prefix") or "No completed transcript is available."
    boundary = record.get("boundary_state_text") or "No speaker is marked active at t."
    offset = int(record.get("forecast_offset_ms", 100))
    option_a, option_b = PAPER_AB_OPTIONS[task]
    return "\n".join(
        [
            "Listen to the attached two-person conversation.",
            "The audio and transcript end exactly at prediction boundary t.",
            "Use no future audio, future transcript, target label, or annotation evidence.",
            "Completed causal transcript before t:",
            str(transcript),
            "",
            "Speaker activity exactly at t:",
            str(boundary),
            "",
            f"At t+{offset} ms, which is more likely?",
            f"A. {option_a}",
            f"B. {option_b}",
            "Answer only A or B.",
        ]
    )


@dataclass
class PreparedQwenPrefix:
    """Frozen multimodal prompt state consumed by the trainable soft suffix."""

    past_key_values: Any
    attention_mask: torch.Tensor
    last_position_ids: torch.Tensor


class ProfileSoftTokenAdapter(nn.Module):
    """Map a profile vector and task identity to K Qwen-sized soft tokens."""

    def __init__(
        self,
        *,
        profile_dim: int,
        qwen_dim: int,
        prefix_length: int = 4,
        bottleneck_dim: int = 256,
        task_names: tuple[str, ...] = PAPER_TASKS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.profile_dim = int(profile_dim)
        self.qwen_dim = int(qwen_dim)
        self.prefix_length = int(prefix_length)
        self.task_names = tuple(task_names)
        self.task_to_index = {name: index for index, name in enumerate(self.task_names)}
        self.base_tokens = nn.Parameter(torch.empty(self.prefix_length, self.qwen_dim))
        self.task_tokens = nn.Parameter(
            torch.zeros(len(self.task_names), self.prefix_length, self.qwen_dim)
        )
        self.profile_net = nn.Sequential(
            nn.LayerNorm(self.profile_dim, elementwise_affine=False),
            nn.Linear(self.profile_dim, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, self.prefix_length * self.qwen_dim, bias=False),
        )
        nn.init.normal_(self.base_tokens, mean=0.0, std=0.02)
        # Start from a task-only soft prompt.  Profile influence grows only when
        # supported by the training loss, which is safer than a large random shift.
        nn.init.zeros_(self.profile_net[-1].weight)

    def forward(self, profile: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.task_to_index:
            raise ValueError(f"Unknown A/B task: {task}")
        if profile.ndim != 2 or profile.shape[1] != self.profile_dim:
            raise ValueError(
                f"Expected profile [batch,{self.profile_dim}], got {tuple(profile.shape)}"
            )
        batch = profile.shape[0]
        present = (profile.abs().sum(dim=-1, keepdim=True) > 0).to(profile.dtype)
        delta = self.profile_net(profile).reshape(batch, self.prefix_length, self.qwen_dim)
        delta = delta * present.unsqueeze(-1)
        task_tokens = self.task_tokens[self.task_to_index[task]].unsqueeze(0)
        return self.base_tokens.unsqueeze(0) + task_tokens + delta


class FrozenQwenSoftPromptAB(nn.Module):
    """Run an opt-in soft suffix through frozen Qwen and score A/B tokens."""

    def __init__(
        self,
        thinker: nn.Module,
        adapter: ProfileSoftTokenAdapter,
        *,
        answer_token_ids: tuple[int, int],
    ) -> None:
        super().__init__()
        self.thinker = thinker
        self.adapter = adapter
        self.answer_token_ids = tuple(int(value) for value in answer_token_ids)
        for parameter in self.thinker.parameters():
            parameter.requires_grad_(False)
        self.thinker.eval()

    def base_forward(self, **inputs: Any) -> Any:
        """Exact ordinary-Qwen path used when the adapter is disabled."""

        return self.thinker(**inputs)

    @torch.no_grad()
    def prepare_multimodal_prefix(self, inputs: Mapping[str, Any]) -> PreparedQwenPrefix:
        """Encode the real prompt once without materializing full-vocabulary logits."""

        thinker = self.thinker
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if input_ids.shape[0] != 1:
            raise ValueError("The memory-safe local implementation currently requires batch size 1")
        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        input_features = inputs.get("input_features")
        feature_attention_mask = inputs.get("feature_attention_mask")
        audio_feature_lengths = inputs.get("audio_feature_lengths")
        if input_features is not None:
            audio_features = thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                return_dict=True,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = thinker.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        if feature_attention_mask is not None:
            audio_feature_lengths = feature_attention_mask.sum(dim=1)
        else:
            audio_feature_lengths = None
        position_ids, rope_deltas = thinker.get_rope_index(
            input_ids,
            None,
            None,
            attention_mask,
            None,
            audio_feature_lengths,
            None,
        )
        outputs = thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        thinker.rope_deltas = rope_deltas - (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        valid = attention_mask[0].bool()
        last_position_ids = position_ids[:, 0, valid][:, -1]
        return PreparedQwenPrefix(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            last_position_ids=last_position_ids,
        )

    def score_prepared(
        self,
        prepared: PreparedQwenPrefix,
        profile: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        """Append profile soft tokens and return Qwen LM-head logits for A/B."""

        soft = self.adapter(profile, task)
        soft = soft.to(next(self.thinker.parameters()).device)
        batch, length, _ = soft.shape
        if batch != 1:
            raise ValueError("Prepared Qwen prefix currently supports batch size 1")
        prefix_mask = prepared.attention_mask.to(soft.device)
        attention_mask = torch.cat(
            [prefix_mask, torch.ones(batch, length, device=soft.device, dtype=prefix_mask.dtype)],
            dim=1,
        )
        increments = torch.arange(1, length + 1, device=soft.device, dtype=torch.long)
        position_ids = prepared.last_position_ids.to(soft.device)[:, None] + increments[None, :]
        position_ids = position_ids[:, None, :].expand(3, batch, length)
        outputs = self.thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prepared.past_key_values,
            inputs_embeds=soft.to(dtype=next(self.thinker.parameters()).dtype),
            use_cache=False,
            return_dict=True,
        )
        final_hidden = outputs.last_hidden_state[:, -1]
        vocabulary_logits = self.thinker.lm_head(final_hidden)
        ids = torch.tensor(self.answer_token_ids, device=vocabulary_logits.device)
        return vocabulary_logits.index_select(dim=-1, index=ids).float()

    def forward(
        self,
        *,
        adapter_enabled: bool,
        profile: torch.Tensor | None = None,
        task: str | None = None,
        prepared: PreparedQwenPrefix | None = None,
        **base_inputs: Any,
    ) -> Any:
        if not adapter_enabled:
            return self.base_forward(**base_inputs)
        if profile is None or task is None:
            raise ValueError("profile and task are required when adapter is enabled")
        if prepared is None:
            prepared = self.prepare_multimodal_prefix(base_inputs)
        return self.score_prepared(prepared, profile, task)


def exact_single_token_id(tokenizer: Any, text: str) -> int:
    """Require an answer such as A or B to map to exactly one vocabulary token."""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{text!r} is not one token: {token_ids}")
    return int(token_ids[0])


__all__ = [
    "FrozenQwenSoftPromptAB",
    "PAPER_TASKS",
    "PreparedQwenPrefix",
    "ProfileSoftTokenAdapter",
    "build_paper_ab_prompt",
    "exact_single_token_id",
]
