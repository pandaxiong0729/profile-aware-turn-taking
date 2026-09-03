from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from profile_turntaking.qwen_soft_prompt_adapter import (
    FrozenQwenSoftPromptAB,
    PreparedQwenPrefix,
    ProfileSoftTokenAdapter,
    build_paper_ab_prompt,
)


class TinyFrozenThinker(nn.Module):
    def __init__(self, dim: int = 12, vocab: int = 17) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.randn(dim))
        self.model = TinyBackbone(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float() + self.anchor[0]


class TinyBackbone(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim, bias=False)

    def forward(self, *, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.projection(inputs_embeds))


def build_model() -> FrozenQwenSoftPromptAB:
    thinker = TinyFrozenThinker()
    adapter = ProfileSoftTokenAdapter(
        profile_dim=8,
        qwen_dim=12,
        prefix_length=3,
        bottleneck_dim=5,
        dropout=0.0,
    )
    return FrozenQwenSoftPromptAB(thinker, adapter, answer_token_ids=(3, 7))


def test_disabled_adapter_is_exact_base_path() -> None:
    model = build_model()
    ids = torch.tensor([[1, 2, 3]])
    direct = model.thinker(input_ids=ids)
    wrapped = model(adapter_enabled=False, input_ids=ids)
    assert torch.equal(direct, wrapped)


def test_soft_prompt_returns_ab_logits_and_only_adapter_trains() -> None:
    model = build_model()
    prepared = PreparedQwenPrefix(
        past_key_values=None,
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        last_position_ids=torch.tensor([3, 3, 3]),
    )
    logits = model(
        adapter_enabled=True,
        profile=torch.randn(1, 8),
        task="turn_change",
        prepared=prepared,
    )
    assert logits.shape == (1, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert any(parameter.grad is not None for parameter in model.adapter.parameters())
    assert all(parameter.grad is None for parameter in model.thinker.parameters())


def test_hidden_and_given_keep_identical_soft_token_shape() -> None:
    model = build_model()
    hidden = model.adapter(torch.zeros(2, 8), "backchannel")
    given = model.adapter(torch.randn(2, 8), "backchannel")
    assert hidden.shape == given.shape == (2, 3, 12)


def test_all_four_prompts_are_causal_and_exclude_profile_and_label() -> None:
    record = {
        "transcript_prefix": "speaker_00: hello",
        "boundary_state_text": "speaker_00 is audible",
        "forecast_offset_ms": 100,
        "profile_text": "SECRET_PROFILE",
        "reference_label": "SECRET_LABEL",
    }
    for task in ("turn_change", "backchannel", "interruption", "floor_taking"):
        prompt = build_paper_ab_prompt(record, task)
        assert "speaker_00: hello" in prompt
        assert "t+100 ms" in prompt
        assert "SECRET_PROFILE" not in prompt
        assert "SECRET_LABEL" not in prompt
        assert "future audio" in prompt
