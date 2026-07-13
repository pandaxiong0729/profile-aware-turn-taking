from __future__ import annotations

import torch

from profile_turntaking.constants import LABELS, PROFILE_FIELDS
from profile_turntaking.model import ModelConfig, ProfileTurnModel


def test_statistical_model_forward() -> None:
    config = ModelConfig(hidden_dimension=32)
    model = ProfileTurnModel(config)
    logits = model(
        torch.randn(4, config.statistical_dimension),
        torch.randn(4, config.text_dimension),
        torch.randint(0, config.profile_buckets, (4, len(PROFILE_FIELDS))),
    )
    assert logits.shape == (4, len(LABELS))
