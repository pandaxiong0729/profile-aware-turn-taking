from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_talking_turns_hidden50_with_qwen_asr.py"
)
SPEC = spec_from_file_location("prepare_hidden50_with_qwen_asr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_add_asr_to_prompt_preserves_prediction_instructions() -> None:
    prompt = (
        "Predict only what happens after t.\n"
        "Causal speaker-timed partial transcript:\nhello\n"
        "\nProfile condition:\nunknown\n"
        "Return only one conclusion."
    )
    result = MODULE.add_asr_to_prompt(prompt, "hello there")
    assert "Predict only what happens after t." in result
    assert "Causal ASR transcript generated from this exact audio:" in result
    assert "hello there" in result
    assert result.count("Profile condition:") == 1
    assert "Return only one conclusion." in result


def test_add_asr_to_prompt_requires_profile_marker() -> None:
    try:
        MODULE.add_asr_to_prompt("missing marker", "hello")
    except ValueError as exc:
        assert "profile marker" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
