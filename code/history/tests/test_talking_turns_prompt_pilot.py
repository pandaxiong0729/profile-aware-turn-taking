from profile_turntaking.talking_turns_prompt_pilot import (
    build_probability_prompt,
    build_semantic_prompt,
)


def test_semantic_prompt_has_causal_inputs_and_no_ab_tokens() -> None:
    prompt = build_semantic_prompt(
        task="turn_change",
        transcript="[speaker_00 0.000-1.000] hello",
        profile_text="Speaker profiles are unknown.",
        reverse_order=False,
    )
    assert "ends exactly at t" in prompt
    assert "[speaker_00 0.000-1.000] hello" in prompt
    assert "CURRENT_SPEAKER_CONTINUES" in prompt
    assert "OTHER_SPEAKER_TAKES_TURN" in prompt
    assert "A =" not in prompt
    assert "B =" not in prompt


def test_reverse_order_changes_only_label_order() -> None:
    kwargs = {
        "task": "backchannel",
        "transcript": "[speaker_01 0.000-1.000] yes",
        "profile_text": "unknown",
    }
    forward = build_semantic_prompt(**kwargs, reverse_order=False)
    reversed_prompt = build_semantic_prompt(**kwargs, reverse_order=True)
    assert forward.index("BACKCHANNEL") < forward.index("NO_BACKCHANNEL")
    assert reversed_prompt.index("NO_BACKCHANNEL") < reversed_prompt.index("BACKCHANNEL")


def test_probability_prompt_marks_exact_boundary_and_event() -> None:
    prompt = build_probability_prompt(
        task="interruption",
        transcript="[speaker_00 0.000-1.000] because",
        profile_text="unknown",
    )
    assert "ends exactly at t" in prompt
    assert "before the current speaker finishes" in prompt
    assert '"probability":INTEGER' in prompt
