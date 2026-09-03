from profile_turntaking.omni_technical_audit import (
    build_future_prompt,
    build_occurred_prompt,
    word_overlap_f1,
)


def test_future_prompt_preserves_causal_transcript_and_profile() -> None:
    prompt = build_future_prompt(
        task="backchannel",
        transcript="[speaker_00 0.000-1.000] hello",
        profile_text="Profiles are unknown.",
        style="paper",
    )
    assert "[speaker_00 0.000-1.000] hello" in prompt
    assert "Profiles are unknown." in prompt
    assert "BACKCHANNEL" in prompt
    assert "NO_BACKCHANNEL" in prompt
    assert "target" not in prompt.lower()


def test_occurred_prompt_marks_exact_boundary() -> None:
    prompt = build_occurred_prompt("interruption")
    assert "5.0 seconds" in prompt
    assert "event recognition" in prompt.lower()
    assert "OTHER_SPEAKER_INTERRUPTS" in prompt


def test_word_overlap_f1_is_bounded_and_ignores_timestamps() -> None:
    score = word_overlap_f1(
        "[speaker_00 0.000-1.000] hello there",
        "hello there friend",
    )
    assert 0.0 < score < 1.0
    assert word_overlap_f1("same words", "same words") == 1.0
    assert word_overlap_f1("hello", "") == 0.0
