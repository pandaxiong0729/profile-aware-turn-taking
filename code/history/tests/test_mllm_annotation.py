from __future__ import annotations

from profile_turntaking.mllm_annotation import _parse_annotation, build_annotation_prompt


def test_annotation_prompt_is_retrospective_and_targeted() -> None:
    event = {
        "target_start_s": 12.0,
        "target_end_s": 12.6,
        "context_start_s": 2.0,
        "candidate_labels": ["BC", "I"],
    }
    prompt = build_annotation_prompt(event, view="floor_questions")
    assert "retrospectively" in prompt
    assert "do not predict" in prompt.lower()
    assert "No profile" in prompt
    assert "candidate label" in prompt
    assert "BC, I" not in prompt
    assert "BEFORE, TARGET, and AFTER" in prompt


def test_annotation_label_is_derived_from_observable_facts() -> None:
    raw = """{
      "label": "BC",
      "speech_in_target": true,
      "target_matches_pre_target_floor_voice": false,
      "brief_feedback_only": true,
      "audible_overlap": true,
      "target_speaker_controls_floor_after": false,
      "original_floor_holder_continues_after": true,
      "confidence": 0.9,
      "needs_review": false,
      "reason": "Brief listener feedback; the prior speaker continues."
    }"""
    parsed = _parse_annotation(raw)
    assert parsed is not None
    assert parsed["derived_label"] == "BC"
    assert parsed["needs_review"] is False


def test_contradictory_observations_are_forced_to_review() -> None:
    raw = """{
      "label": "BC",
      "speech_in_target": true,
      "target_matches_pre_target_floor_voice": false,
      "brief_feedback_only": true,
      "audible_overlap": false,
      "target_speaker_controls_floor_after": true,
      "original_floor_holder_continues_after": true,
      "confidence": 1.0,
      "needs_review": false,
      "reason": "Contradictory on purpose."
    }"""
    parsed = _parse_annotation(raw)
    assert parsed is not None
    assert parsed["derived_label"] == "UNCERTAIN"
    assert parsed["needs_review"] is True
    assert parsed["label_consistent"] is False


def test_simple_direct_judgement_is_accepted() -> None:
    raw = """{
      "label": "T",
      "confidence": 0.82,
      "needs_review": false,
      "reason": "The second speaker cleanly keeps the floor afterwards."
    }"""
    parsed = _parse_annotation(raw)
    assert parsed is not None
    assert parsed["label"] == "T"
    assert parsed["label_consistent"] is True


def test_out_of_range_confidence_is_rejected() -> None:
    raw = """{
      "label": "NA",
      "confidence": 2.0,
      "needs_review": false,
      "reason": "Invalid confidence."
    }"""
    assert _parse_annotation(raw) is None
