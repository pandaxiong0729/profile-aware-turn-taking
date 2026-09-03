from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_talking_turns_hidden_requests.py"
SPEC = spec_from_file_location("hidden_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_prediction_prefers_longer_non_backchannel_label() -> None:
    assert MODULE.parse_prediction(
        "NO_BACKCHANNEL", ["BACKCHANNEL", "NO_BACKCHANNEL"]
    ) == "NO_BACKCHANNEL"


def test_parse_prediction_accepts_named_semantic_output() -> None:
    assert MODULE.parse_prediction(
        "Final: OTHER_SPEAKER_TAKES_TURN.",
        ["CURRENT_SPEAKER_CONTINUES", "OTHER_SPEAKER_TAKES_TURN"],
    ) == "OTHER_SPEAKER_TAKES_TURN"
