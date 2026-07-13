"""Shared label and profile constants."""

from __future__ import annotations

LABELS = ("C", "BC", "T", "I", "NA")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

PROFILE_FIELDS = (
    "speaker_A.age_group",
    "speaker_A.gender",
    "speaker_A.social_role",
    "speaker_A.background",
    "speaker_B.age_group",
    "speaker_B.gender",
    "speaker_B.social_role",
    "speaker_B.background",
    "relationship",
    "situation",
)

UNKNOWN_PROFILE = {
    "speaker_A": {
        "age_group": "unknown",
        "gender": "unknown",
        "social_role": "unknown",
        "background": "unknown",
    },
    "speaker_B": {
        "age_group": "unknown",
        "gender": "unknown",
        "social_role": "unknown",
        "background": "unknown",
    },
    "relationship": "unknown",
    "situation": "unknown",
}

BACKCHANNEL_WORDS = {
    "ah",
    "alright",
    "exactly",
    "got it",
    "hm",
    "hmm",
    "mhm",
    "mm",
    "mm hm",
    "okay",
    "oh",
    "right",
    "sure",
    "uh huh",
    "unhunh",
    "yeah",
    "yes",
    "yep",
}
