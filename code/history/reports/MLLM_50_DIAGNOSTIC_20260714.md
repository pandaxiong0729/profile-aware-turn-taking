# Qwen2.5-Omni-3B 50-sample diagnostic

Date: 2026-07-14

## Decision

Do not use this checkpoint for the profile-effect experiment. It does not form a
stable five-class audio baseline, and detailed profile text shifts it toward `I`
without showing sensitivity to whether the profile is correct or shuffled.

This diagnostic is not a formal result: all 50 targets are unreviewed weak labels.
No positive or negative scientific claim about profile conditioning is allowed.

## Fixed data

- 50 distinct event-onset samples: 10 each of C/BC/T/I/NA;
- all 16 core conversations represented;
- each conversation contributes at most one row to one class;
- within-conversation prediction boundaries are at least 9.24 seconds apart;
- all rows with automatic non-lexical, environment, or already-active BC/I risk
  flags were excluded before inference;
- hidden/given/shuffled use identical audio, transcript, boundary, prompt template,
  model, decoding, and sample IDs; only profile text changes.

The fixed local manifest is
`data/processed/sbcsae_mvp_v2/prompt_pilot_lowrisk_50.jsonl`.

## Diagnostic 1: locked prompt

| Condition | Macro-F1 | Balanced accuracy | Accuracy | Predicted labels |
|---|---:|---:|---:|---|
| hidden | 0.1333 | 0.2400 | 0.2400 | T=18, I=32 |
| given | 0.1060 | 0.2000 | 0.2000 | T=8, I=42 |
| shuffled | 0.1074 | 0.2000 | 0.2000 | T=7, I=43 |

The hidden condition predicts only two labels, so the scientific-validity gate
fails. Given fixes one hidden error but breaks three hidden-correct rows. The
conversation-bootstrap given-minus-hidden Macro-F1 is -0.0274 with 95% interval
[-0.0681, 0.0104].

Replacing the hidden audio with duration-matched digital silence changes 17/50
predictions (34%). The checkpoint is not completely audio-insensitive, but it is
not using the audio strongly enough for the requested five-class boundary task.

## Diagnostic 2: one end-focused prompt repair

The same 50 rows were used strictly as a development diagnostic. One prompt
repair told the model to focus on the final 500 ms, not treat earlier overlap as
target overlap, consider all five labels, and use profile only as a secondary
prior. This variant is preserved in the generated request artifacts but was not
accepted as the locked experiment prompt.

| Condition | Macro-F1 | Balanced accuracy | Accuracy | Predicted labels |
|---|---:|---:|---:|---|
| hidden | 0.2035 | 0.2600 | 0.2600 | C=2, T=8, I=29, NA=11 |
| given | 0.1615 | 0.2400 | 0.2400 | C=1, T=3, I=44, NA=2 |
| shuffled | 0.1584 | 0.2400 | 0.2400 | C=4, T=1, I=43, NA=2 |

The repair makes hidden non-collapsed by the predeclared gate, but `BC` is never
predicted. More importantly, given and shuffled remain almost identical: only
5/50 paired predictions differ, while both push roughly 86–88% of rows to `I`.
Correct profile therefore does not outperform absence or shuffled profile.

## Root-cause conclusion

1. The initial temporal wording contributed to the T/I collapse; focusing on the
   boundary partially fixes hidden.
2. The checkpoint does not reliably interpret the natural-language profile as
   turn-taking evidence. It reacts similarly to correct and shuffled detailed
   profiles, so the change is a text-conditioning bias rather than profile use.
3. A 40 ms future event is too specialized for this zero-shot 3B quantized MLLM.
4. Weak TRN-derived labels remain an independent limitation and must be reviewed.

Continuing to tune prompts on these same 50 targets until `given` wins would be
test-set overfitting. The defensible next experiment is the supervised profile
adapter described in the project plan, or a stronger audio MLLM evaluated once on
a disjoint reviewed set.

## Local artifacts

- locked-prompt run:
  `artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-lowrisk-50-diagnostic/`
- silence control:
  `artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-lowrisk-50-silenced/`
- end-focused development run:
  `artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/onset-balanced-lowrisk-50-prompt-v2-dev/`

Generated audio, requests, raw responses, and predictions remain Git-ignored.
