# SBCSAE five-class label audit and v2 rebuild

Date: 2026-07-14

## Decision

The earlier 500-row Qwen2.5-Omni prompt result is invalid and must not be used as
evidence for or against profile conditioning. The primary failure is not the
Speaker A/B profile mapping. It is a combination of weak-label implementation,
frame sampling, prompt semantics, and transcript-proxy errors.

## Reproduced failures

The old 54,270-row manifest was relabeled at the same sample boundaries after
repairing the parser, transcript cleaner, overlap test, and previous-floor rule.
Exactly 3,953 rows (7.28%) changed class. In the old 500-row MLLM set, 62 rows
(12.4%) changed:

| Old class | Remained | Changed to other classes |
|---|---:|---|
| C | 100/100 | none |
| BC | 98/100 | 1 C, 1 I |
| T | 60/100 | 31 C, 8 BC, 1 I |
| I | 86/100 | 10 BC, 4 T |
| NA | 94/100 | 6 C |

The six changed NA samples are inside an utterance lost by a physically fused
source row in `SBC007.trn`. The parser now recovers that row. A second fused row
in `SBC014.trn` is also recovered.

Additional implementation problems:

- SBCSAE square brackets mark overlap and contain real words. The old cleaner
  deleted the complete bracketed span, so `[Mhm]` became empty and could turn a
  backchannel into interruption.
- The fast labeler called a chunk `I` whenever both speakers intersected the
  chunk, even if A stopped before B started. It now requires true pairwise
  temporal overlap.
- The previous-floor rule depended on row order. It is now deterministic and
  the fast labeler is regression-tested against the reference implementation on
  randomized overlapping sequences.
- The prompt said BC or interruption *begins*, but the targets were chunk states.
  The prompt now asks what state is present during `[t,t+40 ms]`.
- The old 500 rows came from only `SBC007`, `SBC017`, and `SBC058`. They represented
  only 91/81/100/81/11 distinct C/BC/T/I/NA events respectively; in particular,
  100 NA rows repeatedly sampled only 11 continuous events.
- 344/500 old rows ended inside an ongoing human transcript unit, but the text
  input included completed units only. This is causal, but it is not a genuine
  streaming-ASR prefix and must not be described as one.

## VAD caveat

A generic mono WebRTC VAD flags many SBC058 NA chunks as speech because the source
contains explicitly annotated water and dish noise. That does not prove those NA
labels are wrong: NA means no human participant speaks. Formal label generation
therefore needs speaker-aware activity/diarization or the person annotations; a
generic speech/no-speech VAD cannot be treated as ground truth.

## Implemented v2 safeguards

- Rebuilt the catalog and recovered both fused TRN rows.
- Preserved bracketed lexical content and cleaned only non-lexical annotations.
- Unified fast and reference five-class chunk-state semantics.
- Added `event_manifest.jsonl`: one grid-aligned representative frame per
  continuous weak event.
- Prepared a 500-event candidate set with 100 proposed rows per class across all
  16 core dyadic conversations. Using all 16 is valid for this zero-shot pilot
  because the checkpoint was not trained on SBCSAE.
- Added `label_quality` to every run. The current candidate set states
  `human_gold_samples=0`, `weak_label_samples=500`, and
  `formal_claim_allowed=false`.
- Added a local review page with audio, completed-unit transcript, keyboard labels,
  uncertain marking, notes, browser-local progress, and JSON export.
- Invalidated the previous report and summary in the repository.
- The current automated suite has 46 passing tests.

## Required gate before rerunning the MLLM

Open the generated `review.html`, review the candidate targets, export
`reviewed_labels.json`, and apply it with `code/scripts/review_labels.py`. Only a
run regenerated from the reviewed manifest may be scored as the 500-event prompt
experiment. A second annotator should review at least the ambiguous BC/I/T subset;
the single-annotator manifest records that limitation explicitly.

The v2 targets are still candidate weak labels before that review. The rebuild
removes known software errors; it does not manufacture human ground truth.
