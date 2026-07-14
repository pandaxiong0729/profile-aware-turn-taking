# Qwen2.5-Omni-3B Three-Input Prompt Experiment (500 samples)

## Outcome

The corrected experiment completed 1,500/1,500 valid multimodal requests on 500 balanced held-out SBCSAE samples. Every request used causal audio + causal partial transcript + profile and predicted the next-40-ms five-class label.

The correct profile did **not** produce a statistically or scientifically credible improvement with this checkpoint. Given-profile accuracy was 20.2% versus 19.4% hidden (+0.8 percentage points), but Macro-F1 decreased from 0.0823 to 0.0803. The paired exact McNemar p-value was 0.503. The hidden baseline predicted `I` on 89% of samples and never predicted `BC`, `T`, or `NA`, so it failed the precondition for interpreting a profile effect.

This is a valid negative result for the tested zero-shot checkpoint, not evidence that profiles are useless. It shows that Qwen2.5-Omni-3B Q4 with this prompt is not a credible final turn-taking baseline.

## Audited experimental contract

- Split: SBCSAE `test` only.
- Samples: 500, exactly 100 each for `C/BC/T/I/NA`.
- Requests: 1,500 (`hidden/given/shuffled` for every sample).
- Audio input: 16-kHz mono `[max(0,t-30s),t]`.
- Text input: matching causal `transcript_prefix`, only units completed by `t`, with speaker/timestamps.
- Profile input: fixed-template natural language; unavailable, correct, or shuffled by condition.
- Output: exactly one of `C/BC/T/I/NA` for `[t,t+40ms]`.
- Training/fine-tuning: none.
- Excluded: future audio/text, target, label evidence, annotation reasons.

`input_audit.json` passed with zero errors and zero warnings. For each sample, the three requests had identical audio hash, transcript hash, boundary, prompt-template hash, horizon, and output schema; only profile text changed. The current transcript is a causal manual-TRN proxy, not live streaming ASR.

## Metrics

| Profile condition | Macro-F1 | Balanced accuracy | Accuracy | Correct / 500 |
|---|---:|---:|---:|---:|
| hidden | 0.0823 | 0.1940 | 0.1940 | 97 |
| given | 0.0803 | 0.2020 | 0.2020 | 101 |
| shuffled | 0.0746 | 0.2000 | 0.2000 | 100 |

Prediction distributions reveal the failure mode:

| Condition | C | BC | T | I | NA | Dominant-label rate |
|---|---:|---:|---:|---:|---:|---:|
| hidden | 55 | 0 | 0 | 445 | 0 | 89.0% `I` |
| given | 23 | 0 | 0 | 477 | 0 | 95.4% `I` |
| shuffled | 24 | 0 | 1 | 475 | 0 | 95.0% `I` |

Per-class F1:

| Condition | C | BC | T | I | NA |
|---|---:|---:|---:|---:|---:|
| hidden | 0.0774 | 0 | 0 | 0.3339 | 0 |
| given | 0.0650 | 0 | 0 | 0.3362 | 0 |
| shuffled | 0.0323 | 0 | 0 | 0.3409 | 0 |

The given profile fixed 12 hidden errors and broke 8 hidden-correct predictions; 89 samples were correct in both conditions and 391 wrong in both. Hidden versus given changed 66/500 predictions (13.2%), but shuffled profile also changed 64/500 (12.8%). Therefore change alone is not evidence that the correct profile supplied useful information.

## Audio sensitivity diagnostic

A separate diagnostic randomly selected 50 hidden-profile requests and replaced only the WAV with duration-matched digital silence. Transcript, profile, task prompt, boundary, and decoding stayed unchanged. Predictions changed on 7/50 samples (14%):

- original audio: `I=45`, `C=5`;
- silenced audio: `I=44`, `C=6`.

This low change fraction suggests weak audio sensitivity under this checkpoint/prompt. It does not prove the model ignored audio, but together with the label collapse it reinforces that the run cannot support a profile-efficacy claim.

## Runtime

- Model: `ggml-org/Qwen2.5-Omni-3B-GGUF`, revision `75f1b73b657a50f5092502799457ccb4a4a1f9df`.
- Main weights: `Qwen2.5-Omni-3B-Q4_K_M.gguf`.
- Projector: `mmproj-Qwen2.5-Omni-3B-Q8_0.gguf`.
- Runtime: llama.cpp `b9987`, persistent `/v1/chat/completions` server using `input_audio`.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB; observed server usage about 4,658 MiB.
- 1,500-request wall time: about 22.3 minutes; throughput about 1.12 requests/s.
- Per-request latency: median 332 ms, p95 1,253 ms; one valid server stall reached 519.9 seconds and skewed the mean to 887 ms.

llama.cpp reports that audio input is experimental and may have reduced quality.

## Decision

Do not cite this run as “profile improves turn-taking.” Cite it as the corrected low-cost prompt baseline showing no reliable profile gain and a severe checkpoint/output-collapse limitation. Keep the same audited three-input interface for the next model: either test a stronger audio MLLM on a validation subset before the held-out set, or fine-tune the project model/profile adapter and compare hidden/given/shuffled on the identical 500 examples.

Reproduction instructions are in `code/docs/MLLM_PROMPT_BASELINE.md`. Raw SBCSAE clips, requests, responses, and model files stay under gitignored `artifacts/`, `data/`, and `models/` directories.
