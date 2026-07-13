# Qwen2.5-Omni-3B Audio + Profile Prompt Pilot

## Outcome

The intended audio MLLM pipeline ran successfully on the local RTX 5060 Laptop GPU. All 15 requests were valid, but the model predicted `I` for every sample under every profile condition. On this five-sample smoke pilot, adding the correct profile did not improve predictions and did not change a single prediction.

This is a negative smoke-test result, not a statistically supported conclusion about profile conditioning. The pilot has one weakly labelled sample per class and uses a general quantized MLLM that was not trained for 40-ms turn-taking forecasting.

## Experimental contract

- Input audio: the same 30-second, mono, 16-kHz causal WAV `[t-30 s, t]` in all three conditions.
- Prediction horizon: `[t, t+40 ms]`.
- Text input: fixed label instructions plus the profile condition.
- `hidden`: profile unavailable.
- `given`: correct conversation profile.
- `shuffled`: another conversation's profile.
- Excluded from model input: transcript, label, annotation evidence, and future audio.
- Training or fine-tuning: none.

Each sample's three requests have one identical audio SHA-256. Targets are stored only in the separate local `gold.jsonl` file.

## Runtime

- Model: `ggml-org/Qwen2.5-Omni-3B-GGUF`
- Pinned revision: `75f1b73b657a50f5092502799457ccb4a4a1f9df`
- Quantization: `Qwen2.5-Omni-3B-Q4_K_M.gguf`
- Audio projector: `mmproj-Qwen2.5-Omni-3B-Q8_0.gguf`
- Runtime: llama.cpp `b9987` (`ad8d82199`), `llama-mtmd-cli`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB
- Observed peak VRAM: 4,022 MiB

The complete model download is approximately 3.64 GB. The first measured request took 11.21 seconds; across the 15 recorded requests, mean latency was 5.97 seconds and median latency was 4.17 seconds. Each request used a separate CLI process, so a persistent server should be used for a larger run.

## Results

| Profile condition | Macro-F1 | Balanced accuracy | Accuracy | Prediction distribution |
|---|---:|---:|---:|---|
| hidden | 0.0667 | 0.2000 | 0.2000 | I: 5 |
| given | 0.0667 | 0.2000 | 0.2000 | I: 5 |
| shuffled | 0.0667 | 0.2000 | 0.2000 | I: 5 |

| Sample | Weak target | hidden | given | shuffled |
|---|---|---|---|---|
| SBC007-001184600 | NA | I | I | I |
| SBC007-001243680 | T | I | I | I |
| SBC017-000760840 | BC | I | I | I |
| SBC017-001158400 | C | I | I | I |
| SBC058-001070680 | I | I | I | I |

Paired changes: the correct profile fixed zero hidden errors and broke zero hidden-correct cases. One sample was correct under both hidden and given because its target was `I`; four were wrong under both.

## Audio-path sanity check

On sample `SBC017-000760840`, a separate prompt asking for the final audible utterance returned `yeah`. That word is present in the sample's causal transcript context, so the model was receiving and processing audio rather than operating as a text-only model. This is only a pathway check, not an ASR-accuracy evaluation.

llama.cpp also reports that audio support is experimental and may have reduced quality. Together with the all-`I` collapse, this means the current small quantized checkpoint is useful for validating the experiment code but is not a credible final baseline for profile efficacy.

## Reproduction

See `code/docs/MLLM_PROMPT_BASELINE.md`. Local generated requests, WAV clips, raw responses, metrics, and predictions are under:

`artifacts/mllm-prompt-baseline/qwen2.5-omni-3b/pilot-1-per-class/`

The downloaded model and generated SBCSAE clips remain gitignored and are not published.
