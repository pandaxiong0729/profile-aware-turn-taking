# Audio + Causal Transcript + Profile MLLM Prompt Baseline

This is a zero-training, low-cost validation with an existing audio-capable MLLM. It is not the later adapter training experiment.

## Exact input and output

For a prediction boundary `t`, every request contains all three required inputs:

1. a 16-kHz mono conversation WAV covering `[max(0,t-30s),t]`;
2. the matching partial transcript containing only dialogue units completed by `t`, with `speaker_A/speaker_B` and timestamps;
3. a profile rendered with one fixed natural-language template.

The current SBCSAE manifest uses the causal manual-TRN proxy in `transcript_prefix`; it is not a real streaming-ASR output. A deployed system should replace this field with the ASR prefix actually available at `t`.

The only model output is one label for `[t,t+40ms]`:

- `C`: current speaker continues;
- `BC`: listener starts a short backchannel;
- `T`: floor transfers;
- `I`: non-backchannel overlap/interruption begins;
- `NA`: nobody speaks.

The response is schema-constrained to `{"label":"C|BC|T|I|NA"}`. The target and annotation evidence live only in `gold.jsonl` and never enter a request.

## Paired profile comparison

Each held-out sample is sent three times:

- `hidden`: profile unavailable;
- `given`: correct conversation profile;
- `shuffled`: another conversation's profile.

Within one sample, audio SHA-256, transcript SHA-256, prediction boundary, task instructions, output schema, and decoding parameters are identical. Only the profile text changes. `input_audit.json` verifies this contract and aborts the batch on any mismatch or future-text leakage.

## Reproduce the 500-sample run

Run from the repository root. The formal set is balanced at 100 test samples per class, producing 1,500 requests.

```powershell
$python = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "code/src"
$run = "artifacts\mllm-prompt-baseline\qwen2.5-omni-3b\audio-transcript-profile-test-100-per-class"

& $python code\scripts\run_mllm_prompt_baseline.py prepare `
  --manifest data\processed\sbcsae_mvp\manifest.jsonl `
  --output-dir $run `
  --split test `
  --max-per-class 100 `
  --context-seconds 30 `
  --max-transcript-chars 6000 `
  --seed 13

& $python code\scripts\run_mllm_prompt_baseline.py audit `
  --run-dir $run `
  --expected-samples 500 `
  --expected-per-class 100
```

Start one persistent llama.cpp multimodal server. This avoids loading the model for every request:

```powershell
& models\llama.cpp-b9987\llama-server.exe `
  -m models\huggingface\Qwen2.5-Omni-3B-GGUF\Qwen2.5-Omni-3B-Q4_K_M.gguf `
  --mmproj models\huggingface\Qwen2.5-Omni-3B-GGUF\mmproj-Qwen2.5-Omni-3B-Q8_0.gguf `
  --host 127.0.0.1 `
  --port 8091 `
  --ctx-size 8192 `
  --gpu-layers all `
  --parallel 1 `
  --jinja
```

In another terminal:

```powershell
& $python code\scripts\run_mllm_prompt_baseline.py run-server `
  --run-dir $run `
  --endpoint http://127.0.0.1:8091/v1/chat/completions `
  --model qwen2.5-omni-3b-q4_k_m `
  --timeout 180 `
  --retries 2 `
  --seed 13

& $python code\scripts\run_mllm_prompt_baseline.py score --run-dir $run
```

The runner appends one response at a time. Re-running it resumes from `responses.jsonl`.

## Files produced

- `requests.jsonl`: target-free model requests and input hashes;
- `gold.jsonl`: local-only targets used after inference;
- `input_audit.json`: class balance, causality, leakage, and paired-invariant checks;
- `responses.jsonl`: raw model outputs, parsed predictions, validity, and latency;
- `metrics.json`: Macro-F1, balanced accuracy, accuracy, per-class metrics, confusion matrices;
- `predictions.csv/json`: paired sample-level predictions;
- `profile_comparison.csv`: compact three-condition metric table;
- `paired_changes.json`: given fixes/breaks relative to hidden;
- `diagnostics.json`: output distributions, paired changes, latency, and collapse gate.

The measured 500-sample result is in `code/reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md`. Generated SBCSAE audio and raw artifacts remain gitignored.
