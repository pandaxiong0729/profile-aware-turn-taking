# Audio + Causal Transcript + Profile MLLM Prompt Baseline

This is a zero-training, low-cost validation with an existing audio-capable MLLM. It is not the later adapter training experiment.

## Exact input and output

For a prediction boundary `t`, every request contains all three required inputs:

1. a 16-kHz mono conversation WAV covering `[max(0,t-30s),t]`;
2. the matching completed-unit transcript containing only dialogue units completed by `t`, with `speaker_A/speaker_B` and timestamps;
3. a profile rendered with one fixed natural-language template.

The current SBCSAE manifest uses the causal manual-TRN proxy in `transcript_prefix`; it is not a real streaming-ASR output. A deployed system should replace this field with the ASR prefix actually available at `t`.

The only model output is one label for `[t,t+40ms]`:

- `C`: current speaker continues;
- `BC`: a short listener backchannel is present while the other participant keeps the floor;
- `T`: floor transfers;
- `I`: both participants speak and the overlap is not a backchannel;
- `NA`: nobody speaks.

The response is schema-constrained to `{"label":"C|BC|T|I|NA"}`. The target and annotation evidence live only in `gold.jsonl` and never enter a request.

## Paired profile comparison

Each held-out sample is sent three times:

- `hidden`: profile unavailable;
- `given`: correct conversation profile;
- `shuffled`: another conversation's profile.

Within one sample, audio SHA-256, transcript SHA-256, prediction boundary, task instructions, output schema, and decoding parameters are identical. Only the profile text changes. `input_audit.json` verifies this contract and aborts the batch on any mismatch or future-text leakage.

## Prepare the 500-event review set

Run from the repository root. Select the onset frame from each distinct
weak event across all 16 core conversations. This is allowed here because the
existing MLLM is zero-shot and was not trained on SBCSAE. These 500 rows remain
candidate weak labels until review; `run_config.json` must report
`formal_claim_allowed: true` before the result is cited.

```powershell
$python = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "code/src"
$run = "artifacts\mllm-prompt-baseline\qwen2.5-omni-3b\onset-500-review-required"

& $python code\scripts\run_mllm_prompt_baseline.py prepare `
  --manifest data\processed\sbcsae_mvp_v2\event_onset_manifest.jsonl `
  --output-dir $run `
  --split all `
  --max-per-class 100 `
  --context-seconds 30 `
  --max-transcript-chars 6000 `
  --seed 13

& $python code\scripts\run_mllm_prompt_baseline.py audit `
  --run-dir $run `
  --expected-samples 500 `
  --expected-per-class 100

& $python code\scripts\review_labels.py build `
  --run-dir $run `
  --source-manifest data\processed\sbcsae_mvp_v2\event_onset_manifest.jsonl `
  --catalog-dir data\processed\sbcsae_catalog_v2
```

Open `$run\review.html`. The review clip intentionally contains audio before and
after `t`, because an annotator must hear the target evidence. It is marked
annotation-only and is never referenced by `requests.jsonl`. Label with keys
`1–5`, use `U` for uncertain cases, and export `reviewed_labels.json`. The page
also shows time-relative transcript units around the boundary and puts samples
with non-lexical/environment target evidence first. These fields live only in
`review_items.json` and `review.html`; they never enter a model request. Apply
the complete review:

```powershell
& $python code\scripts\review_labels.py apply `
  --source-manifest data\processed\sbcsae_mvp_v2\event_onset_manifest.jsonl `
  --review-json $run\reviewed_labels.json `
  --output-manifest data\processed\sbcsae_mvp_v2\reviewed_500.jsonl `
  --reviewer-id annotator-1
```

Regenerate the run from `reviewed_500.jsonl`; its `run_config.json` must show 500
human-reviewed samples. Do not start the model before this gate passes. Then start
one persistent llama.cpp server:

```powershell
$run = "artifacts\mllm-prompt-baseline\qwen2.5-omni-3b\onset-500-reviewed"
& $python code\scripts\run_mllm_prompt_baseline.py prepare `
  --manifest data\processed\sbcsae_mvp_v2\reviewed_500.jsonl `
  --output-dir $run `
  --split all `
  --max-per-class 0 `
  --context-seconds 30 `
  --seed 13
```

`--max-per-class 0` retains all 500 reviewed sample IDs even if human corrections
make the final class counts unequal. Rebalancing after review would silently change
the predeclared evaluation set.

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
- `bootstrap_95ci.json`: 2,000-resample conversation-level bootstrap 95% intervals;
- `diagnostics.json`: output distributions, paired changes, latency, and collapse gate.

Because the schema returns one hard class rather than five probabilities, this
pilot does not report ROC-AUC, Brier score, or ECE. It also samples event onsets
instead of decoding a complete 40 ms timeline, so ±200 ms event-level F1 belongs
to the later streaming adapter experiment, not this prompt pilot.

The earlier measured result in
`code/reports/MLLM_PROMPT_QWEN2_5_OMNI_3B_REPORT.md` is explicitly invalidated.
Generated SBCSAE audio and raw artifacts remain gitignored.
