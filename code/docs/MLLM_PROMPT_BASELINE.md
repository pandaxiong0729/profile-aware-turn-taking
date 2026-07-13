# Audio + Profile MLLM Prompt Pilot

This is the intended low-cost validation: an existing audio-capable MLLM receives a causal mono WAV clip plus a fixed-template profile and predicts one five-class turn-taking label. It performs inference only; it does not train or fine-tune the MLLM.

For every held-out sample, the runner creates three paired requests:

- `hidden`: identical audio, profile unavailable;
- `given`: identical audio, correct conversation profile;
- `shuffled`: identical audio, a different conversation's profile.

The WAV contains `[t-context, t]` only and ends exactly at the prediction boundary. The target is the event in `[t, t+40 ms]`. `transcript_prefix`, labels, annotation evidence, and future audio are never model inputs. `requests.jsonl` is target-free; `gold.jsonl` remains separate for local scoring.

## Local pilot used in this repository

- Model: `ggml-org/Qwen2.5-Omni-3B-GGUF`, revision `75f1b73b657a50f5092502799457ccb4a4a1f9df`
- Main weights: `Qwen2.5-Omni-3B-Q4_K_M.gguf`
- Audio projector: `mmproj-Qwen2.5-Omni-3B-Q8_0.gguf`
- Runtime: `llama-mtmd-cli` from llama.cpp

The model and generated clips stay under ignored `models/` and `artifacts/` directories.

## Commands

Run from the repository root after installing the package in editable mode.

```powershell
$python = ".\.venv\Scripts\python.exe"
$run = "artifacts\mllm-prompt-baseline\qwen2.5-omni-3b\pilot-1-per-class"

& $python code\scripts\run_mllm_prompt_baseline.py prepare `
  --manifest data\processed\sbcsae_mvp\manifest.jsonl `
  --output-dir $run `
  --split test `
  --max-per-class 1 `
  --context-seconds 30

& $python code\scripts\run_mllm_prompt_baseline.py run `
  --run-dir $run `
  --executable models\llama.cpp-b9987\llama-mtmd-cli.exe `
  --model models\huggingface\Qwen2.5-Omni-3B-GGUF\Qwen2.5-Omni-3B-Q4_K_M.gguf `
  --mmproj models\huggingface\Qwen2.5-Omni-3B-GGUF\mmproj-Qwen2.5-Omni-3B-Q8_0.gguf

& $python code\scripts\run_mllm_prompt_baseline.py score --run-dir $run
```

Use `--limit 1` on the `run` command for a one-request hardware check. Re-running the command resumes from `responses.jsonl` rather than repeating completed requests.

## Interpretation

This tiny pilot checks whether the pipeline works and whether profile text changes predictions. It is not evidence of a statistically reliable improvement. The formal comparison must use a larger held-out set, paired confidence intervals, and manually checked labels for the priority classes.
