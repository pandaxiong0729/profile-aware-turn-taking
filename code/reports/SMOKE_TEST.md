# Smoke test report

Date: 2026-07-13
Environment: Windows, Python 3.12, CPU-only PyTorch

## Verification completed

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\profile-turntaking.exe smoke --bundled-fixture --work-dir artifacts\smoke-bundled-five
.\.venv\Scripts\profile-turntaking.exe smoke --work-dir artifacts\smoke-final-real
```

Results:

- Unit/integration tests: 7 passed.
- Python bytecode compilation: passed.
- Bundled fixture: prepare, CPU train, checkpoint reload and three-mode evaluation passed.
- Local SBC041: parsed 117 real TRN utterance units from the first 120 seconds, generated aligned mono synthetic audio, trained a checkpoint and evaluated it successfully.

## Five-class fixture coverage

| Label | Prepared samples |
| --- | ---: |
| `C` | 64 |
| `BC` | 54 |
| `T` | 10 |
| `I` | 11 |
| `NA` | 39 |

The bundled fixture covers all five code paths. Its 24-sample smoke test produced the expected `profile_comparison.csv` with hidden/given/shuffled rows and a `given_minus_hidden` row.

## Local SBC041 pipeline

| Item | Result |
| --- | --- |
| Transcript source | Real `SBC041.trn` timestamps/text |
| Profile source | `../intro/sbcsae_profile_turntaking_training_example.json` |
| Audio | Synthetic mono WAV aligned to real intervals |
| Prepared samples | 186 |
| Labels observed | `C`, `BC`, `T`, `I`, `NA` |
| Best validation Macro-F1 | 0.6275 |
| Hidden test Macro-F1 | 0.4668 |
| Given test Macro-F1 | 0.4079 |

These numbers are not research results. The run uses one conversation, synthetic audio, 3-second context, heuristic TRN labels, and a smoke-only stratified split. In particular, the `Given < Hidden` result is neither surprising nor interpretable because every sample has the same profile and the audio contains synthetic tones.

## Remaining scientific-data gate

Before reporting the experiment described in the research Markdown:

1. download real SBCSAE WAV files;
2. use at least three speaker-connected split groups;
3. replace TRN-only silence boundaries with audio VAD/overlap detection;
4. run 30-second context with a frozen Whisper encoder;
5. retain natural test distribution and inspect rare-class support.
