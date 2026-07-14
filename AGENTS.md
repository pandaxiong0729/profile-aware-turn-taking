# Research experiment invariants

- ALWAYS preserve the primary model input contract: causal audio + the matching causal partial transcript + profile.
- In `hidden/given/shuffled` comparisons, ONLY profile content may change. Audio, transcript, sample IDs, prediction boundary, instructions, and decoding settings MUST remain identical.
- NEVER include future audio, future transcript, target labels, or annotation evidence in a model request.
- Before a batch run, MUST audit class counts, causal transcript timestamps, audio SHA-256 equality, transcript SHA-256 equality, target absence, and the exact output schema.
- NEVER present a hardware or pipeline smoke test as profile-effect evidence. The hidden baseline MUST first show valid, non-collapsed, audio-sensitive behavior.
