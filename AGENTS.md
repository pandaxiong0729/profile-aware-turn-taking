# Research experiment invariants

- ALWAYS preserve the primary model input contract: causal audio + the matching causal partial transcript + profile.
- In `hidden/given/shuffled` comparisons, ONLY profile content may change. Audio, transcript, sample IDs, prediction boundary, instructions, and decoding settings MUST remain identical.
- NEVER include future audio, future transcript, target labels, or annotation evidence in a model request.
- Before a batch run, MUST audit class counts, causal transcript timestamps, audio SHA-256 equality, transcript SHA-256 equality, target absence, and the exact output schema.
- NEVER present a hardware or pipeline smoke test as profile-effect evidence. The hidden baseline MUST first show valid, non-collapsed, audio-sensitive behavior.
- When a modality control replaces only audio but keeps the correct causal transcript, interpret unchanged predictions only as low incremental audio sensitivity given that transcript. NEVER infer that the model cannot judge the task or fully ignores audio from that control alone.
- ALWAYS name each profile representation by its actual fields. The current 59-dimensional main profile view is `causal dynamic interaction statistics + relationship/situation categories`; it does NOT contain age, gender, occupation, education, ethnicity, or other static speaker metadata. NEVER call it a complete static speaker profile. Report static-only, dynamic-only, and combined variants separately when they exist.
