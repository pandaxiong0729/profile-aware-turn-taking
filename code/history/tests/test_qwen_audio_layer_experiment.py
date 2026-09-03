from __future__ import annotations

import numpy as np

from profile_turntaking.qwen_audio_layer_experiment import _savez_compressed_atomic


def test_atomic_npz_checkpoint_replaces_complete_prior_file(tmp_path) -> None:
    path = tmp_path / "cache.partial.npz"
    _savez_compressed_atomic(path, values=np.asarray([1, 2, 3], dtype=np.int64))
    with np.load(path, allow_pickle=False) as payload:
        assert payload["values"].tolist() == [1, 2, 3]

    _savez_compressed_atomic(path, values=np.asarray([4, 5], dtype=np.int64))
    with np.load(path, allow_pickle=False) as payload:
        assert payload["values"].tolist() == [4, 5]
    assert not path.with_name(f"{path.stem}.tmp.npz").exists()
