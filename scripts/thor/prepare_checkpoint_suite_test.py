import json

from prepare_checkpoint_suite import digest
from prepare_checkpoint_suite import prepare
import pytest


def test_rebind_preserves_observation_and_uses_checkpoint_norm(tmp_path):
    source, checkpoint, output = [tmp_path / name for name in ("source", "checkpoint", "output")]
    source.mkdir()
    (checkpoint / "assets/yam").mkdir(parents=True)
    (checkpoint / "params").mkdir()
    (checkpoint / "assets/yam/norm_stats.json").write_text('{"new": 1}')
    (checkpoint / "params/_METADATA").write_text("metadata")
    (source / "sample.npz").write_bytes(b"immutable-observation")
    (source / "sample.json").write_text(json.dumps({"sample_sha256": digest(source / "sample.npz")}))
    (source / "suite.json").write_text(json.dumps({"samples": [{"sample": "sample.npz", "provenance": "sample.json"}]}))
    prepare(source, checkpoint, output)
    result = json.loads((output / "suite.json").read_text())
    assert result["norm_stats_sha256"] == digest(checkpoint / "assets/yam/norm_stats.json")
    assert digest(output / "sample.npz") == digest(source / "sample.npz")
    assert not result["base_model_only"]
    assert json.loads((output / "sample.json").read_text())["norm_stats_sha256"] == result["norm_stats_sha256"]
    with pytest.raises(FileExistsError):
        prepare(source, checkpoint, output)
