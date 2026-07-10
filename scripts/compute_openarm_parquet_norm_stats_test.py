import json

from scripts import compute_openarm_parquet_norm_stats as norm_stats


def test_write_json_atomic_replaces_complete_file(tmp_path):
    output = tmp_path / "norm_stats.json"
    output.write_text('{"stale": true}\n')

    norm_stats._write_json_atomic(output, {"norm_stats": {"state": {"mean": [0.0]}}})  # noqa: SLF001

    assert json.loads(output.read_text()) == {"norm_stats": {"state": {"mean": [0.0]}}}
    assert not list(tmp_path.glob(".norm_stats.json.tmp-*"))
