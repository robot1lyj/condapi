import json
import sys

import pytest
from select_lego_rtc_episodes import main
from select_lego_rtc_episodes import select_episodes


def test_selection_reproducible_whole_episodes():
    rows = [{"episode_index": i, "length": 900, "tasks": ["sort lego"]} for i in range(20)]
    chosen = select_episodes(rows, fps=30, hours=0.05, seed=42)
    assert chosen == select_episodes(rows, fps=30, hours=0.05, seed=42)
    assert len(chosen) == 6
    assert all(row in rows for row in chosen)
    assert rows[0]["episode_index"] == 0


def test_not_enough_data():
    with pytest.raises(ValueError, match="Not enough"):
        select_episodes([], fps=30, hours=2, seed=42)


def test_cli_preserves_source_and_refuses_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "train"
    meta = source / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text('{"fps": 30}')
    raw = json.dumps({"episode_index": 7, "length": 216000, "tasks": ["sort lego"]}) + "\n"
    (meta / "episodes.jsonl").write_text(raw)
    output = tmp_path / "subset"
    monkeypatch.setattr(sys, "argv", ["select", "--repo", str(source), "--output", str(output)])
    main()
    assert json.loads((output / "episodes.json").read_text()) == [7]
    assert (meta / "episodes.jsonl").read_text() == raw
    with pytest.raises(FileExistsError):
        main()


def test_published_v3_conversion_manifest(tmp_path, monkeypatch):
    source = tmp_path / "train"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text('{"fps": 30, "codebase_version": "v3.0"}')
    manifest = {"split": "train", "episodes": [{"episode_index": 3, "length": 1080000}]}
    (source / "conversion_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "ten_hours"
    monkeypatch.setattr(sys, "argv", ["select", "--repo", str(source), "--output", str(output), "--hours", "10"])
    main()
    assert json.loads((output / "episodes.json").read_text()) == [3]
    report = json.loads((output / "selection.json").read_text())
    assert report["frames"] == 1080000
    assert report["source_episode_metadata"].endswith("conversion_manifest.json")
