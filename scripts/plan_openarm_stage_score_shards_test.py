import json

from scripts import plan_openarm_stage_score_shards as planner


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_shard_manifest_filters_and_balances_by_frames(tmp_path):
    dataset = tmp_path / "dataset"
    _write_json(dataset / "meta/info.json", {"total_episodes": 5})
    _write_jsonl(
        dataset / "meta/episodes.jsonl",
        [{"episode_index": index, "length": length} for index, length in enumerate((100, 90, 80, 70, 60))],
    )
    annotations = dataset / "annotations/stage.jsonl"
    _write_jsonl(
        annotations,
        [{"episode_index": index, "quality": "failure" if index == 2 else "success"} for index in range(5)],
    )

    manifest = planner.build_shard_manifest(
        dataset,
        num_shards=2,
        annotations=annotations,
        quality="success",
    )

    assert manifest["selected_episode_count"] == 4
    assert manifest["selected_total_frames"] == 320
    assert manifest["excluded"] == [{"episode_index": 2, "reason": "quality=failure"}]
    assert sorted(index for shard in manifest["shards"] for index in shard["episodes"]) == [0, 1, 3, 4]
    assert [shard["total_frames"] for shard in manifest["shards"]] == [160, 160]
