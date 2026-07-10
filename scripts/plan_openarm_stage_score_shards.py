"""Create frame-balanced episode shards for resumable OpenArm Stage scoring."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
from typing import Any


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def build_shard_manifest(
    dataset: pathlib.Path,
    *,
    num_shards: int,
    annotations: pathlib.Path | None,
    quality: str | None,
) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    dataset = dataset.resolve()
    info = _load_json(dataset / "meta/info.json")
    episode_rows = {int(row["episode_index"]): row for row in _load_jsonl(dataset / "meta/episodes.jsonl")}

    annotation_rows: dict[int, dict[str, Any]] = {}
    if annotations is not None:
        annotation_rows = {int(row["episode_index"]): row for row in _load_jsonl(annotations)}

    selected = []
    excluded = []
    for episode_index, row in sorted(episode_rows.items()):
        annotation = annotation_rows.get(episode_index)
        if annotations is not None and annotation is None:
            excluded.append({"episode_index": episode_index, "reason": "missing_annotation"})
            continue
        annotation_quality = str(annotation.get("quality", "success")) if annotation is not None else None
        if quality is not None and annotation_quality != quality:
            excluded.append({"episode_index": episode_index, "reason": f"quality={annotation_quality}"})
            continue
        selected.append(
            {
                "episode_index": episode_index,
                "length": int(row["length"]),
                "quality": annotation_quality,
            }
        )

    if not selected:
        raise ValueError("No episodes matched the Stage score shard filters")
    if num_shards > len(selected):
        raise ValueError(f"num_shards={num_shards} exceeds selected episodes={len(selected)}")

    shards = [{"shard_index": index, "episodes": [], "total_frames": 0} for index in range(num_shards)]
    for episode in sorted(selected, key=lambda item: (-int(item["length"]), int(item["episode_index"]))):
        target = min(
            shards, key=lambda shard: (int(shard["total_frames"]), len(shard["episodes"]), shard["shard_index"])
        )
        target["episodes"].append(int(episode["episode_index"]))
        target["total_frames"] += int(episode["length"])

    for shard in shards:
        shard["episodes"].sort()
        shard["episode_count"] = len(shard["episodes"])
        shard["episode_spec"] = ",".join(str(index) for index in shard["episodes"])

    return {
        "schema_version": "openarm_stage_score_shards_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "dataset": str(dataset),
        "dataset_total_episodes": int(info["total_episodes"]),
        "annotations": str(annotations.resolve()) if annotations is not None else None,
        "quality_filter": quality,
        "selected_episode_count": len(selected),
        "selected_total_frames": sum(int(item["length"]) for item in selected),
        "excluded": excluded,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--annotations", type=pathlib.Path)
    parser.add_argument("--quality", default=None, help="Annotation quality to include, for example success")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    manifest = build_shard_manifest(
        args.dataset,
        num_shards=args.num_shards,
        annotations=args.annotations,
        quality=args.quality,
    )
    _write_json_atomic(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
