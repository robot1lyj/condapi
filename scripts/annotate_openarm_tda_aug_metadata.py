"""Annotate an existing OpenArm TDA augmented dataset with source mapping metadata.

This repairs older `openarm_hq_tda_aug_v1` directories that were generated
before `source_episode_index` and `augmentation_type` were written into
`meta/episodes.jsonl`. It does not touch parquet files or videos.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import pathlib
from typing import Any


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mapping_for_episode(
    episode_index: int,
    *,
    source: str,
    source_episodes: int,
    time_scaled_episodes: int,
    extraction_factor: int,
) -> dict[str, Any]:
    if episode_index < source_episodes:
        return {
            "source_dataset": source,
            "source_episode_index": episode_index,
            "augmentation_type": "original",
            "source_frame_stride": 1,
            "source_frame_offset": 0,
            "mirror": False,
        }
    if episode_index < source_episodes + time_scaled_episodes:
        return {
            "source_dataset": source,
            "source_episode_index": episode_index - source_episodes,
            "augmentation_type": "time",
            "source_frame_stride": extraction_factor,
            "source_frame_offset": 0,
            "mirror": False,
        }
    return {
        "source_dataset": source,
        "source_episode_index": episode_index - source_episodes - time_scaled_episodes,
        "augmentation_type": "mirror",
        "source_frame_stride": 1,
        "source_frame_offset": 0,
        "mirror": True,
    }


def annotate_dataset(
    dataset: pathlib.Path,
    *,
    source: str | None,
    source_episodes: int | None,
    time_scaled_episodes: int | None,
    extraction_factor: int,
    dry_run: bool,
) -> dict[str, Any]:
    report_path = dataset / "augment_report.json"
    report = _load_json(report_path)
    source = source or report["source"]
    source_episodes = source_episodes if source_episodes is not None else int(report["source_episodes"])
    time_scaled_episodes = (
        time_scaled_episodes if time_scaled_episodes is not None else int(report["time_scaled_episodes"])
    )
    extraction_factor = int(report.get("time_extraction_factor", extraction_factor))

    episodes_path = dataset / "meta/episodes.jsonl"
    episodes = _load_jsonl(episodes_path)
    type_counts = {"original": 0, "time": 0, "mirror": 0}
    annotated = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        mapping = _mapping_for_episode(
            episode_index,
            source=source,
            source_episodes=source_episodes,
            time_scaled_episodes=time_scaled_episodes,
            extraction_factor=extraction_factor,
        )
        type_counts[mapping["augmentation_type"]] += 1
        annotated.append({**row, **mapping})

    summary = {
        "dataset": str(dataset),
        "source": source,
        "source_episodes": source_episodes,
        "time_scaled_episodes": time_scaled_episodes,
        "time_extraction_factor": extraction_factor,
        "total_episodes": len(episodes),
        "type_counts": type_counts,
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    _write_jsonl(episodes_path, annotated)
    report["time_extraction_factor"] = extraction_factor
    report["source_mapping"] = {
        "episodes_jsonl_fields": [
            "source_dataset",
            "source_episode_index",
            "augmentation_type",
            "source_frame_stride",
            "source_frame_offset",
            "mirror",
        ],
        "type_counts": type_counts,
    }
    _write_json(report_path, report)
    _write_json(dataset / "source_mapping_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--source-episodes", type=int, default=None)
    parser.add_argument("--time-scaled-episodes", type=int, default=None)
    parser.add_argument("--extraction-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    annotate_dataset(
        args.dataset.resolve(),
        source=args.source,
        source_episodes=args.source_episodes,
        time_scaled_episodes=args.time_scaled_episodes,
        extraction_factor=args.extraction_factor,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
