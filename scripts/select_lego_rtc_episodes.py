"""Select whole published LeRobot episodes by duration; metadata-only, no dataset mutation.

This generates candidates, not a claim that the demonstrations are successful.
Review task/visual quality before passing episodes.json to train_lego_full.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random


def read_episode_ids(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or not values:
        raise ValueError("Episode selection must be a nonempty JSON list")
    if any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError("Episode IDs must be unique nonnegative integers")
    return sorted(values)


def select_episodes(episodes, *, fps, hours, seed):
    if fps <= 0 or hours <= 0:
        raise ValueError("fps and hours must be positive")
    rows = list(episodes)
    ids = [row["episode_index"] for row in rows]
    if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Invalid or duplicate episode indices")
    if any(type(row["length"]) is not int or row["length"] <= 0 for row in rows):
        raise ValueError("Invalid episode lengths")
    random.Random(seed).shuffle(rows)
    selected, frames = [], 0
    for row in rows:
        selected.append(row)
        frames += row["length"]
        if frames >= hours * 3600 * fps:
            break
    if frames < hours * 3600 * fps:
        raise ValueError("Not enough data for the requested duration")
    return sorted(selected, key=lambda row: row["episode_index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Existing processed train split, never val/test")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths refused")
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=Path, help="Optional reviewed source episode ID list")
    args = parser.parse_args()
    source = args.repo.resolve()
    info_bytes = (source / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    metadata_path = source / "conversion_manifest.json"
    if metadata_path.is_file():
        episode_bytes = metadata_path.read_bytes()
        manifest = json.loads(episode_bytes)
        if manifest["split"] != "train":
            raise ValueError("Only the published train split can be selected")
        rows = manifest["episodes"]
    else:
        metadata_path = source / "meta/episodes.jsonl"
        episode_bytes = metadata_path.read_bytes()
        rows = [json.loads(line) for line in episode_bytes.splitlines() if line.strip()]
    if args.candidates:
        candidates = set(read_episode_ids(args.candidates))
        if not candidates <= {row["episode_index"] for row in rows}:
            raise ValueError("Candidate IDs missing from source metadata")
        rows = [row for row in rows if row["episode_index"] in candidates]
    selected = select_episodes(rows, fps=info["fps"], hours=args.hours, seed=args.seed)
    report = {
        "source_repo": str(source),
        "source_info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "source_episode_metadata": str(metadata_path),
        "source_episodes_sha256": hashlib.sha256(episode_bytes).hexdigest(),
        "seed": args.seed,
        "fps": info["fps"],
        "frames": sum(row["length"] for row in selected),
        "review_status": "candidate: verify success, task diversity and held-out split before training",
        "episodes": selected,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "episodes.json").write_text(json.dumps([row["episode_index"] for row in selected]) + "\n")
    (args.output / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Selected {len(selected)} complete episodes, {report['frames'] / info['fps'] / 3600:.3f} hours")


if __name__ == "__main__":
    main()
