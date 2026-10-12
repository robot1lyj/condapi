"""Create the server's research dataset catalog; preserve raw recordings in place."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--classification", type=Path, required=True)
    args = p.parse_args()
    directory = args.project_root / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.replay / "manifest.json").read_text())
    catalog = {
        "schema": "pld_research_dataset_catalog_v1",
        "dataset_id": args.replay.name,
        "raw": {
            "path": str(args.raw_root.resolve()),
            "immutable": True,
            "note": "existing verified server backup; no duplicate/rewrite/deletion",
        },
        "classification": {
            "path": str(args.classification.resolve()),
            "candidates": 342,
            "candidate_sha256": manifest["source_candidates_sha256"],
        },
        "rewards": {
            "path": str(args.replay / "rewards.json"),
            "sha256": manifest["reward_sha256"],
            "eligible": manifest["approved_rewards"],
            "rule": "attempt terminal 1 held / 0 empty; invalid evidence excluded",
        },
        "replay": {
            "path": str(args.replay),
            "sha256": manifest["replay_sha256"],
            "train": "E02-E14",
            "validation": "E01",
            "training_attempts": manifest["train_attempts"],
            "validation_attempts": manifest["validation_attempts"],
            "all_outcomes_retained": True,
            "not_online_ready": manifest["not_online_ready"],
        },
        "features": json.loads((args.replay / "feature-manifest.json").read_text()),
        "classification_is_not_independent_ground_truth": True,
        "retention": "keep all raw, labels, exclusions, configs, logs, checkpoints and evaluations; new evidence creates a new version",
    }
    versions = directory / "catalog"
    versions.mkdir(exist_ok=True)
    path = versions / (args.replay.name + ".json")
    if path.exists():
        raise FileExistsError("published dataset catalog entry is immutable")
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    # The index is navigational; immutable version records own the actual facts.
    index = [{"dataset_id": x.stem, "path": str(x), "sha256": sha(x)} for x in sorted(versions.glob("*.json"))]
    (directory / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
