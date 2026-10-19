"""Reject verified source episodes using their repository, revision and split identity.

This is a metadata check, not a completeness claim for Pantheon's annotations.
Source files and frozen experiment selections are never modified here.
"""

import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "configs/datasets/yam-excluded-episodes.json"


def find_excluded_episodes(rows, *, split="train", registry=None):
    record = json.loads(Path(registry or REGISTRY).read_text())
    if record["schema_version"] != 1:
        raise ValueError("Unsupported YAM episode exclusion registry")
    excluded = {}
    for item in record["episodes"]:
        key = (item["source_repo"], item["source_revision"], item["split"], item["source_episode_index"])
        if (
            type(key[3]) is not int
            or key[3] < 0
            or item["identity_verified"] is not True
            or not isinstance(item["flags"], list)
            or not item["flags"]
            or any(not isinstance(flag, str) or not flag for flag in item["flags"])
        ):
            raise ValueError("Exclusions require verified source identity and explicit QC evidence")
        if key in excluded:
            raise ValueError("Duplicate source identity in YAM episode exclusions")
        excluded[key] = item
    matches = []
    for row in rows:
        key = (row.get("source_repo"), row.get("source_revision"), split, row.get("source_episode_index"))
        if key in excluded:
            matches.append({**excluded[key], "episode_index": row["episode_index"]})
    return matches


def require_no_excluded_episodes(repo, selected_ids=None, *, registry=None):
    manifest_path = Path(repo) / "conversion_manifest.json"
    if not manifest_path.is_file():
        return  # Other native/legacy datasets have no verified source mapping here.
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("episodes", [])
    if selected_ids is not None:
        selected = set(selected_ids)
        rows = [row for row in rows if row["episode_index"] in selected]
    matches = find_excluded_episodes(rows, split=manifest["split"], registry=registry)
    if matches:
        ids = sorted(row["episode_index"] for row in matches)
        raise ValueError(
            f"Verified QC-excluded episodes selected: {ids}. "
            "Publish a new reviewed selection and its norm/recipe; keep the original files and run unchanged."
        )
