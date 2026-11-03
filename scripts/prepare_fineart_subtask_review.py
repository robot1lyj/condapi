#!/usr/bin/env python3
"""Prepare Flora review sidecars; export language rows only after explicit review.

No model, dataset writer, video decoder, or training loop. All intervals are
half-open parent frame intervals. Native timestamps require actual per-frame
parquet timestamps supplied in a separate hash-bound JSON input.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import pairwise
import json
import math
from pathlib import Path
import re
import sys

FPS = 30
FLAG_TEXT = re.compile(
    r"\b(?:unknown|mistake|uncertain|unrecognized|unrecognised|unlabeled|unlabelled|error)\b|^\s*(?:n/?a|none|null)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return json.loads(content), {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def save_new(path, data):
    path = Path(path)
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def integer(value, name):
    require(type(value) is int, f"{name} must be an integer frame index")
    return value


def caption_problem(content):
    if not isinstance(content, str) or not content.strip():
        return "EMPTY_INSTRUCTION"
    if FLAG_TEXT.search(content):
        return "NON_INSTRUCTION_OR_UNCERTAIN_LABEL"
    return None


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def metadata_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        content = path.read_bytes()
        rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
        return rows, {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
    value, source = load(path)
    return value if isinstance(value, list) else value["parents"], source


def local_boundary(seconds):
    require(type(seconds) in (float, int) and math.isfinite(seconds), "Non-finite / non-numeric local boundary")
    # Select frames with local frame time f/FPS in [start_s, end_s).
    # The tiny tolerance only removes float serialization noise at exact frames.
    return math.ceil(seconds * FPS - 1e-6)


def add_issue(parent, code, **details):
    issue_id = f"{parent['pool']}:{parent['source_episode_index']}:{len(parent['issues']):04d}"
    parent["issues"].append({"id": issue_id, "code": code, **details})


def atomic_review_intervals(parent):
    length = parent["length"]
    labels = [x for x in parent["raw_labels"] if x["core_owned_interval"] is not None]
    points = {0, length}
    for label in labels:
        points.update(label["core_owned_interval"])
    for window in parent["windows"]:
        points.update([window["core_from_frame"], window["core_to_frame"]])
    points = sorted(p for p in points if 0 <= p <= length)
    result = []
    for start, end in pairwise(points):
        active = [x for x in labels if x["core_owned_interval"][0] <= start and x["core_owned_interval"][1] >= end]
        owners = [w["id"] for w in parent["windows"] if w["core_from_frame"] <= start and w["core_to_frame"] >= end]
        if len(owners) != 1:
            add_issue(
                parent,
                "MISSING_WINDOW_CORE" if not owners else "OVERLAPPING_WINDOW_CORES",
                from_frame=start,
                to_frame=end,
                window_ids=owners,
            )
        if not active:
            add_issue(parent, "UNCOVERED_LABEL_FRAMES", from_frame=start, to_frame=end)
        elif len(active) > 1:
            add_issue(
                parent,
                "OVERLAPPING_RAW_LABELS",
                from_frame=start,
                to_frame=end,
                source_label_ids=[x["id"] for x in active],
            )
        content = (
            active[0]["raw_segment"].get("caption") if len(active) == 1 and active[0]["eligible_suggestion"] else None
        )
        result.append(
            {
                "from_frame": start,
                "to_frame": end,
                "content": content,
                "source_label_ids": [x["id"] for x in active],
                "review_status": "UNREVIEWED",
                "reviewer": None,
                "note": "",
            }
        )
    return result


def make_draft(lineage_file, parent_metadata, results_dir):
    lineage, lineage_source = load(lineage_file)
    rows, metadata_source = metadata_rows(parent_metadata)
    windows = lineage["windows"]
    require(isinstance(windows, list) and windows, "No actual windows in lineage; a design plan is not lineage")
    require(lineage.get("annotation_windows", len(windows)) == len(windows), "Lineage window count mismatch")
    index = {}
    for row in rows:
        eid = row.get("source_episode_index", row.get("episode_index"))
        require(type(eid) is int, "Parent metadata needs source_episode_index / episode_index")
        key = (row.get("pool"), eid)
        require(key not in index, f"Duplicate parent metadata {key}")
        index[key] = row
    parents = {}
    seen_windows = set()
    for receipt in windows:
        wid, pool, eid = receipt["id"], receipt["pool"], receipt["source_episode_index"]
        require(
            isinstance(wid, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", wid) and wid not in {".", ".."},
            "Unsafe window id",
        )
        require(wid not in seen_windows, f"Duplicate lineage window {wid}")
        seen_windows.add(wid)
        key = (pool, eid)
        meta = index.get(key, index.get((None, eid)))
        require(meta is not None, f"Missing actual parent length metadata {key}")
        if "videos" in meta and "status" in meta:
            record_hash = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
            require(receipt["parent_record_sha256"] == record_hash, f"Parent audit record differs from lineage {wid}")
        length = integer(meta["length"], "parent length")
        require(length > 0 and meta.get("fps", FPS) == FPS, "Only nonempty native 30 FPS parents supported")
        if key not in parents:
            parents[key] = {
                "pool": pool,
                "source_episode_index": eid,
                "length": length,
                "fps": FPS,
                "windows": [],
                "raw_labels": [],
                "issues": [],
            }
        parent = parents[key]
        require(
            receipt["fps"] == FPS and receipt.get("training_episode") is False,
            "Expected annotation-only 30 FPS lineage",
        )
        start, end, core_start, core_end = [
            integer(receipt[k], k) for k in ("parent_from_frame", "parent_to_frame", "core_from_frame", "core_to_frame")
        ]
        require(0 <= start <= core_start < core_end <= end <= length, f"Invalid lineage frame bounds {wid}")
        require(
            abs(receipt["local_to_parent_offset_s"] - start / FPS) < 1e-8, f"Incorrect local-to-parent offset {wid}"
        )
        require(set(receipt["cameras"]) == {"top", "left_wrist", "right_wrist"}, "Expected three YAM lineage cameras")
        require(
            all(
                c["frames"] == end - start and isinstance(c["sha256"], str) and c["sha256"]
                for c in receipt["cameras"].values()
            ),
            "Lineage camera frame count / hash missing",
        )
        window = {**receipt, "result_source": None, "tempseg_source": None, "result_metadata": None}
        parent["windows"].append(window)
        tempseg_path = Path(results_dir) / f"{wid}.tempseg.json"
        if tempseg_path.exists():
            _, window["tempseg_source"] = load(tempseg_path)
        result_path = Path(results_dir) / f"{wid}.json"
        if not result_path.exists():
            add_issue(parent, "MISSING_FINAL_RESULT", window_id=wid)
            continue
        result, window["result_source"] = load(result_path)
        window["result_metadata"] = {
            k: result.get(k)
            for k in (
                "schema_version",
                "duration_s",
                "profile",
                "camera_roles",
                "input_sha256",
                "model_revisions",
                "status",
            )
        }
        duration = (end - start) / FPS
        header_valid = True
        if type(result.get("duration_s")) not in (float, int) or abs(result["duration_s"] - duration) > 1e-5:
            add_issue(
                parent,
                "RESULT_DURATION_MISMATCH",
                window_id=wid,
                expected_s=duration,
                observed_s=result.get("duration_s"),
            )
            header_valid = False
        expected_hashes = {
            "main": receipt["cameras"]["top"]["sha256"],
            "left_wrist": receipt["cameras"]["left_wrist"]["sha256"],
            "right_wrist": receipt["cameras"]["right_wrist"]["sha256"],
        }
        if result.get("input_sha256") != expected_hashes:
            add_issue(parent, "RESULT_INPUT_HASH_MISMATCH", window_id=wid)
            header_valid = False
        if result.get("profile") != "YAM":
            add_issue(parent, "RESULT_PROFILE_MISMATCH", window_id=wid, profile=result.get("profile"))
            header_valid = False
        if result.get("status", "complete") != "complete":
            add_issue(parent, "INCOMPLETE_RESULT", window_id=wid, status=result["status"])
            header_valid = False
        layers = {k: result[k] for k in ("coarse_segments", "fine_segments") if k in result}
        if layers:
            window["auxiliary_raw_label_layers"] = layers
            add_issue(parent, "MULTIPLE_LABEL_LAYERS_REQUIRE_SELECTION", window_id=wid, layers=list(layers))
        segments = result.get("segments")
        if not isinstance(segments, list):
            add_issue(parent, "MISSING_SEGMENTS", window_id=wid)
            continue
        for i, segment in enumerate(segments):
            require(isinstance(segment, dict), f"Invalid native segment object {wid}/{i}")
            label = {
                "id": f"{wid}:s{i:05d}",
                "window_id": wid,
                "raw_segment": segment,
                "parent_interval": None,
                "core_owned_interval": None,
                "eligible_suggestion": False,
            }
            parent["raw_labels"].append(label)
            problem = caption_problem(segment.get("caption"))
            if problem:
                add_issue(parent, problem, source_label_id=label["id"])
            if segment.get("status") != "complete":
                add_issue(parent, "INCOMPLETE_SEGMENT", source_label_id=label["id"], status=segment.get("status"))
            try:
                a, b = segment["start_s"], segment["end_s"]
                left, right = local_boundary(a), local_boundary(b)
                require(
                    0 <= a < b <= duration + 1e-5 and 0 <= left < right <= end - start,
                    "Local segment outside input or contains no video frame",
                )
            except (KeyError, TypeError, ValueError) as error:
                add_issue(parent, "INVALID_LOCAL_INTERVAL", source_label_id=label["id"], reason=str(error))
                continue
            label["parent_interval"] = [start + left, start + right]
            owned = [max(start + left, core_start), min(start + right, core_end)]
            if owned[0] < owned[1]:
                label["core_owned_interval"] = owned
            label["eligible_suggestion"] = (
                header_valid and problem is None and segment.get("status") == "complete" and not layers
            )
    review_parents = []
    for parent in parents.values():
        suggestions = atomic_review_intervals(parent)
        review_parents.append(
            {
                "pool": parent["pool"],
                "source_episode_index": parent["source_episode_index"],
                "intervals": suggestions,
                "issue_resolutions": [
                    {"issue_id": issue["id"], "resolution": "", "reviewer": ""} for issue in parent["issues"]
                ],
            }
        )
    draft = {
        "schema_version": 1,
        "status": "DRAFT_REQUIRES_SEMANTIC_REVIEW",
        "training_ready": False,
        "parent_scope": "parents represented in this actual lineage only; not full dataset release coverage",
        "interval_semantics": "half-open parent frame intervals; ceil(seconds*30), core ownership only",
        "lineage_source": lineage_source,
        "parent_metadata_source": metadata_source,
        "parents": list(parents.values()),
    }
    review = {
        "schema_version": 1,
        "draft_sha256": None,
        "reviewer": "",
        "review_method": "",
        "reviewed_at": "",
        "parents": review_parents,
    }
    return draft, review


def validate_review(draft_file, reviewed_file, timestamp_file):
    draft, draft_source = load(draft_file)
    reviewed, reviewed_source = load(reviewed_file)
    timestamps, timestamp_source = load(timestamp_file)
    require(
        draft.get("schema_version") == reviewed.get("schema_version") == timestamps.get("schema_version") == 1,
        "Unsupported schema",
    )
    require(reviewed["draft_sha256"] == draft_source["sha256"], "Review is bound to another draft version")
    require(
        nonempty(reviewed.get("reviewer")) and nonempty(reviewed.get("reviewed_at")),
        "Explicit reviewer and review time required",
    )
    require(
        nonempty(reviewed.get("review_method")),
        "Record the actual semantic review method; human or agent/model-assisted review is allowed",
    )

    def by_parent(rows):
        result = {}
        for row in rows:
            key = (row["pool"], row["source_episode_index"])
            require(key not in result, f"Duplicate parent {key}")
            result[key] = row
        return result

    originals = by_parent(draft["parents"])
    reviews = by_parent(reviewed["parents"])
    times = by_parent(timestamps["parents"])
    require(set(originals) == set(reviews) == set(times), "Review / timestamp parent set must exactly match draft")
    output_parents = []
    for key, parent in originals.items():
        review, time = reviews[key], times[key]
        length = parent["length"]
        expected_issues = {i["id"] for i in parent["issues"]}
        resolved = review["issue_resolutions"]
        require(
            len(resolved) == len({x["issue_id"] for x in resolved})
            and {x["issue_id"] for x in resolved} == expected_issues,
            f"Issue resolution set differs {key}",
        )
        require(
            all(nonempty(x.get("resolution")) and nonempty(x.get("reviewer")) for x in resolved),
            f"Unresolved source issues {key}",
        )
        provenance = time["timestamp_source"]
        require(
            provenance["kind"] == "actual_parquet_timestamp_column", "Synthetic FPS-derived times cannot be exported"
        )
        require(provenance.get("column") == "timestamp", "Actual source timestamp column must be identified")
        parquet = Path(provenance["path"])
        if not parquet.is_absolute():
            parquet = Path(timestamp_file).resolve().parent / parquet
        require(digest(parquet) == provenance["sha256"], "Timestamp parquet source hash changed")
        values = time["timestamps"]
        require(time["fps"] == FPS and len(values) == length and length > 0, f"Timestamp count / FPS mismatch {key}")
        require(all(type(t) in (int, float) and math.isfinite(t) for t in values), "Invalid timestamp value")
        require(all(b > a for a, b in pairwise(values)), "Timestamps must strictly increase")
        require(
            all(abs(t - (values[0] + i / FPS)) <= 1e-4 for i, t in enumerate(values)),
            "Timestamp gap / non-30FPS sequence requires data audit",
        )
        labels = {x["id"] for x in parent["raw_labels"]}
        previous, native_rows = 0, []
        for interval in review["intervals"]:
            start = integer(interval["from_frame"], "from_frame")
            end = integer(interval["to_frame"], "to_frame")
            require(
                start == previous and 0 <= start < end <= length,
                f"Timeline gap / overlap / invalid bounds {key} at {previous}",
            )
            require(
                interval.get("review_status") == "APPROVED" and nonempty(interval.get("reviewer")),
                "Every interval needs a recorded semantic-review decision (APPROVED)",
            )
            require(
                caption_problem(interval.get("content")) is None,
                "Empty / unknown / mistake text cannot be an instruction",
            )
            refs = interval["source_label_ids"]
            require(
                isinstance(refs, list) and len(refs) == len(set(refs)) and set(refs) <= labels,
                "Unknown / duplicate raw label reference",
            )
            require(
                nonempty(interval.get("note")), "Record review decision or visual-review evidence for every interval"
            )
            # No carry-forward, filling, sorting, merging, or resolving concurrency here.
            native_rows.append(
                {
                    "role": "assistant",
                    "content": interval["content"].strip(),
                    "style": "subtask",
                    "timestamp": values[start],
                    "camera": None,
                    "tool_calls": None,
                }
            )
            previous = end
        require(previous == length, f"Timeline does not cover complete parent {key}")
        output_parents.append(
            {
                "pool": key[0],
                "source_episode_index": key[1],
                "length": length,
                "language_persistent": native_rows,
                "language_events": [],
                "timestamp_source": {**provenance, "path": str(parquet.resolve())},
            }
        )
    return {
        "schema_version": 1,
        "status": "LANGUAGE_ROWS_VALIDATED_DATASET_RELEASE_PENDING",
        "training_ready": False,
        "draft_source": draft_source,
        "reviewed_source": reviewed_source,
        "timestamp_input_source": timestamp_source,
        "reviewer": reviewed["reviewer"],
        "review_method": reviewed["review_method"],
        "reviewed_at": reviewed["reviewed_at"],
        "parents": output_parents,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--validate-reviewed", action="store_true")
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--parent-metadata", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--draft-file", type=Path)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--frame-timestamps", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "Output exists; choose a new artifact version")
        if args.draft:
            require(
                all((args.lineage, args.parent_metadata, args.results_dir)),
                "Draft requires --lineage --parent-metadata --results-dir",
            )
            review_path = args.output.with_suffix(".review.json")
            require(review_path != args.output and not review_path.exists(), "Review form exists; choose a new version")
            draft, review = make_draft(args.lineage, args.parent_metadata, args.results_dir)
            save_new(args.output, draft)
            review["draft_sha256"] = digest(args.output)
            save_new(review_path, review)
            result = {
                "status": draft["status"],
                "draft": str(args.output),
                "review_form": str(review_path),
                "parents": len(draft["parents"]),
                "issues": sum(len(p["issues"]) for p in draft["parents"]),
                "training_ready": False,
            }
        else:
            require(
                all((args.draft_file, args.reviewed, args.frame_timestamps)),
                "Validation requires --draft-file --reviewed --frame-timestamps from actual parquet",
            )
            data = validate_review(args.draft_file, args.reviewed, args.frame_timestamps)
            save_new(args.output, data)
            result = {
                "status": data["status"],
                "output": str(args.output),
                "parents": len(data["parents"]),
                "training_ready": False,
            }
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(
            json.dumps({"status": "BLOCKED", "error": str(error), "training_ready": False}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
