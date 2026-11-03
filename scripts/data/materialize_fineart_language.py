#!/usr/bin/env python3
# PyArrow is intentionally loaded only for materialization, not CLI inspection.
# ruff: noqa: PLC0415
"""Materialize a hash-bound language view into new per-parent Parquet files.

This is not a LeRobot dataset publisher. Videos, source Parquet, action/state
values, episode identity, and training loops remain untouched.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

VIEW_SCHEMA = "condapi_fineart_native_empty_anchor_view_v1"
FORMAT_VERSION = 1
STATUS = "LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED"
LANGUAGE_COLUMNS = ("language_persistent", "language_events")
FPS = 30
SCHEMA_REVISION = "lerobot:f5c9bc02137c331b009eccd44ece6f11b62187a8:datasets/language.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def publish_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="." + path.name, suffix=".partial", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def official_language_types(pa):
    """Mirror the fixed upstream Arrow builders without importing datasets/Torch."""
    tools = pa.list_(pa.json_() if hasattr(pa, "json_") else pa.string())
    fields = [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=True),
        pa.field("style", pa.string(), nullable=True),
    ]
    tail = [pa.field("camera", pa.string(), nullable=True), pa.field("tool_calls", tools, nullable=True)]
    persistent = pa.list_(pa.struct([*fields, pa.field("timestamp", pa.float32(), nullable=False), *tail]))
    events = pa.list_(pa.struct(fields + tail))
    return persistent, events


def nested_array(values, data_type, pa):
    """Build nested JSON extensions explicitly; pa.array cannot infer these."""
    if isinstance(data_type, pa.BaseExtensionType):
        storage = nested_array(values, data_type.storage_type, pa)
        return pa.ExtensionArray.from_storage(data_type, storage)
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        flattened, offsets = [], [0]
        for value in values:
            if value is not None:
                flattened.extend(value)
            offsets.append(len(flattened))
        child = nested_array(flattened, data_type.value_type, pa)
        large = pa.types.is_large_list(data_type)
        offset_array = pa.array(offsets, type=pa.int64() if large else pa.int32())
        mask = pa.array([value is None for value in values], type=pa.bool_())
        factory = pa.LargeListArray if large else pa.ListArray
        return factory.from_arrays(offset_array, child, type=data_type, mask=mask)
    if pa.types.is_struct(data_type):
        children = [
            nested_array([value.get(field.name) if value is not None else None for value in values], field.type, pa)
            for field in data_type
        ]
        mask = pa.array([value is None for value in values], type=pa.bool_())
        return pa.StructArray.from_arrays(children, fields=list(data_type), mask=mask)
    return pa.array(values, type=data_type)


def column_field(table, name, fallback, pa):
    if name not in table.column_names:
        return (
            pa.field(name, fallback, nullable=False),
            "fixed_upstream_schema",
            {"schema_upgraded": False, "old_type": None, "new_type": str(fallback)},
        )
    field = table.schema.field(name)
    audit = {"schema_upgraded": not field.type.equals(fallback), "old_type": str(field.type), "new_type": str(fallback)}
    if not audit["schema_upgraded"]:
        return field, "source_arrow_schema", audit
    # ABC language columns may have inferred null-only fields or float64
    # message timestamps. Their values are being replaced, so use the complete
    # fixed native target schema without modifying any source/non-language data.
    upgraded = pa.field(name, fallback, nullable=field.nullable, metadata=field.metadata)
    return upgraded, "upgraded_to_fixed_upstream_schema", audit


def source_path(parent, view_file):
    provenance = parent["timestamp_source"]
    require(
        provenance["kind"] == "actual_parquet_timestamp_column" and provenance["column"] == "timestamp",
        "Actual source Parquet timestamp provenance is required",
    )
    path = Path(provenance["path"])
    if not path.is_absolute():
        path = Path(view_file).resolve().parent / path
    return path.resolve(), provenance["sha256"]


def timestamp_values(table, length):
    require("timestamp" in table.column_names, "Source Parquet has no actual timestamp column")
    values = table["timestamp"].to_pylist()
    require(
        len(values) == length and all(type(value) in (int, float) and math.isfinite(value) for value in values),
        "Invalid actual source timestamps",
    )
    require(all(right > left for left, right in pairwise(values)), "Source timestamps must strictly increase")
    require(
        all(abs(value - (values[0] + index / FPS)) <= 1e-4 for index, value in enumerate(values)),
        "Source timestamp gap/non-30FPS sequence requires data audit",
    )
    return values


def prepare_timeline(parent, times, policy):
    length = parent["length"]
    expected = [None] * length
    timeline = []
    coverage, unknown = [], 0
    for interval in parent["valid_language_rows"]:
        start, end = interval["from_frame"], interval["to_frame"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= length, "Invalid known frame interval")
        require(
            interval["language_events"] == []
            and isinstance(interval["language_persistent"], list)
            and len(interval["language_persistent"]) == 1,
            "Known intervals need one native persistent row and no events",
        )
        row = interval["language_persistent"][0]
        require(
            set(row) == {"role", "content", "style", "timestamp", "camera", "tool_calls"},
            "Native persistent message fields differ",
        )
        require(
            row["role"] == "assistant"
            and row["style"] == "subtask"
            and isinstance(row["content"], str)
            and row["content"].strip()
            and row["content"].strip().upper() not in {"UNKNOWN", "NONE", "NULL", "N/A"},
            "Null/empty/UNKNOWN placeholder messages are forbidden",
        )
        require(
            row["camera"] is None and row["tool_calls"] is None,
            "This subtask materializer supports camera/tool_calls=None only",
        )
        require(
            type(row["timestamp"]) in (int, float)
            and math.isfinite(row["timestamp"])
            and row["timestamp"] == times[start],
            "Language row timestamp must be read from the actual source start frame",
        )
        index = len(timeline)
        timeline.append(row)
        for frame in range(start, end):
            require(expected[frame] is None, "Language interval overlap")
            expected[frame] = index
        coverage.append((start, end, "known"))
    for interval in parent["empty_language_anchor_intervals"]:
        start, end = interval["from_frame"], interval["to_frame"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= length, "Invalid empty-anchor interval")
        require(
            interval["language_persistent"] == [] and interval["language_events"] == [],
            "Unknown anchors require both language columns [], never null/content tombstones",
        )
        for frame in range(start, end):
            require(expected[frame] is None, "Empty-anchor interval overlaps another interval")
            expected[frame] = -1
        coverage.append((start, end, "empty"))
        unknown += end - start
    cursor, run, max_run = 0, 0, 0
    for start, end, kind in sorted(coverage):
        require(start == cursor, "Language view must cover every original parent frame exactly once")
        cursor = end
        run = run + end - start if kind == "empty" else 0
        max_run = max(max_run, run)
    require(
        cursor == length and all(value is not None for value in expected), "Language view has uncovered parent frames"
    )
    require(
        unknown <= Fraction(str(policy["max_parent_unknown_fraction"])) * length
        and max_run <= policy["max_contiguous_unknown_frames"],
        "Unknown anchors exceed the bound view policy",
    )
    require(
        parent["action_frames"] == length
        and parent["unknown_anchor_frames"] == unknown
        and parent["valid_language_frames"] == length - unknown,
        "View frame counts disagree with actual intervals",
    )
    require(
        all(right["timestamp"] > left["timestamp"] for left, right in pairwise(timeline)),
        "Known persistent timeline must have strictly ordered native onset timestamps",
    )
    return timeline, expected


def verify_language(table, source, expected_array, expected_rows, times):
    require(table.num_rows == source.num_rows, "Materialized frame count changed")
    require(table.schema.metadata == source.schema.metadata, "Original Arrow schema metadata changed")
    for name in source.column_names:
        if name not in LANGUAGE_COLUMNS:
            require(
                table.schema.field(name) == source.schema.field(name) and table[name].equals(source[name]),
                f"Non-language column changed: {name}",
            )
    require(
        table["language_persistent"].combine_chunks().equals(expected_array), "Persistent language roundtrip differs"
    )
    persistent = table["language_persistent"].to_pylist()
    events = table["language_events"].to_pylist()
    require(all(value == [] for value in events), "Every materialized language_events anchor must be []")
    for frame, index in enumerate(expected_rows):
        rows = persistent[frame]
        require(isinstance(rows, list), "A materialized language column is null")
        if index == -1:
            require(rows == [] and events[frame] == [], "Unknown anchor was not materialized as two empty arrays")
            continue
        require(
            rows and all(isinstance(row["content"], str) and row["content"].strip() for row in rows),
            "Null/empty content tombstone found",
        )
        require(
            all(right["timestamp"] > left["timestamp"] for left, right in pairwise(rows)),
            "Stored persistent onset timestamps are not unique/ordered",
        )
        active = [position for position, row in enumerate(rows) if row["timestamp"] <= times[frame]]
        require(
            active and active[-1] == index,
            "Native persistent resolver would select the wrong unique active instruction",
        )


def materialize_parent(parent, view_file, root, view_source, pa, pq, policy):
    pool, eid, length = parent["pool"], parent["source_episode_index"], parent["length"]
    require(
        isinstance(pool, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}", pool) and pool not in {".", ".."},
        "Unsafe pool name",
    )
    require(
        type(eid) is int and eid >= 0 and type(length) is int and length > 0 and parent["fps"] == FPS,
        "Invalid parent identity/length/FPS",
    )
    require(
        parent.get("materialization_required") is True and parent.get("training_ready") is False,
        "Parent view must explicitly require materialization and retain training_ready=false",
    )
    source_file, source_sha = source_path(parent, view_file)
    require(digest(source_file) == source_sha, "Source Parquet SHA-256 changed")
    source = pq.ParquetFile(source_file).read(use_threads=False)
    require(
        len(source.column_names) == len(set(source.column_names)) and source.num_rows == length,
        "Source frame count/columns differ from view",
    )
    if "frame_index" in source.column_names:
        require(
            source["frame_index"].to_pylist() == list(range(length)),
            "Source frame indices are not the intact original parent",
        )
    if "episode_index" in source.column_names:
        require(
            all(value == eid for value in source["episode_index"].to_pylist()), "Source episode index differs from view"
        )
    times = timestamp_values(source, length)
    timeline, expected = prepare_timeline(parent, times, policy)
    persistent_type, event_type = official_language_types(pa)
    persistent_field, persistent_origin, persistent_upgrade = column_field(
        source, LANGUAGE_COLUMNS[0], persistent_type, pa
    )
    event_field, event_origin, event_upgrade = column_field(source, LANGUAGE_COLUMNS[1], event_type, pa)
    persistent = nested_array([[] if index == -1 else timeline for index in expected], persistent_field.type, pa)
    events = nested_array([[] for _ in expected], event_field.type, pa)
    output = source
    for field, array in ((persistent_field, persistent), (event_field, events)):
        if field.name in output.column_names:
            output = output.set_column(output.schema.get_field_index(field.name), field, array)
        else:
            output = output.append_column(field, array)
    directory = root / "parents" / pool / f"episode-{eid:06d}"
    directory.mkdir(parents=True)
    for path in (root / "parents", root / "parents" / pool, directory):
        path.chmod(0o755)
    partial, final = directory / "data.partial.parquet", directory / "data.parquet"
    pq.write_table(output, partial)
    partial.chmod(0o644)
    restored = pq.ParquetFile(partial).read(use_threads=False)
    verify_language(restored, source, persistent, expected, times)
    require(digest(source_file) == source_sha, "Source Parquet changed during materialization")
    os.link(partial, final)
    partial.unlink()
    unknown = sum(index == -1 for index in expected)
    receipt = {
        "pool": pool,
        "source_episode_index": eid,
        "length": length,
        "status": STATUS,
        "training_ready": False,
        "view_source": view_source,
        "source_parquet": {"path": str(source_file), "sha256": source_sha},
        "output_parquet": {"path": str(final.relative_to(root)), "sha256": digest(final)},
        "schema_upgraded": persistent_upgrade["schema_upgraded"] or event_upgrade["schema_upgraded"],
        "language_schema_audit": {"language_persistent": persistent_upgrade, "language_events": event_upgrade},
        "schema_source": {
            "language_persistent": persistent_origin,
            "language_events": event_origin,
            "missing_column_schema_revision": SCHEMA_REVISION,
            "persistent_timestamp_type": str(persistent_field.type.value_type.field("timestamp").type),
            "timestamp_origin": "actual source timestamp column, cast only to native/source language Arrow dtype",
        },
        "frame_audit": {
            "action_frames": length,
            "language_anchor_frames": length - unknown,
            "empty_language_anchor_frames": unknown,
            "non_language_columns_equal": True,
            "normal_anchors_keep_full_timeline": True,
            "unique_active_instruction_verified": True,
            "unknown_both_columns_empty": True,
            "events_all_empty": True,
        },
        "dataset_metadata_updated": False,
        "videos_actions_and_H50_unchanged": True,
    }
    publish_json(directory / "receipt.json", receipt)
    return receipt


def materialize(view_file, output_root):
    view, view_source = load(view_file)
    require(
        view.get("schema_version") == VIEW_SCHEMA
        and view.get("status") == "NOT_MATERIALIZED_NOT_TRAINING_READY"
        and view.get("unknown_mode") == "native-empty-anchor-fallback"
        and view.get("materialized") is False
        and view.get("training_ready") is False,
        "Input must be an unmaterialized native empty-anchor view",
    )
    policy = view["fallback_policy"]
    require(
        policy["max_parent_unknown_fraction"] == 0.02
        and policy["max_contiguous_unknown_frames"] == 30
        and policy["language_persistent"] == []
        and policy["language_events"] == []
        and policy["materialization_required"] is True,
        "Unsupported/null-tombstone fallback policy",
    )
    require(
        view["parents"]
        and view["materialization_contract"]["write_both_language_columns_for_every_parent_frame"] is True
        and view["materialization_contract"]["preserve_all_parent_video_action_frames_and_H50"] is True
        and view["materialization_contract"]["null_or_empty_content_tombstones_allowed"] is False,
        "View must require complete per-anchor materialization without tombstones",
    )
    keys = [(parent["pool"], parent["source_episode_index"]) for parent in view["parents"]]
    require(len(keys) == len(set(keys)), "Duplicate parent in language view")
    output_root = Path(output_root).expanduser().resolve()
    require(
        not output_root.exists(), "Output root already exists; this tool is new-directory only and never overwrites"
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_root.mkdir(parents=True)
    output_root.chmod(0o755)
    publish_json(
        output_root / "ownership.json",
        {
            "tool": "materialize_fineart_language",
            "format_version": FORMAT_VERSION,
            "view_source": view_source,
            "implementation_sha256": digest(__file__),
            "training_ready": False,
            "dataset_release": False,
        },
    )
    receipts = []
    try:
        receipts.extend(
            materialize_parent(parent, view_file, output_root, view_source, pa, pq, policy)
            for parent in view["parents"]
        )
        require(digest(view_source["path"]) == view_source["sha256"], "Input view changed during materialization")
        actions = sum(item["frame_audit"]["action_frames"] for item in receipts)
        language = sum(item["frame_audit"]["language_anchor_frames"] for item in receipts)
        require(
            view["coverage"]["action_frames"] == actions
            and view["coverage"]["valid_language_frames"] == language
            and view["coverage"]["empty_language_anchor_frames"] == actions - language,
            "View aggregate counts differ from materialized frames",
        )
        result = {
            "schema_version": "condapi_fineart_language_materialization_v1",
            "status": STATUS,
            "training_ready": False,
            "dataset_release": False,
            "view_source": view_source,
            "parents": receipts,
            "counts": {
                "parents": len(receipts),
                "action_frames": actions,
                "valid_language_frames": language,
                "empty_language_anchor_frames": actions - language,
                "action_hours": actions / FPS / 3600,
                "valid_language_hours": language / FPS / 3600,
            },
            "remaining_release_work": [
                "LeRobot v3 metadata/identity/publication",
                "native normalization",
                "FAST/language validation",
                "dataset release audit",
            ],
        }
        publish_json(output_root / "manifest.json", result)
        return result
    except Exception as error:
        publish_json(
            output_root / "failure.json",
            {
                "status": "MATERIALIZATION_INCOMPLETE",
                "training_ready": False,
                "view_source": view_source,
                "completed_parents": len(receipts),
                "error": str(error),
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = materialize(args.view, args.output_root)
        print(json.dumps({"status": result["status"], "counts": result["counts"], "training_ready": False}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "training_ready": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
