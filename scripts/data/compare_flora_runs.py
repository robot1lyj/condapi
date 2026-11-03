#!/usr/bin/env python3
"""Strict CPU comparison of FLORA baseline/session-cache artifacts and logs.

Requires FLORA_BATCH progress markers from run_flora_tshirt_batch.py to bind
native input-tensor audits to the correct job and stage. Timing is reported
separately and never used as a substitute for output/input parity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

NATIVE_SCHEMA = "flora_demo_annotation_v1"
CHECKPOINT_SCHEMA = "condapi_flora_stage_v1"
STAGES = ("tempseg", "captioner")


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def difference(left, right, path="$", limit=180):
    if type(left) is not type(right):
        return f"{path}: types {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys only_baseline={sorted(left.keys() - right.keys())}, only_candidate={sorted(right.keys() - left.keys())}"
        for key in sorted(left):
            value = difference(left[key], right[key], f"{path}.{key}", limit)
            if value:
                return value
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            value = difference(a, b, f"{path}[{index}]", limit)
            if value:
                return value
    elif left != right:
        return f"{path}: {repr(left)[:limit]} != {repr(right)[:limit]}"
    return None


def require_equal(left, right, label):
    error = difference(left, right)
    if error:
        raise ValueError(f"{label}: {error}")


def native_results(directory):
    result = {}
    for path in sorted(directory.glob("*.json")):
        value = load(path)
        if isinstance(value, dict) and value.get("schema_version") == NATIVE_SCHEMA:
            result[path.stem] = value
    if not result:
        raise ValueError(f"No native FLORA final outputs in {directory}")
    return result


def native_semantics(result):
    # Only this native output timing field is ignored. Frame times, durations,
    # source clocks, raw text, prompt hashes and the complete pipeline remain.
    return {key: value for key, value in result.items() if key != "elapsed_s"}


def validate_native(result, job):
    duration = result["duration_s"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{job}: invalid duration")
    temporal = result["pipeline"]["temporal_windows"]
    captions = result["pipeline"]["caption_inputs"]
    if not isinstance(temporal, list) or not temporal or not isinstance(captions, list):
        raise ValueError(f"{job}: full native temporal/caption audits are required")
    for index, window in enumerate(temporal):
        if not isinstance(window["output"], str) or not window["output"] or len(window["prompt_sha256"]) != 64:
            raise ValueError(f"{job}: missing raw temporal output/prompt hash at {index}")
    for row in result["segments"]:
        start, end = row["start_s"], row["end_s"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.001):
            raise ValueError(f"{job}: invalid native segment interval")
        if not isinstance(row["caption"], str) or not isinstance(row["status"], str):
            raise ValueError(f"{job}: missing native caption/status")
    return {"tempseg": len(temporal), "captioner": len(captions)}


def checkpoints(directory, job, final):
    source_path = directory / (job + ".tempseg.json")
    caption_path = directory / (job + ".captioner.json")
    source, caption = load(source_path), load(caption_path)
    for value, stage in ((source, "tempseg"), (caption, "captioner")):
        if value.get("schema_version") != CHECKPOINT_SCHEMA or value.get("stage") != stage or value.get("id") != job:
            raise ValueError(f"{job}: invalid {stage} checkpoint")
    require_equal(source["identity"], caption["identity"], f"{job}: checkpoint identities")
    if caption["tempseg_sha256"] != sha256(source_path):
        raise ValueError(f"{job}: caption checkpoint TempSeg SHA-256 mismatch")
    require_equal(caption["native_result"], final, f"{job}: published final vs caption checkpoint")
    for key in ("duration_s", "profile", "camera_roles", "input_sha256", "model_revisions"):
        require_equal(source["identity"][key], final[key], f"{job}: final {key} vs checkpoint identity")
    require_equal(source["temporal_audit"], final["pipeline"]["temporal_windows"], f"{job}: temporal audit chain")
    return source, caption


def structured_events(path):
    decoder = json.JSONDecoder()
    for number, line in enumerate(path.read_text(errors="strict").splitlines(), 1):
        for marker in ("FLORA_BATCH ", "FLORA_RUNTIME "):
            offset = line.find(marker)
            if offset < 0:
                continue
            try:
                value, _ = decoder.raw_decode(line[offset + len(marker) :])
            except ValueError as error:
                raise ValueError(f"{path}:{number}: malformed {marker.strip()} JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: structured event is not an object")
            yield marker.strip(), value, f"{path}:{number}"


def generation_audits(paths):
    records = defaultdict(lambda: defaultdict(list))
    total_generation_s = 0.0
    for path in paths:
        active, pending = None, None
        for source, value, location in structured_events(path):
            name = value.get("event")
            if source == "FLORA_BATCH":
                if name == "run_start":
                    active = None
                elif name == "progress":
                    if value.get("stage") not in STAGES or not isinstance(value.get("id"), str):
                        raise ValueError(f"{location}: progress lacks job/stage binding")
                    active = (value["id"], value["stage"])
                continue
            if name == "generation_start":
                if active is None or pending is not None:
                    raise ValueError(f"{location}: unbound or overlapping native generation")
                tensors = value.get("input_tensors")
                if not isinstance(tensors, dict) or not tensors:
                    raise ValueError(f"{location}: missing full native input tensor audit")
                calculated = hashlib.sha256(
                    json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if calculated != value.get("input_tensor_digest"):
                    raise ValueError(f"{location}: inconsistent input_tensor_digest")
                pending = (
                    active,
                    {
                        key: value[key]
                        for key in ("input_tokens", "output_limit", "input_tensor_digest", "input_tensors")
                    },
                )
            elif name == "generation_end":
                if pending is None or active != pending[0]:
                    raise ValueError(f"{location}: native generation_end lacks matching start")
                if value.get("complete") is not True:
                    raise ValueError(f"{location}: native generation is incomplete")
                pair, audit = pending
                for key in ("input_tokens", "output_limit"):
                    if audit[key] != value[key]:
                        raise ValueError(f"{location}: generation start/end {key} differs")
                audit.update(generated_tokens=value["generated_tokens"], complete=value["complete"])
                records[pair[0]][pair[1]].append(audit)
                total_generation_s += value["elapsed_s"]
                pending = None
        if pending is not None:
            raise ValueError(f"{path}: unfinished generation; use complete original log files")
    return records, total_generation_s


def compare(args):
    baseline = native_results(args.baseline_dir)
    candidate = native_results(args.candidate_dir)
    jobs = sorted(set(args.ids) if args.ids else baseline.keys() | candidate.keys())
    for name, results in (("baseline", baseline), ("candidate", candidate)):
        missing = set(jobs) - results.keys()
        if missing:
            raise ValueError(f"{name}: missing complete native final outputs: {sorted(missing)}")
    baseline_audits, baseline_generation_s = generation_audits(args.baseline_log)
    candidate_audits, candidate_generation_s = generation_audits(args.candidate_log)
    if not args.ids:
        for name, audits in (("baseline", baseline_audits), ("candidate", candidate_audits)):
            unknown = audits.keys() - set(jobs)
            if unknown:
                raise ValueError(f"{name}: logs contain generations without final outputs: {sorted(unknown)}")
    details = []
    for job in jobs:
        expected = validate_native(baseline[job], job)
        require_equal(expected, validate_native(candidate[job], job), f"{job}: native call counts")
        require_equal(
            native_semantics(baseline[job]),
            native_semantics(candidate[job]),
            f"{job}: native output, raw temporal text, segment/caption/audit",
        )
        left = checkpoints(args.baseline_dir, job, baseline[job])
        right = checkpoints(args.candidate_dir, job, candidate[job])
        require_equal(
            {key: left[0][key] for key in ("identity", "segments", "temporal_audit")},
            {key: right[0][key] for key in ("identity", "segments", "temporal_audit")},
            f"{job}: raw TempSeg checkpoint",
        )
        timings = {}
        for stage, index in (("tempseg", 0), ("captioner", 1)):
            for name, audits in (("baseline", baseline_audits), ("candidate", candidate_audits)):
                if len(audits.get(job, {}).get(stage, [])) != expected[stage]:
                    raise ValueError(
                        f"{name}/{job}/{stage}: log has {len(audits.get(job, {}).get(stage, []))} generation calls, native audit expects {expected[stage]}; include all original logs, not only a resume log"
                    )
            require_equal(
                baseline_audits.get(job, {}).get(stage, []),
                candidate_audits.get(job, {}).get(stage, []),
                f"{job}/{stage}: input tensor digests and generation completion/tokens",
            )
            timings[stage] = {"baseline": left[index]["timing"], "candidate": right[index]["timing"]}
        details.append(
            {
                "id": job,
                "status": "EXACT_PARITY",
                "duration_s": baseline[job]["duration_s"],
                "segments": len(baseline[job]["segments"]),
                "generation_calls": expected,
                "timings": timings,
            }
        )
    return {
        "status": "EXACT_PARITY",
        "jobs": len(jobs),
        "checked": "native final except top-level elapsed_s; identity; raw TempSeg segments/metadata/output; all pipeline/prompt audits; per-call full input tensor digests; generated token counts and complete flags",
        "ignored": "top-level native elapsed_s and checkpoint/runtime timing, model-load counts, mode, UTC, memory measurements",
        "all_supplied_log_generation_wall_s": {"baseline": baseline_generation_s, "candidate": candidate_generation_s},
        "details": details,
        "limitation": "Native logs do not contain generated token IDs; this checks exact decoded temporal output and final caption strings, not unlogged token IDs. This is parity, not semantic correctness or seam acceptance.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, action="append", required=True)
    parser.add_argument("--candidate-log", type=Path, action="append", required=True)
    parser.add_argument("--ids", nargs="+", help="Explicit subset; otherwise all outputs/logged jobs must be complete.")
    parser.add_argument("--output", type=Path, help="Optional new JSON report; existing files are never overwritten.")
    args = parser.parse_args(argv)
    try:
        result = compare(args)
        code = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "PARITY_FAILED", "error": str(error), "error_type": type(error).__name__}
        code = 1
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
