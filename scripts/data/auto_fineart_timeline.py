#!/usr/bin/env python3
"""Atomize native FLORA events and apply bounded, hash-bound candidate decisions.

Standard-library orchestration only. Source videos/results remain unchanged;
this module neither decodes images nor imports a model or training framework.
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.util
from itertools import pairwise
import json
import math
from pathlib import Path
import re
import sys

REVIEW_PATH = Path(__file__).resolve().parents[1] / "prepare_fineart_subtask_review.py"
if not REVIEW_PATH.exists():
    REVIEW_PATH = Path(__file__).resolve().with_name("prepare_fineart_subtask_review.py")
SPEC = importlib.util.spec_from_file_location("fineart_native_review", REVIEW_PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
PLAN_SCHEMA = "condapi_fineart_auto_timeline_v1"
DECISION_SCHEMA = "condapi_fineart_selector_decisions_v1"
DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "configs/annotation/tshirt-fineart-v1.json"
SOFT_ISSUES = {
    "OVERLAPPING_RAW_LABELS",
    "UNCOVERED_LABEL_FRAMES",
    "EMPTY_INSTRUCTION",
    "NON_INSTRUCTION_OR_UNCERTAIN_LABEL",
    "INCOMPLETE_SEGMENT",
}
CAMERAS = ("top", "left_wrist", "right_wrist")


def require(condition, message):
    review.require(condition, message)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def task_sha(task):
    return canonical_sha({key: value for key, value in task.items() if key != "task_sha256"})


def normalize_caption(value):
    return " ".join(value.split())


def parent_key(value):
    return value["pool"], value["source_episode_index"]


def resolve_file(path, directory):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (directory / value).resolve()


def verify_file(path, expected):
    path = Path(path)
    before = path.stat()
    require(review.digest(path) == expected, f"Source/evidence SHA-256 changed: {path}")
    after = path.stat()
    require(
        (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"Source changed while hashing: {path}",
    )


def load_policy(path):
    contract, source = review.load(path)
    require(
        contract.get("schema_version") == "condapi_fineart_annotation_contract_v1",
        "Unsupported annotation policy schema",
    )
    checks = [
        (contract["human_approval_required"] is False, "No per-segment human/user approval is required"),
        (
            contract["scope"]["fps"] == 30 and contract["scope"]["keep_parent_episodes"] is True,
            "Native 30 FPS complete parent policy required",
        ),
        (
            contract["candidates"]["neighbor_max_distance_frames"] == 150,
            "This implementation supports neighbor distance 150 only",
        ),
        (
            contract["candidates"]["neighbor_max_distinct_captions"] == 4,
            "This implementation supports four neighbor captions only",
        ),
        (
            contract["candidates"]["caption_equivalence"]
            == "whitespace-only normalization: ' '.join(text.split()); exact case-sensitive equality; no synonym rewrite",
            "Only whitespace-normalized exact caption equivalence is supported",
        ),
        (
            contract["evidence"]["context_frames_each_side"] == 30,
            "This implementation supports 30-frame evidence context only",
        ),
        (
            contract["window_seam"]["radius_frames"] == 150,
            "This implementation supports 150-frame seam review radius only",
        ),
        (contract["evidence"]["cameras"] == list(CAMERAS), "Three named YAM evidence cameras required"),
        (contract["candidates"]["include_UNKNOWN"] is True, "UNKNOWN must remain available"),
        (
            contract["candidates"]["allow_free_caption_in_selector"] is False
            and contract["candidates"]["allow_free_boundary_in_selector"] is False,
            "Selector must only choose offered IDs",
        ),
        (contract["routing"]["default_previous_label"] == "FORBIDDEN", "Forward-fill is forbidden"),
        (
            contract["routing"]["single_eligible_active_caption"] == "RULE_SINGLE_ACTIVE"
            and contract["routing"]["multiple_active_same_normalized_caption"] == "RULE_DUPLICATE_CAPTION",
            "Rule quality grades differ from supported policy",
        ),
        (
            isinstance(contract["routing"]["fixed_folding_order"], str)
            and contract["routing"]["fixed_folding_order"].startswith("FORBIDDEN"),
            "A fixed fold order is unsupported",
        ),
        (
            contract["merge"]["adjacent_only"] is True and contract["merge"]["cross_parent_episode"] is False,
            "Only adjacent same-caption merges within a parent are supported",
        ),
        (
            contract["native_adapter"]["full_parent_coverage_before_export"] is True
            and contract["native_adapter"]["review_method"] == "automatic_candidate_selector",
            "Automatic complete-parent native review contract required",
        ),
        (
            contract["models"]["flora"]["official_prompt_modified"] is False
            and contract["models"]["flora"]["raw_outputs_modified"] is False,
            "Official FLORA sources must remain unchanged",
        ),
        (
            contract["training_ready"] is False and contract["native_adapter"]["training_ready"] is False,
            "Annotation policy cannot certify training readiness",
        ),
        (
            contract["selector_io"]["plan_schema_version"] == PLAN_SCHEMA
            and contract["selector_io"]["decision_schema_version"] == DECISION_SCHEMA,
            "Policy selector schema differs from implementation",
        ),
    ]
    for valid, message in checks:
        require(valid, message)
    if "unknown_anchor_fallback" in contract:
        fallback = contract["unknown_anchor_fallback"]
        require(
            fallback["mode"] == "native-empty-anchor-fallback"
            and fallback["max_parent_unknown_fraction"] == 0.02
            and fallback["max_contiguous_unknown_frames"] == 30,
            "This implementation supports the explicit 2%/30-frame empty-anchor fallback policy only",
        )
        require(
            fallback["language_persistent"] == []
            and fallback["language_events"] == []
            and fallback["materialization_required"] is True
            and fallback["status"] == "NOT_MATERIALIZED_NOT_TRAINING_READY"
            and fallback["enabled_by_default"] is False
            and fallback["default_mode"] == "strict-parent",
            "Fallback requires explicit empty arrays, materialization and strict-parent default; null tombstones are forbidden",
        )
    policy = {
        "version": "complete_active_caption_atoms_v1",
        "neighbor_frames": contract["candidates"]["neighbor_max_distance_frames"],
        "max_neighbor_captions": contract["candidates"]["neighbor_max_distinct_captions"],
        "evidence_context_frames": contract["evidence"]["context_frames_each_side"],
        "seam_radius_frames": contract["window_seam"]["radius_frames"],
        "caption_equivalence": contract["candidates"]["caption_equivalence"],
    }
    return contract, source, policy


def source_context(draft, contract):
    """Rebuild native draft, verify actual files, and locate parent video clocks."""
    bindings = {}

    def bind(path, sha, kind):
        path = Path(path).resolve()
        verify_file(path, sha)
        key = str(path)
        if key in bindings:
            require(bindings[key]["sha256"] == sha, "Contradictory source hashes")
            bindings[key]["kinds"] = sorted(set(bindings[key]["kinds"] + [kind]))
        else:
            bindings[key] = {"path": key, "sha256": sha, "kinds": [kind]}
        return key

    for key in ("lineage_source", "parent_metadata_source"):
        source = draft[key]
        bind(source["path"], source["sha256"], key)
    lineage_dir = Path(draft["lineage_source"]["path"]).parent
    metadata_path = Path(draft["parent_metadata_source"]["path"])
    rows, _ = review.metadata_rows(metadata_path)
    metadata = {parent_key(row): row for row in rows}
    require(len(metadata) == len(rows), "Duplicate actual parent metadata")
    result_dirs = {
        str(Path(window["result_source"]["path"]).parent)
        for parent in draft["parents"]
        for window in parent["windows"]
        if window["result_source"]
    }
    require(len(result_dirs) == 1, "A native draft must bind one actual result directory")
    rebuilt, _ = review.make_draft(draft["lineage_source"]["path"], metadata_path, next(iter(result_dirs)))
    require(
        rebuilt == draft, "Draft differs from current native source reconstruction; create a new draft/plan version"
    )
    context = {}
    for parent in draft["parents"]:
        key = parent_key(parent)
        meta = metadata[key]
        scope = contract["scope"]
        require(
            meta.get("task") == scope["task"]
            and meta.get("source_repo") == scope["source_repo"]
            and meta.get("source_revision") == scope["source_revision"],
            "Actual parent task/source revision differs from annotation policy",
        )
        require(meta.get("status") == "PREPARED_REVIEW_PENDING", "Parent structural preparation is not complete")
        require(set(meta["videos"]) == set(CAMERAS), "Actual parent needs three camera records")
        videos = {}
        for camera in CAMERAS:
            clip = meta["videos"][camera]
            require(clip["metrics"]["frames"] == parent["length"], "Parent decoded-frame audit differs")
            source = clip["artifact"]
            path = bind(resolve_file(source["path"], metadata_path.parent), source["sha256"], "parent_camera")
            offset = clip["offset_s"]
            require(type(offset) in (int, float) and math.isfinite(offset), "Invalid parent video offset")
            origin = round(offset * review.FPS)
            require(
                origin >= 0 and abs(offset - origin / review.FPS) < 1e-5,
                "Parent video offset not aligned to audited native frame clock",
            )
            videos[camera] = {
                "path": path,
                "sha256": source["sha256"],
                "offset_s": offset,
                "parent_frame_zero_source_index": origin,
            }
        parquet = meta["data"]
        parquet_path = bind(resolve_file(parquet["path"], metadata_path.parent), parquet["sha256"], "parent_parquet")
        context[key] = {"videos": videos, "timestamp_parquet": {"path": parquet_path, "sha256": parquet["sha256"]}}
        for window in parent["windows"]:
            for field in ("result_source", "tempseg_source"):
                if window[field]:
                    bind(window[field]["path"], window[field]["sha256"], field)
            for camera in CAMERAS:
                source = window["cameras"][camera]
                bind(resolve_file(source["path"], lineage_dir), source["sha256"], "window_camera")
    return sorted(bindings.values(), key=lambda value: value["path"]), context


def eligible(label):
    raw = label["raw_segment"]
    return (
        label["eligible_suggestion"]
        and raw.get("status") == "complete"
        and review.caption_problem(raw.get("caption")) is None
    )


def caption_candidates(parent, start, end, policy):
    active = [
        label
        for label in parent["raw_labels"]
        if label["core_owned_interval"] is not None
        and label["core_owned_interval"][0] <= start
        and label["core_owned_interval"][1] >= end
    ]
    groups = {}

    def add(label, origin):
        caption = normalize_caption(label["raw_segment"]["caption"])
        group = groups.setdefault(caption, {"caption": caption, "source_label_ids": [], "origins": []})
        group["source_label_ids"].append(label["id"])
        group["origins"].append(origin)

    for label in active:
        if eligible(label):
            add(label, "active")
    active_captions = set(groups)
    neighbors = []
    for label in parent["raw_labels"]:
        if label in active or not eligible(label) or label["parent_interval"] is None:
            continue
        left, right = label["parent_interval"]
        distance = max(start - right, left - end, 0)
        if distance <= policy["neighbor_frames"]:
            neighbors.append((distance, label["id"], label))
    neighbor_captions = set()
    for _, _, label in sorted(neighbors):
        caption = normalize_caption(label["raw_segment"]["caption"])
        if caption not in active_captions and caption not in neighbor_captions:
            if len(neighbor_captions) >= policy["max_neighbor_captions"]:
                continue
            neighbor_captions.add(caption)
        add(label, "neighbor")
    candidates = []
    for index, caption in enumerate(sorted(groups)):
        group = groups[caption]
        origins = sorted(set(group.pop("origins")))
        group["source_label_ids"] = sorted(set(group["source_label_ids"]))
        candidates.append(dict(candidate_id=f"C{index:03d}", origin="_and_".join(origins), **group))
    active_ids = [candidate["candidate_id"] for candidate in candidates if candidate["caption"] in active_captions]
    neighbor_ids = [candidate["candidate_id"] for candidate in candidates if "neighbor" in candidate["origin"]]
    return candidates, active_ids, neighbor_ids, [label["id"] for label in active]


def evidence(parent, atom_id, start, end, videos, policy):
    context = policy["evidence_context_frames"]
    points = sorted(
        {max(0, start - context), start, (start + end - 1) // 2, end - 1, min(parent["length"] - 1, end - 1 + context)}
    )
    frames = [
        {
            "frame_id": f"{atom_id}:{camera}:f{frame:08d}",
            "camera": camera,
            "parent_frame_index": frame,
            "source_frame_index": videos[camera]["parent_frame_zero_source_index"] + frame,
            "video_sha256": videos[camera]["sha256"],
        }
        for frame in points
        for camera in CAMERAS
    ]
    return {"videos": videos, "parent_frame_indices": points, "frames": frames}


def seam_conflicts(parent, start, end, policy):
    boundaries = {window["core_from_frame"] for window in parent["windows"]} - {0}
    conflicts = []
    labels = [label for label in parent["raw_labels"] if eligible(label) and label["parent_interval"]]
    for boundary in sorted(boundaries):
        if end < boundary - policy["seam_radius_frames"] or start > boundary + policy["seam_radius_frames"]:
            continue
        covering = [
            label for label in labels if label["parent_interval"][0] <= start and label["parent_interval"][1] >= end
        ]
        # A differing description from the other window's context is evidence
        # of a seam question, even though it is not the atom's core owner.
        descriptions = {normalize_caption(label["raw_segment"]["caption"]) for label in covering}
        windows = {label["window_id"] for label in covering}
        if len(descriptions) > 1 and len(windows) > 1:
            conflicts.append(boundary)
            continue
        if boundary not in (start, end):
            continue
        left = [
            label for label in labels if label["core_owned_interval"] and label["core_owned_interval"][1] == boundary
        ]
        right = [
            label for label in labels if label["core_owned_interval"] and label["core_owned_interval"][0] == boundary
        ]
        if any(
            a["window_id"] != b["window_id"]
            and normalize_caption(a["raw_segment"]["caption"]) != normalize_caption(b["raw_segment"]["caption"])
            for a in left
            for b in right
        ):
            conflicts.append(boundary)
    return conflicts


def make_plan(draft_file, policy_file=DEFAULT_POLICY):
    contract, policy_source, policy = load_policy(policy_file)
    draft, draft_source = review.load(draft_file)
    require(draft.get("schema_version") == 1, "Unsupported native draft schema")
    bindings, context = source_context(draft, contract)
    parents, tasks = [], []
    for parent in draft["parents"]:
        points = {0, parent["length"]}
        for label in parent["raw_labels"]:
            for field in ("parent_interval", "core_owned_interval"):
                if label[field] is not None:
                    points.update(label[field])
        for window in parent["windows"]:
            points.update((window["core_from_frame"], window["core_to_frame"]))
        points = sorted(value for value in points if 0 <= value <= parent["length"])
        atoms = []
        for index, (start, end) in enumerate(pairwise(points)):
            atom_id = f"{parent['pool']}:{parent['source_episode_index']}:a{index:05d}"
            candidates, active, neighbors, raw_active = caption_candidates(parent, start, end, policy)
            seams = seam_conflicts(parent, start, end, policy)
            reasons = (["WINDOW_SEAM"] if seams else []) + (
                ["CONFLICTING_ACTIVE_CAPTIONS"]
                if len(active) > 1
                else ["NO_ELIGIBLE_ACTIVE_CAPTION"]
                if not active
                else []
            )
            atom = {
                "atom_id": atom_id,
                "from_frame": start,
                "to_frame": end,
                "active_source_label_ids": raw_active,
                "candidates": candidates,
                "auto_selection": None,
                "task_id": None,
                "seam_boundaries": seams,
                "reason_codes": reasons,
            }
            if len(active) == 1 and not seams:
                choice = next(candidate for candidate in candidates if candidate["candidate_id"] == active[0])
                # Automatic rules retain only active support, not nearby labels.
                refs = [
                    label["id"]
                    for label in parent["raw_labels"]
                    if label["id"] in raw_active
                    and eligible(label)
                    and normalize_caption(label["raw_segment"]["caption"]) == choice["caption"]
                ]
                atom["auto_selection"] = {**choice, "source_label_ids": sorted(refs)}
                atom["rule"] = "same_caption_overlap_dedup" if len(refs) > 1 else "unique_complete_active_caption"
            else:
                task_id = atom_id + ":select"
                media = evidence(parent, atom_id, start, end, context[parent_key(parent)]["videos"], policy)
                task = {
                    "task_id": task_id,
                    "atom_id": atom_id,
                    "pool": parent["pool"],
                    "source_episode_index": parent["source_episode_index"],
                    "from_frame": start,
                    "to_frame": end,
                    "active_candidate_ids": active,
                    "neighbor_candidate_ids": neighbors,
                    "candidates": candidates,
                    "reason_codes": reasons,
                    "seam_boundaries": seams,
                    "evidence": media,
                    "evidence_frame_ids": [frame["frame_id"] for frame in media["frames"]],
                }
                task["task_sha256"] = task_sha(task)
                tasks.append(task)
                atom["task_id"] = task_id
            atoms.append(atom)
        parents.append(
            {
                "pool": parent["pool"],
                "source_episode_index": parent["source_episode_index"],
                "length": parent["length"],
                "fps": review.FPS,
                "atoms": atoms,
                "hard_issues": [issue for issue in parent["issues"] if issue["code"] not in SOFT_ISSUES],
                "timestamp_parquet": context[parent_key(parent)]["timestamp_parquet"],
            }
        )
    status = (
        "BLOCKED_SOURCE_CONTRACT"
        if any(parent["hard_issues"] for parent in parents)
        else "NEEDS_CANDIDATE_SELECTION"
        if tasks
        else "RULE_TIMELINE_RESOLVED"
    )
    return {
        "schema_version": PLAN_SCHEMA,
        "status": status,
        "training_ready": False,
        "draft_source": draft_source,
        "implementation_source": {"path": str(Path(__file__).resolve()), "sha256": review.digest(__file__)},
        "review_tool_sha256": review.digest(REVIEW_PATH),
        "policy": policy,
        "policy_source": policy_source,
        "policy_contract": contract,
        "source_bindings": bindings,
        "parents": parents,
        "tasks": tasks,
        "seam_scope": "Core/frame coverage is mechanical; differing cross-window captions require bounded candidate selection, not an assumed subtask cut or synonym merge",
    }


def validate_selector(value, contract):
    require(
        isinstance(value, dict)
        and set(value) == {"model_id", "model_revision", "implementation_version", "prompt_sha256"},
        "Selector identity needs model/revision/implementation/prompt version",
    )
    require(all(review.nonempty(item) for item in value.values()), "Empty selector version identity")
    require(re.fullmatch(r"[0-9a-f]{64}", value["prompt_sha256"]) is not None, "Selector prompt SHA-256 missing")
    model = contract["models"]["selector"]
    require(
        value["model_id"] == model["model_id"] and value["model_revision"] == model["model_revision"],
        "Selector model/revision differs from bound policy",
    )


def read_decisions(plan_file, plan, decisions_file):
    if decisions_file is None:
        return {}, None
    document, source = review.load(decisions_file)
    require(document.get("schema_version") == DECISION_SCHEMA, "Unsupported decision schema")
    require(
        document["task_manifest_sha256"] == review.digest(plan_file), "Decision bound to different task manifest bytes"
    )
    selector = document["selector"]
    validate_selector(selector, plan["policy_contract"])
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    choices = {}
    allowed = {
        "task_id",
        "atom_id",
        "task_sha256",
        "selected_id",
        "evidence_frame_ids",
        "input_video_sha256",
        "inference_receipt",
    }
    for decision in document["decisions"]:
        require(set(decision) == allowed, "Decision fields must be bounded IDs/evidence, not free captions or times")
        task_id = decision["task_id"]
        require(task_id in tasks and task_id not in choices, "Unknown or duplicate decision task")
        task = tasks[task_id]
        require(
            decision["atom_id"] == task["atom_id"] and decision["task_sha256"] == task["task_sha256"] == task_sha(task),
            "Decision atom/task binding changed",
        )
        require(
            decision["selected_id"] in {candidate["candidate_id"] for candidate in task["candidates"]} | {"UNKNOWN"},
            "Selected ID was not offered by this atom",
        )
        require(
            decision["evidence_frame_ids"] == task["evidence_frame_ids"],
            "Decision evidence frame IDs differ from fixed task evidence",
        )
        videos = {camera: task["evidence"]["videos"][camera]["sha256"] for camera in CAMERAS}
        require(decision["input_video_sha256"] == videos, "Decision input video hashes differ")
        receipt_source = decision["inference_receipt"]
        receipt_path = resolve_file(receipt_source["path"], Path(decisions_file).resolve().parent)
        verify_file(receipt_path, receipt_source["sha256"])
        receipt, _ = review.load(receipt_path)
        for field in ("task_id", "atom_id", "task_sha256", "selected_id", "input_video_sha256"):
            require(receipt.get(field) == decision[field], f"Inference receipt {field} mismatch")
        require(receipt.get("selector") == selector, "Inference receipt selector version differs")
        images = receipt["evidence"]
        require(
            [image["frame_id"] for image in images] == task["evidence_frame_ids"],
            "Inference receipt image frame IDs differ",
        )
        for image in images:
            verify_file(resolve_file(image["path"], receipt_path.parent), image["sha256"])
        choices[task_id] = {
            "selected_id": decision["selected_id"],
            "inference_receipt": {"path": str(receipt_path), "sha256": receipt_source["sha256"]},
            "selector": selector,
            "evidence_frame_ids": decision["evidence_frame_ids"],
        }
    return choices, source


def apply_decisions(plan_file, decisions_file=None):
    plan, source = review.load(plan_file)
    require(plan.get("schema_version") == PLAN_SCHEMA, "Unsupported automatic timeline schema")
    require(
        plan == make_plan(plan["draft_source"]["path"], plan["policy_source"]["path"]),
        "Plan/source/policy version changed; regenerate tasks",
    )
    choices, decision_source = read_decisions(plan_file, plan, decisions_file)
    draft, _ = review.load(plan["draft_source"]["path"])
    originals = {parent_key(parent): parent for parent in draft["parents"]}
    reviewer_id = "automatic_candidate_selector:" + plan["implementation_source"]["sha256"][:12]
    reviewed, pending = [], []
    for parent in plan["parents"]:
        intervals, unresolved, resolved_atoms = [], [], []
        for atom in parent["atoms"]:
            task_id = atom["task_id"]
            selection = atom["auto_selection"]
            method = (
                "RULE_DUPLICATE_CAPTION" if atom.get("rule") == "same_caption_overlap_dedup" else "RULE_SINGLE_ACTIVE"
            )
            provenance = {
                "rule": atom.get("rule"),
                "atom_id": atom["atom_id"],
                "candidate_id": selection["candidate_id"] if selection else None,
            }
            if task_id is not None:
                decision = choices.get(task_id)
                if decision is None or decision["selected_id"] == "UNKNOWN":
                    unresolved.append(
                        {
                            "task_id": task_id,
                            "atom_id": atom["atom_id"],
                            "from_frame": atom["from_frame"],
                            "to_frame": atom["to_frame"],
                            "reason": "UNKNOWN" if decision else "MISSING_DECISION",
                            "inference_receipt": decision["inference_receipt"] if decision else None,
                        }
                    )
                    continue
                selection = next(
                    candidate
                    for candidate in atom["candidates"]
                    if candidate["candidate_id"] == decision["selected_id"]
                )
                method, provenance = (
                    "VLM_SELECTED_CANDIDATE",
                    dict(atom_id=atom["atom_id"], candidate_id=selection["candidate_id"], **decision),
                )
            require(selection is not None, "Resolved atom lacks a bounded selection")
            note = json.dumps(dict(quality=method, **provenance), ensure_ascii=False, sort_keys=True)
            resolved_atoms.append(
                {
                    "atom_id": atom["atom_id"],
                    "from_frame": atom["from_frame"],
                    "to_frame": atom["to_frame"],
                    "quality": method,
                    "candidate_id": selection["candidate_id"],
                    "source_label_ids": selection["source_label_ids"],
                    "inference_receipt": provenance.get("inference_receipt"),
                }
            )
            interval = {
                "from_frame": atom["from_frame"],
                "to_frame": atom["to_frame"],
                "content": selection["caption"],
                "source_label_ids": selection["source_label_ids"],
                "review_status": "APPROVED",
                "reviewer": reviewer_id,
                "note": note,
            }
            if (
                intervals
                and intervals[-1]["to_frame"] == interval["from_frame"]
                and intervals[-1]["content"] == interval["content"]
            ):
                previous = intervals[-1]
                previous["to_frame"] = interval["to_frame"]
                previous["source_label_ids"] = sorted(set(previous["source_label_ids"] + interval["source_label_ids"]))
                previous["note"] += "\n" + note
            else:
                intervals.append(interval)
        if unresolved or parent["hard_issues"]:
            pending.append(
                {
                    "pool": parent["pool"],
                    "source_episode_index": parent["source_episode_index"],
                    "length": parent["length"],
                    "local_selection_tasks": unresolved,
                    "hard_issues": parent["hard_issues"],
                    "resolved_intervals": intervals,
                    "action": "LOCAL_RETRY_OR_EXCEPTION_QUEUE; preserve whole parent, do not forward-fill",
                }
            )
            continue
        original = originals[parent_key(parent)]
        issue_resolutions = []
        raw = {label["id"]: label for label in original["raw_labels"]}
        for issue in original["issues"]:
            span = [issue.get("from_frame", 0), issue.get("to_frame", parent["length"])]
            if issue.get("source_label_id") in raw:
                label = raw[issue["source_label_id"]]
                span = label["core_owned_interval"] or label["parent_interval"] or span
            routes = [
                route for route in resolved_atoms if route["from_frame"] < span[1] and route["to_frame"] > span[0]
            ]
            issue_resolutions.append(
                {
                    "issue_id": issue["id"],
                    "reviewer": reviewer_id,
                    "resolution": json.dumps(
                        {"issue_code": issue["code"], "resolved_atoms": routes, "raw_source_issue_preserved": True},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        reviewed.append(
            {
                "pool": parent["pool"],
                "source_episode_index": parent["source_episode_index"],
                "intervals": intervals,
                "issue_resolutions": issue_resolutions,
            }
        )
    return {
        "schema_version": 1,
        "status": "RESOLVED_PARENTS_READY" if reviewed else "PENDING_LOCAL_SELECTION",
        "training_ready": False,
        "plan_source": source,
        "decision_source": decision_source,
        "draft_sha256": plan["draft_source"]["sha256"],
        "reviewer": reviewer_id,
        "review_method": "automatic_candidate_selector",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "parents": reviewed,
        "pending_parents": pending,
    }


def export_resolved(plan_file, decisions_file, timestamp_file, output, unknown_mode="strict-parent"):
    """Export only complete parents with an explicit hash-bound subset scope."""
    require(unknown_mode in {"strict-parent", "native-empty-anchor-fallback"}, "Unsupported UNKNOWN mode")
    if unknown_mode == "native-empty-anchor-fallback":
        return export_empty_anchor_view(plan_file, decisions_file, timestamp_file, output)
    application = apply_decisions(plan_file, decisions_file)
    plan, _ = review.load(plan_file)
    draft, _ = review.load(plan["draft_source"]["path"])
    output = Path(output)
    queue_path = output.with_suffix(".queue.json")
    scope = {
        "original_plan_source": application["plan_source"],
        "original_draft_source": plan["draft_source"],
        "included_parents": [list(parent_key(parent)) for parent in application["parents"]],
        "pending_parents": [list(parent_key(parent)) for parent in application["pending_parents"]],
    }
    require(not output.exists() and not queue_path.exists(), "Output exists; choose a new artifact version")
    if not application["parents"] or timestamp_file is None:
        report = {
            **application,
            "status": "PENDING_LOCAL_SELECTION"
            if not application["parents"]
            else "RESOLVED_TIMELINE_TIMESTAMP_EXPORT_PENDING",
            "scope": scope,
        }
        review.save_new(
            queue_path, {"schema_version": 1, "scope": scope, "pending_parents": application["pending_parents"]}
        )
        review.save_new(output, report)
        return report
    timestamps, timestamp_source = review.load(timestamp_file)
    times = {parent_key(parent): parent for parent in timestamps["parents"]}
    require(len(times) == len(timestamps["parents"]), "Duplicate timestamp parent")
    expected_parquets = {parent_key(parent): parent["timestamp_parquet"] for parent in plan["parents"]}
    keys = {parent_key(parent) for parent in application["parents"]}
    require(keys <= times.keys(), "Actual timestamp input missing resolved parents")
    for key in keys:
        provenance = times[key]["timestamp_source"]
        actual = resolve_file(provenance["path"], Path(timestamp_file).resolve().parent)
        expected = expected_parquets[key]
        require(
            str(actual) == expected["path"] and provenance["sha256"] == expected["sha256"],
            "Timestamp JSON not bound to original parent parquet",
        )
    draft_path = output.with_suffix(".scope.draft.json")
    time_path = output.with_suffix(".scope.timestamps.json")
    reviewed_path = output.with_suffix(".reviewed.json")
    for path in (draft_path, time_path, reviewed_path):
        require(not path.exists(), "Scope artifact exists; choose a new version")
    subset_draft = {
        **draft,
        "parents": [parent for parent in draft["parents"] if parent_key(parent) in keys],
        "automatic_scope": scope,
    }
    subset_times = {
        **timestamps,
        "parents": [parent for parent in timestamps["parents"] if parent_key(parent) in keys],
        "original_timestamp_input_source": timestamp_source,
        "automatic_scope": scope,
    }
    # Relative parquet paths must remain bound to the original timestamp input,
    # even when these subset sidecars are written to a different directory.
    subset_times = json.loads(json.dumps(subset_times))
    for parent in subset_times["parents"]:
        parent["timestamp_source"]["path"] = expected_parquets[parent_key(parent)]["path"]
    review.save_new(draft_path, subset_draft)
    review.save_new(time_path, subset_times)
    reviewed = {
        key: application[key] for key in ("schema_version", "reviewer", "review_method", "reviewed_at", "parents")
    }
    reviewed["draft_sha256"] = review.digest(draft_path)
    reviewed["automatic_scope"] = scope
    review.save_new(reviewed_path, reviewed)
    result = review.validate_review(draft_path, reviewed_path, time_path)
    result["automatic_scope"] = scope
    result["decision_source"] = application["decision_source"]
    result["pending_queue"] = {"path": str(queue_path.resolve()), "parents": len(application["pending_parents"])}
    review.save_new(
        queue_path, {"schema_version": 1, "scope": scope, "pending_parents": application["pending_parents"]}
    )
    review.save_new(output, result)
    return result


def unknown_runs(tasks):
    runs = []
    for task in sorted(tasks, key=lambda item: item["from_frame"]):
        if runs and runs[-1]["to_frame"] == task["from_frame"]:
            runs[-1]["to_frame"] = task["to_frame"]
            runs[-1]["task_ids"].append(task["task_id"])
            runs[-1]["inference_receipts"].append(task["inference_receipt"])
        else:
            runs.append(
                {
                    "from_frame": task["from_frame"],
                    "to_frame": task["to_frame"],
                    "task_ids": [task["task_id"]],
                    "inference_receipts": [task["inference_receipt"]],
                }
            )
    return runs


def checked_timestamps(value, expected, timestamp_file, length):
    provenance = value["timestamp_source"]
    path = resolve_file(provenance["path"], Path(timestamp_file).resolve().parent)
    require(
        provenance["kind"] == "actual_parquet_timestamp_column" and provenance["column"] == "timestamp",
        "Actual parquet timestamp provenance is required",
    )
    require(
        str(path) == expected["path"] and provenance["sha256"] == expected["sha256"],
        "Timestamp JSON not bound to original parent parquet",
    )
    verify_file(path, expected["sha256"])
    values = value["timestamps"]
    require(value["fps"] == review.FPS and len(values) == length and length > 0, "Timestamp count/FPS mismatch")
    require(all(type(item) in (int, float) and math.isfinite(item) for item in values), "Invalid actual timestamp")
    require(all(b > a for a, b in pairwise(values)), "Actual timestamps must strictly increase")
    require(
        all(abs(item - (values[0] + index / review.FPS)) <= 1e-4 for index, item in enumerate(values)),
        "Timestamp gap/non-30FPS sequence requires data audit",
    )
    return values, {**provenance, "path": str(path)}


def export_empty_anchor_view(plan_file, decisions_file, timestamp_file, output):
    """Describe full-frame materialization; never present this sidecar as a dataset."""
    require(timestamp_file is not None, "Empty-anchor fallback requires actual timestamps and original parquet hashes")
    application = apply_decisions(plan_file, decisions_file)
    plan, _ = review.load(plan_file)
    policy = plan["policy_contract"].get("unknown_anchor_fallback")
    require(policy is not None, "Bound policy does not enable an explicit empty-anchor fallback view")
    timestamps, timestamp_source = review.load(timestamp_file)
    require(timestamps["schema_version"] == 1, "Unsupported actual timestamp schema")
    times = {parent_key(parent): parent for parent in timestamps["parents"]}
    require(len(times) == len(timestamps["parents"]), "Duplicate timestamp parent")
    original = {parent_key(parent): parent for parent in plan["parents"]}
    candidates = [
        {**parent, "length": original[parent_key(parent)]["length"], "local_selection_tasks": [], "hard_issues": []}
        for parent in application["parents"]
    ]
    candidates.extend(application["pending_parents"])
    included, pending, accepted_unknown = [], [], []
    for parent in candidates:
        tasks = parent["local_selection_tasks"]
        runs = unknown_runs(tasks)
        count = sum(run["to_frame"] - run["from_frame"] for run in runs)
        max_run = max((run["to_frame"] - run["from_frame"] for run in runs), default=0)
        fraction = Fraction(str(policy["max_parent_unknown_fraction"]))
        permitted = (
            not parent["hard_issues"]
            and all(task["reason"] == "UNKNOWN" and task["inference_receipt"] for task in tasks)
            and count <= fraction * parent["length"]
            and max_run <= policy["max_contiguous_unknown_frames"]
        )
        if not permitted:
            pending.append(
                {
                    **parent,
                    "fallback_rejection": {
                        "unknown_frames": count,
                        "max_contiguous_unknown_frames": max_run,
                        "unknown_fraction": count / parent["length"],
                        "policy": policy,
                    },
                }
            )
            continue
        key = parent_key(parent)
        require(key in times, "Actual timestamp input missing fallback parent")
        length = original[key]["length"]
        values, provenance = checked_timestamps(times[key], original[key]["timestamp_parquet"], timestamp_file, length)
        intervals = parent.get("intervals", parent.get("resolved_intervals", []))
        coverage = sorted(
            [(row["from_frame"], row["to_frame"]) for row in intervals]
            + [(run["from_frame"], run["to_frame"]) for run in runs]
        )
        cursor = 0
        for start, end in coverage:
            require(
                type(start) is int and type(end) is int and start == cursor and 0 <= start < end <= length,
                "Language/empty-anchor views must cover every original parent frame exactly once",
            )
            cursor = end
        require(cursor == length, "Fallback view does not cover the complete parent")
        rows = []
        for interval in intervals:
            require(
                review.caption_problem(interval["content"]) is None and interval["review_status"] == "APPROVED",
                "Fallback cannot insert UNKNOWN, null or placeholder captions",
            )
            native = {
                "role": "assistant",
                "content": interval["content"],
                "style": "subtask",
                "timestamp": values[interval["from_frame"]],
                "camera": None,
                "tool_calls": None,
            }
            rows.append(
                {
                    "from_frame": interval["from_frame"],
                    "to_frame": interval["to_frame"],
                    "language_persistent": [native],
                    "language_events": [],
                    "source_label_ids": interval["source_label_ids"],
                    "review_evidence": interval["note"],
                }
            )
        empty = [{**run, "language_persistent": [], "language_events": []} for run in runs]
        included.append(
            {
                "pool": key[0],
                "source_episode_index": key[1],
                "length": length,
                "fps": review.FPS,
                "valid_language_rows": rows,
                "empty_language_anchor_intervals": empty,
                "timestamp_source": provenance,
                "action_frames": length,
                "valid_language_frames": length - count,
                "unknown_anchor_frames": count,
                "unknown_fraction": count / length,
                "materialization_required": True,
                "training_ready": False,
            }
        )
        if tasks:
            accepted_unknown.append(
                {
                    "pool": key[0],
                    "source_episode_index": key[1],
                    "tasks": tasks,
                    "action": "Optional local repair remains queued; language-empty fallback is within engineering thresholds",
                }
            )
    output = Path(output)
    queue_path = output.with_suffix(".queue.json")
    require(not output.exists() and not queue_path.exists(), "Output exists; choose a new artifact version")
    actions = sum(parent["action_frames"] for parent in included)
    language = sum(parent["valid_language_frames"] for parent in included)
    scope = {
        "included_parents": [list(parent_key(parent)) for parent in included],
        "pending_parents": [list(parent_key(parent)) for parent in pending],
    }
    result = {
        "schema_version": "condapi_fineart_native_empty_anchor_view_v1",
        "status": policy["status"],
        "training_ready": False,
        "materialized": False,
        "unknown_mode": policy["mode"],
        "fallback_policy": policy,
        "policy_source": plan["policy_source"],
        "plan_source": application["plan_source"],
        "decision_source": application["decision_source"],
        "timestamp_input_source": timestamp_source,
        "review_method": application["review_method"],
        "automatic_scope": scope,
        "parents": included,
        "coverage": {
            "action_frames": actions,
            "valid_language_frames": language,
            "empty_language_anchor_frames": actions - language,
            "action_hours": actions / review.FPS / 3600,
            "valid_language_hours": language / review.FPS / 3600,
            "empty_language_anchor_hours": (actions - language) / review.FPS / 3600,
        },
        "materialization_contract": {
            "preserve_all_parent_video_action_frames_and_H50": True,
            "write_both_language_columns_for_every_parent_frame": True,
            "empty_intervals_override_persistent_carry_forward": True,
            "null_or_empty_content_tombstones_allowed": False,
            "sidecar_is_not_a_materialized_training_dataset": True,
        },
        "pending_queue": {"path": str(queue_path.resolve()), "parents": len(pending)},
    }
    review.save_new(
        queue_path,
        {
            "schema_version": 1,
            "scope": scope,
            "pending_parents": pending,
            "accepted_unknown_local_repair_tasks": accepted_unknown,
        },
    )
    review.save_new(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--apply", action="store_true")
    parser.add_argument("--draft-file", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        help="Actual versioned contract (plan defaults to the repository policy; apply uses its bound policy).",
    )
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--frame-timestamps", type=Path)
    parser.add_argument(
        "--unknown-mode", choices=("strict-parent", "native-empty-anchor-fallback"), default="strict-parent"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.plan:
            require(args.draft_file is not None, "--plan requires --draft-file")
            result = make_plan(args.draft_file, args.policy or DEFAULT_POLICY)
            review.save_new(args.output, result)
        else:
            require(args.plan_file is not None, "--apply requires --plan-file")
            if args.policy is not None:
                bound, _ = review.load(args.plan_file)
                require(
                    str(args.policy.resolve()) == bound["policy_source"]["path"]
                    and review.digest(args.policy) == bound["policy_source"]["sha256"],
                    "--apply policy differs from the exact policy bound by the plan",
                )
            result = export_resolved(
                args.plan_file, args.decisions, args.frame_timestamps, args.output, args.unknown_mode
            )
        print(
            json.dumps(
                {"status": result["status"], "output": str(args.output), "training_ready": False}, ensure_ascii=False
            )
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(
            json.dumps({"status": "BLOCKED", "error": str(error), "training_ready": False}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
