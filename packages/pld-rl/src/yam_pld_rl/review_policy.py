"""Shared model-suggestion and human-override reward publication policy."""

from yam_pld_rl.grasp_outcomes import reviewed_outcome

POLICY = "qwen_dino_suggestion_v1"


def endpoint_rows(candidate):
    proof = candidate["outcome"]
    chosen, seen = [], set()
    for frame in reversed(candidate["frames"]):
        if frame["row"] not in proof["lifted_rows"] or candidate["arm"] not in frame["views"]:
            continue
        ref = frame["views"][candidate["arm"]]
        identity = (ref["source_video"], ref["video_frame_index"])
        if identity in seen:
            continue
        seen.add(identity)
        chosen.append(frame["row"])
        if len(chosen) == 2:
            break
    return sorted(chosen)


def automatic_reward(candidate, judged):
    """Use every supported model suggestion, including weak/disagreeing evidence.

    User selected this inclusive policy. Preserve uncertainty as provenance;
    absence of a model suggestion or invalid causal evidence still needs review.
    """
    proof = candidate.get("outcome")
    if not proof or not proof["kinematic_complete"] or not proof["fresh_rows"]:
        return None, "missing_complete_endpoint"
    if not judged or judged.get("suggested_label") not in ("success", "failure"):
        return None, "no_model_suggestion"
    outcome = judged["suggested_label"]
    rows = endpoint_rows(candidate)
    if not rows or rows[-1] != proof["fresh_rows"][-1]:
        return None, "no_supported_endpoint_suggestion"
    review = {"candidate_id": candidate["candidate_id"], "candidate_sha256": candidate["candidate_sha256"],
              "status": "held" if outcome == "success" else "empty", "approved": True,
              "reviewer": POLICY, "evidence_rows": rows, "label_source": "automatic_model_suggestion",
              "human_reviewed": False, "grasp_intent_confirmed": False, "held_then_lost_confirmed": False}
    review.update(model_confidence=judged.get("confidence"), model_audit_flags=judged.get("audit_flags", []),
                  qwen_suggestion=outcome, dino_suggestion=judged.get("dino", {}).get("outcome"))
    try:
        reviewed_outcome(candidate, review)
    except (ValueError, KeyError, IndexError) as exc:
        return None, "unsupported_endpoint: " + str(exc)
    return review, "automatic_model_suggestion"


def human_reward(candidate, state):
    """Bind an explicit UI choice to the original immutable candidate proof."""
    status = state.get("status")
    if status not in ("held", "empty", "not_lifted", "no_closure", "dropped"):
        raise ValueError("select a supported outcome or exclude")
    proof = candidate["outcome"]
    rows = endpoint_rows(candidate) if status in ("held", "empty") else proof["fresh_rows"][-1:]
    review = {"candidate_id": candidate["candidate_id"], "candidate_sha256": candidate["candidate_sha256"],
              "status": status, "approved": True, "reviewer": state.get("reviewer") or "user_review",
              "evidence_rows": rows, "grasp_intent_confirmed": state.get("grasp_intent_confirmed") is True,
              "held_then_lost_confirmed": state.get("held_then_lost_confirmed") is True,
              "note": state.get("note", ""), "label_source": "human_review", "human_reviewed": True}
    reviewed_outcome(candidate, review)
    return review


def merge_rewards(candidates, judged, states):
    rewards, pending, excluded, automatic = [], [], [], []
    for candidate in candidates:
        if not candidate.get("outcome"):
            continue
        cid = candidate["candidate_id"]
        proposed, reason = automatic_reward(candidate, judged.get(cid))
        if proposed:
            automatic.append(cid)
        state = states.get(cid, {})
        if state.get("status") == "exclude":
            excluded.append({"candidate_id": cid, "reason": state.get("note") or "explicit user exclusion"})
            continue
        if state.get("status"):
            try:
                proposed = human_reward(candidate, state)
            except (ValueError, KeyError, IndexError) as exc:
                pending.append({"candidate_id": cid, "reason": "invalid_human_evidence: " + str(exc)})
                continue
        if proposed:
            rewards.append(proposed)
        else:
            pending.append({"candidate_id": cid, "reason": reason})
    return {"rewards": rewards, "pending": pending, "excluded": excluded, "automatic_ids": automatic,
            "policy": POLICY, "accuracy": None}
