"""Replayable local selectors/verifiers and logical reset requests.

No motor commands, CAN, FK, camera capture or remote task messaging lives here.
Inputs are timestamped evidence produced by the existing client/perception.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GraspContract:
    arm: str
    calibration_id: str
    # Scene-specific entry band and openness thresholds need review.
    entry_low_m: float
    entry_high_m: float
    open_threshold: float
    closed_threshold: float
    attempt_budget_s: float
    entry_confirmations: int = 3
    lift_m: float = 0.05
    hold_s: float = 1.0
    max_evidence_age_s: float = 0.25
    max_evidence_gap_s: float = 0.25

    def __post_init__(self):
        if self.arm not in ("left", "right") or not self.calibration_id:
            raise ValueError("arm and calibrated height frame required")
        numbers = (
            self.entry_low_m, self.entry_high_m, self.open_threshold, self.closed_threshold,
            self.attempt_budget_s, self.lift_m, self.hold_s,
            self.max_evidence_age_s, self.max_evidence_gap_s,
        )
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("nonfinite grasp contract")
        if not 0 <= self.entry_low_m < self.entry_high_m:
            raise ValueError("invalid calibrated tabletop entry band")
        if not 0 <= self.closed_threshold < self.open_threshold <= 1:
            raise ValueError("invalid 0-closed/1-open thresholds")
        if min(numbers[4:]) <= 0 or self.attempt_budget_s <= self.hold_s or self.entry_confirmations < 1:
            raise ValueError("invalid confirmation/time budget")


@dataclass(frozen=True)
class Evidence:
    epoch: int
    tick: int
    time_s: float
    observation_id: str
    visual_time_s: float
    visual_id: str
    calibration_id: str
    height_m: float
    openness: float
    pose_valid: bool
    # None means unknown/occluded, not false. All predicates are reviewed.
    centered_brick: bool | None
    outside_bins: bool | None
    holding_brick: bool | None
    empty_hand: bool | None

    def __post_init__(self):
        if self.epoch < 0 or self.tick < 0 or not self.observation_id or not self.visual_id:
            raise ValueError("versioned evidence identity required")
        if not all(math.isfinite(x) for x in (self.time_s, self.visual_time_s, self.height_m, self.openness)):
            raise ValueError("nonfinite feedback")
        if not 0 <= self.openness <= 1 or self.visual_time_s > self.time_s:
            raise ValueError("openness/clock contract mismatch")
        if any(x is not None and type(x) is not bool for x in (
            self.centered_brick, self.outside_bins, self.holding_brick, self.empty_hand,
        )):
            raise ValueError("visual predicates must be boolean or unknown")
        if self.empty_hand is True and self.holding_brick is True:
            raise ValueError("contradictory holding evidence")


class GraspSupervisor:
    """One arm, recurring local attempts; output events rather than commands."""

    def __init__(self, contract, *, mode):
        if mode not in ("practice", "evaluate"):
            raise ValueError("explicit local-practice/full-task-evaluation mode required")
        self.contract, self.mode = contract, mode
        self.phase = "base"
        self.count = 0
        self.attempt = None
        self.serial = 0
        self.last = None
        self.last_visual_id = None
        self.last_visual_time = None
        self.hold_start = None
        self.closing_started = False

    def _valid(self, evidence):
        return (
            evidence.pose_valid and evidence.calibration_id == self.contract.calibration_id
            and 0 <= evidence.time_s - evidence.visual_time_s <= self.contract.max_evidence_age_s
        )

    def _entry(self, evidence):
        c = self.contract
        return (
            self._valid(evidence) and c.entry_low_m <= evidence.height_m <= c.entry_high_m
            and evidence.openness >= c.open_threshold and evidence.empty_hand is True
            and evidence.centered_brick is True and evidence.outside_bins is True
        )

    def _finish(self, success, reason, evidence):
        event = {
            "type": "attempt_end", "attempt_id": self.attempt["attempt_id"],
            "arm": self.contract.arm, "success": success, "reward": None if success is None else int(success),
            "reason": reason, "epoch": evidence.epoch, "tick": evidence.tick, "time_s": evidence.time_s,
            "terminal_observation_id": evidence.observation_id,
        }
        self.attempt = None
        self.count = 0
        self.hold_start = None
        if success is None:
            self.phase = "review"
            event["next"] = "review_outcome"
        elif success and self.mode == "evaluate":
            self.phase = "carrying"
            event["next"] = "handoff_to_pi"
        elif self.mode == "practice":
            self.phase = "reset_pending"
            event["next"] = "request_local_reset"
        else:
            self.phase = "base" if self._entry(evidence) else "blocked"
            event["next"] = "retry" if self.phase == "base" else "wait_for_reentry"
        return event

    def update(self, evidence):
        c = self.contract
        if (self.last and evidence.epoch == self.last.epoch
                and (evidence.tick <= self.last.tick or evidence.time_s <= self.last.time_s)):
            raise ValueError("duplicate/out-of-order control evidence")
        events = []
        epoch_changed = self.last is not None and evidence.epoch != self.last.epoch
        if epoch_changed:
            if self.attempt:
                events.append(self._finish(None, "epoch_changed", evidence))
            else:
                # A new epoch cannot silently clear a pending reset/review or carrying state.
                self.count = 0
            self.last_visual_id = None
            self.last_visual_time = None
            self.hold_start = None
        self.last = evidence
        fresh = evidence.visual_id != self.last_visual_id
        if fresh and self.last_visual_time is not None and evidence.visual_time_s <= self.last_visual_time:
            raise ValueError("visual evidence IDs advanced with nonadvancing capture time")
        visual_gap = (
            self.last_visual_time is not None
            and evidence.visual_time_s - self.last_visual_time > c.max_evidence_gap_s
        )
        if fresh:
            self.last_visual_id, self.last_visual_time = evidence.visual_id, evidence.visual_time_s
        if epoch_changed:
            return events
        if not self._valid(evidence) or visual_gap:
            self.count = 0
            self.hold_start = None
        if self.phase in ("review", "reset_pending"):
            return events
        if self.phase in ("carrying", "blocked"):
            # Grasp must not retrigger while transporting a brick or inside a bin.
            released = (
                self._valid(evidence) and evidence.empty_hand is True and evidence.holding_brick is False
                and evidence.openness >= c.open_threshold and evidence.outside_bins is True
                and evidence.height_m > c.entry_high_m
            )
            self.count = self.count + 1 if fresh and released else (0 if not released else self.count)
            if self.count >= c.entry_confirmations:
                self.phase, self.count = "base", 0
                events.append({"type": "rearmed", "arm": c.arm, "tick": evidence.tick})
            return events
        if self.phase == "base":
            eligible = self._entry(evidence)
            self.count = self.count + 1 if fresh and eligible else (0 if not eligible else self.count)
            if self.count >= c.entry_confirmations:
                self.serial += 1
                self.attempt = {
                    "attempt_id": f"{evidence.epoch}:{c.arm}:{self.serial}",
                    "entry_time_s": evidence.time_s, "entry_height_m": evidence.height_m,
                    "entry_tick": evidence.tick, "entry_observation_id": evidence.observation_id,
                }
                self.phase, self.count = "active", 0
                self.hold_start, self.closing_started = None, False
                events.append({"type": "attempt_start", "arm": c.arm, **self.attempt})
            return events
        self.closing_started |= evidence.openness <= c.closed_threshold
        if fresh:
            held = (
                self._valid(evidence) and evidence.holding_brick is True and evidence.empty_hand is False
                and evidence.height_m >= self.attempt["entry_height_m"] + c.lift_m
            )
            if held:
                if self.hold_start is None:
                    self.hold_start = evidence.visual_time_s
                if evidence.visual_time_s - self.hold_start >= c.hold_s:
                    events.append(self._finish(success=True, reason="lift_and_visual_hold", evidence=evidence))
                    return events
            else:
                self.hold_start = None
            if (self._valid(evidence) and self.closing_started and evidence.empty_hand is True
                    and evidence.openness >= c.open_threshold):
                events.append(self._finish(success=False, reason="reopened_empty", evidence=evidence))
                return events
        if evidence.time_s - self.attempt["entry_time_s"] >= c.attempt_budget_s:
            known = self._valid(evidence) and evidence.holding_brick is not None and evidence.empty_hand is not None
            events.append(self._finish(False if known else None, "attempt_budget" if known else "uncertain_timeout", evidence))
        return events

    def reset_acknowledged(self, evidence):
        """External reset executor supplies fresh evidence; never auto-execute."""
        if self.phase != "reset_pending" or not self._valid(evidence):
            raise ValueError("no valid pending reset acknowledgement")
        if self.last and (evidence.epoch != self.last.epoch or evidence.tick <= self.last.tick
                          or evidence.time_s <= self.last.time_s):
            raise ValueError("reset acknowledgement must be newer in the same epoch")
        if (evidence.visual_id == self.last_visual_id
                or (self.last_visual_time is not None and evidence.visual_time_s <= self.last_visual_time)):
            raise ValueError("reset acknowledgement requires a new visual observation")
        if evidence.empty_hand is not True or evidence.holding_brick is not False:
            raise ValueError("reset did not establish an empty hand")
        if evidence.openness < self.contract.open_threshold:
            raise ValueError("reset did not reopen the gripper")
        self.phase, self.count, self.last = "base", 0, evidence
        self.last_visual_id, self.last_visual_time = evidence.visual_id, evidence.visual_time_s


def reset_intent(outcome, *, entry_height_m, budget_s):
    """Paper local-practice reset, expressed as a request to the existing client.

    No trajectory, joint target or hardware command is generated here. An
    approved client implementation must keep planar position, lower, release,
    verify and report failure/time budget. Evaluation must never use this reset.
    """
    if outcome.get("next") != "request_local_reset" or outcome.get("reward") not in (0, 1):
        raise ValueError("reset is only valid after a known local-practice outcome")
    if not math.isfinite(entry_height_m) or not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("invalid reset reference/budget")
    return {
        "type": "parts_local_reset_request_v1", "attempt_id": outcome["attempt_id"], "arm": outcome["arm"],
        "entry_height_m": entry_height_m, "budget_s": budget_s,
        "steps": ["lower_at_current_planar_position_if_holding", "release_and_open", "verify_empty_hand"],
        "on_failure": "request_human_reset", "executes_motion": False,
    }
