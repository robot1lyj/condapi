"""Connection-scoped causal intervention gate running on Thor, not the client.

window_id describes an online control window. Offline physical segmentation
creates the definitive attempt_id independently. Height is a handoff guard,
not a reward or proof that an object is held.
"""

from uuid import uuid4

import numpy as np

from yam_pld_rl.block_contract import array
from yam_pld_rl.block_contract import integer

LEGACY_GATE = {"open": 0.65, "closed": 0.62, "drop": 0.18,
               "lookahead_steps": 30, "lift_m": 0.05, "lift_observations": 2,
               "max_observation_gap_s": 0.5}
V2_GATE = {"version": 2, "close_level": 0.8, "forecast_drop": 0.12,
                "close_drop": 0.10, "strong_close_drop": 0.18, "reopen_delta": 0.12,
                "lookahead_steps": 30, "lift_m": 0.05, "lift_observations": 2,
                "lift_rise_m": 0.003, "max_observation_gap_s": 0.5}
DEFAULT_GATE = {k: v for k, v in V2_GATE.items() if k not in {"lift_rise_m", "lift_observations"}}
DEFAULT_GATE.update(version=3, forecast_timeout_s=2.0, closed_timeout_s=3.0,
                    settle_observation_ticks=2)


class InterventionSession:
    def __init__(self, contract):
        self.contract = contract
        self.reset()

    def reset(self):
        self.identity, self.last_tick, self.active = None, None, None
        self.serial = 0
        self.session_id = uuid4().hex
        self.blocked = set()
        self.arms = {}

    def select(self, request, rtc, state, base, heights):
        if self.contract["gate"].get("version", 1) == 1:
            return self._legacy_select(request, rtc, state, base, heights)
        identity = tuple(request[k] for k in ("run_id", "episode_id", "epoch", "mode"))
        epoch = integer(request["epoch"], "epoch")
        tick = integer(rtc["observation_policy_tick"], "observation tick")
        delay = integer(rtc["delay_steps"], "delay")
        if self.identity != identity:
            self.reset()
            self.identity = identity
        if self.last_tick is not None and tick <= self.last_tick:
            raise ValueError("nonmonotonic observation tick; reset epoch on reconnect/restart")
        cfg, dt = self.contract["gate"], self.contract["action_dt_s"]
        modern = cfg.get("version") == 3
        state = array(state, (14,), "gate state")
        base = array(base, (50, 14), "gate Pi plan")
        if not all(np.isfinite(heights[arm]) for arm in ("left", "right")):
            raise ValueError("finite measured FK heights required")
        ended = None
        if self.last_tick is not None and (tick - self.last_tick) * dt > cfg["max_observation_gap_s"]:
            if self.active:
                ended = {"window_id": self.active["window_id"], "arm": self.active["arm"], "reason": "observation_gap"}
            self.active = None
            # Never carry a motion trend across missing observations.
            self.arms = {}
        self.last_tick = tick
        for arm, index in (("left", 6), ("right", 13)):
            g = float(state[index])
            tracker = self.arms.setdefault(arm, {"armed": True, "minimum": g})
            tracker["minimum"] = min(tracker["minimum"], g)
            if not tracker["armed"] and g - tracker["minimum"] >= cfg["reopen_delta"]:
                tracker["armed"] = True
        if modern and self.active and self.active.get("phase") == "draining":
            a = self.active
            if tick >= a["drain_until_tick"] + cfg["settle_observation_ticks"]:
                ended = {k: a[k] for k in ("window_id", "arm", "close_tick", "close_z", "stop_tick", "drain_until_tick")}
                ended.update(reason=a["stop_reason"], completed_tick=tick, gate_version=3)
                g = float(state[6 if a["arm"] == "left" else 13])
                self.arms[a["arm"]] = {"armed": g-a["minimum"] >= cfg["reopen_delta"], "minimum": a["minimum"]}
                self.active = None
        elif self.active:
            a = self.active
            g = float(state[6 if a["arm"] == "left" else 13])
            a["peak"] = max(a["peak"], g)
            a["minimum"] = min(a["minimum"], g)
            a["minimum_z"] = min(a["minimum_z"], heights[a["arm"]])
            contraction = a["peak"] - g
            if a["close_tick"] is None and (
                contraction >= cfg["strong_close_drop"]
                or (g <= cfg["close_level"] and contraction >= cfg["close_drop"])
            ):
                a["close_tick"] = tick
                a["minimum"] = g
            # Follow the lowest measured approach point: an early close signal
            # can precede the end of descent. No future video/FK is consulted.
            if a["close_tick"] is not None:
                a["close_z"] = a["minimum_z"]
            reason = None
            if self.contract.get("trajectory") and not self._in_workspace(heights["measured_xyz"][a["arm"]], a["arm"]):
                reason = "left_grasp_workspace"
            elif (tick - a["start_tick"]) * dt >= self.contract["max_attempt_seconds"]:
                reason = "timeout"
            elif modern and a["close_tick"] is None and (tick-a["start_tick"])*dt >= cfg["forecast_timeout_s"]:
                reason = "forecast_timeout"
            elif a["close_tick"] is not None:
                if g - a["minimum"] >= cfg["reopen_delta"]:
                    reason = "reopened"
                else:
                    lifted = heights[a["arm"]] - a["close_z"] >= cfg["lift_m"]
                    confirmed = lifted if modern else heights[a["arm"]] - a["last_z"] >= cfg["lift_rise_m"]
                    a["lift_count"] = a["lift_count"] + 1 if confirmed else 0
                    # At the first valid lift, stop *new* corrections. The
                    # draining state and post-tail visual outcome provide the
                    # subsequent confirmation without extending actuation.
                    if lifted and (modern or a["lift_count"] >= cfg["lift_observations"]):
                        reason = "lift_guard"
                    elif modern and (tick-a["close_tick"])*dt >= cfg["closed_timeout_s"]:
                        reason = "closed_timeout"
            a["last_z"] = heights[a["arm"]]
            if reason:
                if modern:
                    # Keep the window for queue/observation transitions, but
                    # produce no new residual during this mandatory tail.
                    pending = a.get("pending_until_tick", tick)
                    if self.contract.get("trajectory"):
                        # This reply adds no correction after its immutable
                        # prefix. Uncommitted old tails are superseded. Replay
                        # must prove this reply was accepted before rewarding.
                        pending = min(pending, tick+delay-1)
                    a.update(phase="draining", stop_tick=tick, stop_reason=reason,
                             drain_until_tick=max(tick, pending))
                else:
                    ended = {"window_id": a["window_id"], "reason": reason, "arm": a["arm"]}
                    self.arms[a["arm"]] = {"armed": False, "minimum": a["minimum"]}
                    self.active = None
        forecasts = []
        if self.active is None and (ended is None or (modern and ended.get("gate_version") == 3)):
            for arm, index in (("left", 6), ("right", 13)):
                if not self.arms[arm]["armed"]:
                    continue
                if self.contract.get("trajectory") and (
                    not self._in_workspace(heights["measured_xyz"][arm], arm)
                    or not self._in_workspace(heights["planned_xyz"][arm][delay], arm)
                ):
                    continue
                # Committed RTC prefix is past policy ownership. Only the
                # current Pi proposal's future editable suffix can trigger.
                end = min(delay + cfg["lookahead_steps"], len(base))
                hits = np.flatnonzero(state[index] - base[delay:end, index] >= cfg["forecast_drop"])
                scope = self.contract.get("grasp_z_range_m")
                if scope:
                    lo, hi = scope[arm]
                    planned = array(heights["planned_z_m"][arm], (50,), "planned FK height")
                    hits = np.asarray([i for i in hits if lo <= planned[delay + i] <= hi], dtype=int)
                if len(hits):
                    forecasts.append((delay + int(hits[0]), arm))
            if forecasts:
                close_in, arm = min(forecasts)
                self.serial += 1
                g = float(state[6 if arm == "left" else 13])
                self.active = {"window_id": f"e{epoch}:s{self.session_id}:w{self.serial}", "arm": arm,
                               "start_tick": tick, "peak": g, "minimum": g, "minimum_z": heights[arm],
                               "last_z": heights[arm], "phase": "correcting",
                               "close_z": None, "close_tick": None, "lift_count": 0, "forecast_close_in": close_in}
                self.arms[arm]["armed"] = False
        result = {"active": self.active is not None, "ended": ended,
                  "forecasts": [{"arm": arm, "close_in_steps": n} for n, arm in forecasts], "gate_version": cfg["version"]}
        if self.active:
            result.update({k: self.active[k] for k in ("window_id", "arm", "start_tick", "close_tick", "close_z")})
            result["elapsed_seconds"] = (tick - self.active["start_tick"]) * dt
            if modern:
                result.update(phase=self.active["phase"], apply_residual=self.active["phase"] == "correcting")
                for key in ("stop_tick", "stop_reason", "drain_until_tick"):
                    if key in self.active:
                        result[key] = self.active[key]
        return result

    def _in_workspace(self, xyz, arm):
        from yam_pld_rl.trajectory import in_workspace  # noqa: PLC0415
        return bool(in_workspace(xyz, arm, self.contract))

    def note_proposal(self, tick, delay, *, end_index=None):
        """Conservative last possible residual tick, even for zero/probe output.

        Previously sent proposals may remain in the RTC prefix. Only the
        recorder can later prove which commands were actually consumed.
        """
        if self.contract["gate"].get("version") == 3 and self.active and self.active["phase"] == "correcting":
            end = tick + (delay + self.contract["block_steps"] if end_index is None else end_index) - 1
            self.active["pending_until_tick"] = max(end, self.active.get("pending_until_tick", end))

    def _legacy_select(self, request, rtc, state, base, heights):
        identity = tuple(request[k] for k in ("run_id", "episode_id", "epoch", "mode"))
        epoch = integer(request["epoch"], "epoch")
        tick = integer(rtc["observation_policy_tick"], "observation tick")
        if self.identity != identity:
            self.reset()
            self.identity = identity
        if self.last_tick is not None and tick <= self.last_tick:
            raise ValueError("nonmonotonic observation tick; reset epoch on reconnect/restart")
        cfg, dt = self.contract["gate"], self.contract["action_dt_s"]
        ended = None
        if self.last_tick is not None and (tick - self.last_tick) * dt > cfg["max_observation_gap_s"]:
            if self.active:
                ended = {"window_id": self.active["window_id"], "reason": "observation_gap"}
            self.active = None
        self.last_tick = tick
        state, base = array(state, (14,), "gate state"), array(base, (50, 14), "gate Pi plan")
        for arm, index in (("left", 6), ("right", 13)):
            if state[index] >= cfg["open"]:
                self.blocked.discard(arm)
        if self.active:
            a = self.active
            index = 6 if a["arm"] == "left" else 13
            g = state[index]
            a["peak"] = max(a["peak"], float(g))
            if a["close_z"] is None and g <= cfg["closed"] and a["peak"] - g >= cfg["drop"]:
                a["close_z"], a["close_tick"] = heights[a["arm"]], tick
            a["minimum"] = min(a["minimum"], float(g))
            reason = None
            if (tick - a["start_tick"]) * dt >= self.contract["max_attempt_seconds"]:
                reason = "timeout"
            elif a["close_z"] is not None:
                if g >= cfg["open"] and g - a["minimum"] >= cfg["drop"]:
                    reason = "reopened"
                lifted = heights[a["arm"]] - a["close_z"] >= cfg["lift_m"] and g < cfg["open"]
                a["lift_count"] = a["lift_count"] + 1 if lifted else 0
                if a["lift_count"] >= cfg["lift_observations"]:
                    reason = "lift_guard"
            if reason:
                ended = {"window_id": a["window_id"], "reason": reason, "arm": a["arm"]}
                self.blocked.add(a["arm"])
                self.active = None
        forecasts = []
        if self.active is None and ended is None:
            # The forecast is the current Pi proposal, never later rollout frames.
            for arm, index in (("left", 6), ("right", 13)):
                if arm in self.blocked or state[index] < cfg["open"]:
                    continue
                hits = np.flatnonzero((base[:cfg["lookahead_steps"], index] <= cfg["closed"])
                                      & (state[index] - base[:cfg["lookahead_steps"], index] >= cfg["drop"]))
                if len(hits):
                    forecasts.append((int(hits[0]), arm))
            if forecasts:
                close_in, arm = min(forecasts)  # deterministic left first on a tie
                self.serial += 1
                self.active = {"window_id": f"e{epoch}:s{self.session_id}:w{self.serial}", "arm": arm, "start_tick": tick,
                               "peak": float(state[6 if arm == "left" else 13]), "minimum": 1.0,
                               "close_z": None, "close_tick": None, "lift_count": 0, "forecast_close_in": close_in}
        result = {"active": self.active is not None, "ended": ended,
                  "forecasts": [{"arm": arm, "close_in_steps": n} for n, arm in forecasts]}
        if self.active:
            result.update({k: self.active[k] for k in ("window_id", "arm", "start_tick", "close_tick", "close_z")})
            result["elapsed_seconds"] = (tick - self.active["start_tick"]) * dt
        return result
