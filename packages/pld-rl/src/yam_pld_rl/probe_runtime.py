"""Thor probe diagnostics without actor weights or invented physical limits.

Records causal windows and original Pi targets. It cannot modify actions and
its traces explicitly lack the policy contract needed for SAC replay.
"""

import numpy as np

from yam_pld_rl.block_contract import SCHEMA
from yam_pld_rl.block_contract import array
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import integer
from yam_pld_rl.data import sha256
from yam_pld_rl.intervention import DEFAULT_GATE
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.kinematics import MODEL_SHA256
from yam_pld_rl.kinematics import YamKinematics


class ProbeRuntime:
    def __init__(self, model, *, base_weights_sha256, norm_sha256, engine_sha256, source_revision):
        self.fk = YamKinematics(model)
        self.contract = {"schema": SCHEMA, "kind": "probe_diagnostics_only", "gate": DEFAULT_GATE.copy(),
                         "action_dt_s": 1 / 30, "max_attempt_seconds": 15, "fk_model_sha256": MODEL_SHA256,
                         "base_weights_sha256": base_weights_sha256, "norm_sha256": norm_sha256,
                         "engine_sha256": engine_sha256, "physical_limits": None, "actor": None}
        self.snapshot = fingerprint({"contract": self.contract, "source_revision": source_revision,
                                     "runtime_sha256": sha256(__file__)})

    @property
    def metadata(self):
        return {"schema": SCHEMA, "expert": "PLD-Grasp", "phase": "probe_diagnostics_only",
                "snapshot_sha256": self.snapshot, "contract_sha256": fingerprint(self.contract),
                "modes": ["off", "probe"], "actor_loaded": False, "replay_contract_ready": False,
                "gate_owner": "thor", "curator_owner": "thor_offline_or_server",
                "recording": "native_HIL_with_full_pld_reply",
                "disabled_reason": "formal actor bundle and verified physical limits not installed"}

    def new_session(self):
        return ProbeSession(self)

    def validate_request(self, obs, rtc, request):
        if request.get("mode") == "off":
            return
        if request.get("mode") != "probe":
            raise ValueError("probe-only deployment: collect/eval/shadow unavailable; no actor or verified limits installed")
        if not isinstance(rtc, dict) or any(request.get(k) != self.metadata[k] for k in ("schema", "contract_sha256", "snapshot_sha256")):
            raise ValueError("probe requires trained RTC and exact handshake identities")
        for key in ("run_id", "episode_id"):
            if not isinstance(request.get(key), str) or not 0 < len(request[key]) <= 160:
                raise ValueError(f"probe {key} required")
        integer(request["epoch"], "epoch")
        integer(request["observation_id"], "observation_id")
        rid = request["request_id"]
        if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not 0 < len(str(rid)) <= 160:
            raise ValueError("native request_id required")
        integer(rtc["observation_policy_tick"], "observation tick")
        integer(rtc["delay_steps"], "RTC delay")
        array(request["previous_command"], (14,), "last submitted command")
        array(obs["observation.state"], (14,), "state")

    def infer(self, session, obs, rtc, request, output):
        self.validate_request(obs, rtc, request)
        if request["mode"] == "off":
            session.reset()
            return output
        array(output["actions"], (50, 14), "Pi targets")
        base = np.asarray(output["actions"])
        state = array(obs["observation.state"], (14,), "state")
        heights = self.fk.heights(state)
        gate = session.select(request, rtc, state, base, heights)
        trace = {**self.metadata, **{k: request[k] for k in ("run_id", "episode_id", "epoch", "request_id", "observation_id", "mode")},
                 "decision_id": f"e{request['epoch']}:r{request['request_id']}", "start_tick": rtc["observation_policy_tick"],
                 "delay_steps": rtc["delay_steps"], "gate": gate, "state": state,
                 "previous_command": np.asarray(request["previous_command"]), "base": base,
                 "actions": base, "base_z_m": heights, "prompt": obs.get("prompt"),
                 "residual_applied": False, "replay_eligible": False, "physical_residual": np.zeros(14)}
        if gate["active"]:
            trace.update({k: gate[k] for k in ("arm", "window_id", "elapsed_seconds")})
        # Preserve the original floating dtype and values of the entire chunk.
        return {**output, "pld": trace}


class ProbeSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.gate = InterventionSession(runtime.contract)

    def validate_request(self, obs, rtc, request):
        self.runtime.validate_request(obs, rtc, request)

    def reset(self):
        self.gate.reset()

    def __call__(self, obs, rtc, request, output):
        return self.runtime.infer(self.gate, obs, rtc, request, output)
