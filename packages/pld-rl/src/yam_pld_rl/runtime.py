"""Optional Thor postprocessor. Loading it does not enable physical collection."""

import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

from yam_pld_rl.block_contract import SCHEMA
from yam_pld_rl.block_contract import array
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import feature_dim
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import integer
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import physical_baseline
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.block_contract import wire_plan
from yam_pld_rl.collector import center_latent
from yam_pld_rl.collector import execution_contract
from yam_pld_rl.collector import validate_collector
from yam_pld_rl.data import sha256
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.sac import SquashedActor
from yam_pld_rl.trajectory import CorrelatedNoise
from yam_pld_rl.trajectory import planning_context
from yam_pld_rl.trajectory import trajectory_decode


class BlockRuntime:
    def __init__(self, bundle, encoder_weights, *, base_weights_sha256, device="cuda"):
        bundle = Path(bundle)
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        self.contract = self.manifest["contract"]
        validate_contract(self.contract)
        validate_collector(self.manifest)
        self.actor_contract = self.manifest.get("actor_contract", self.contract)
        if self.manifest["observation_dim"] != feature_dim(self.contract):
            raise ValueError("bundle feature dimension differs from gate/action contract")
        if self.contract["base_weights_sha256"] != base_weights_sha256:
            raise ValueError("PLD/Pi checkpoint mismatch")
        if sha256(bundle / "actor.pt") != self.manifest["actor_sha256"]:
            raise ValueError("PLD actor hash mismatch")
        if sha256(encoder_weights) != self.manifest["encoder_sha256"]:
            raise ValueError("PLD encoder hash mismatch")
        self.device = device
        self.snapshot = sha256(bundle / "manifest.json")
        self.actor = SquashedActor(self.manifest["observation_dim"], self.manifest["config"]).to(device).eval()
        self.actor.load_state_dict(torch.load(bundle / "actor.pt", map_location=device, weights_only=True))
        self.encoder = resnet18(weights=None)
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location="cpu", weights_only=True))
        self.encoder.fc = nn.Identity()
        self.encoder.requires_grad_(requires_grad=False).to(device).eval()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.fk = YamKinematics(bundle / "fk.xml")
        torch.set_num_threads(2)
        with torch.inference_mode():
            for _ in range(2):
                self.encoder(torch.zeros((3, 3, 224, 224), device=device))
                self.actor(torch.zeros((1, self.manifest["observation_dim"]), device=device), deterministic=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    @property
    def metadata(self):
        return {"schema": SCHEMA, "expert": "PLD-Grasp", "snapshot_sha256": self.snapshot,
                "contract_sha256": fingerprint(self.contract), "modes": ["off", "shadow", "probe", "collect", "eval"],
                "block_steps": self.contract["block_steps"], "segmentation_owner": "thor_offline_or_server",
                "actor_loaded": True, "replay_contract_ready": True, "actor_phase": self.manifest["phase"],
                "gate_version": self.contract["gate"].get("version", 1),
                "reward_contract": self.contract.get("reward_contract", "lift_only_legacy"),
                "gate_owner": "thor", "recording": "native_HIL_with_full_pld_reply",
                "collector": self.manifest.get("collector"),
                "collect_joint_radius_rad": self.contract["radius"][0],
                "eval_joint_radius_rad": self.actor_contract["radius"][0]}

    def new_session(self):
        """Share GPU networks, isolate causal gate state per websocket connection."""
        return RuntimeSession(self)

    @torch.inference_mode()
    def infer(self, session, obs, rtc, request, output):
        start = time.monotonic()
        c = execution_contract(self.manifest, request.get("mode"))
        smooth = ((self.manifest.get("collector") or {}).get("kind") == "coherent_trajectory_v1"
                  and (request.get("mode") in {"collect", "eval"}
                       or (c.get("trajectory") and request.get("mode") in {"probe", "shadow"})))
        diagnostic_zero = bool(smooth and request.get("mode") in {"probe", "shadow"})
        if request.get("mode") == "off":
            session.reset()
            return output
        if request.get("mode") not in {"shadow", "probe", "collect", "eval"}:
            raise ValueError("unknown PLD mode")
        if not rtc or any(request.get(k) != self.metadata[k] for k in ("schema", "contract_sha256", "snapshot_sha256")):
            raise ValueError("PLD requires aligned trained RTC and exact bundle identity")
        if obs.get("prompt") != c["task"]:
            raise ValueError("task differs from specialist bundle")
        for key in ("run_id", "episode_id"):
            if not isinstance(request.get(key), str) or not 0 < len(request[key]) <= 160:
                raise ValueError(f"PLD {key} required")
        tick = integer(rtc["observation_policy_tick"], "tick")
        delay = integer(rtc["delay_steps"], "delay")
        epoch = integer(request["epoch"], "epoch")
        observation_id = integer(request["observation_id"], "observation_id")
        rid = request["request_id"]
        if isinstance(rid, bool) or not isinstance(rid, (str, int)) or not 0 < len(str(rid)) <= 160:
            raise ValueError("native request_id required")
        array(output["actions"], (50, 14), "Pi actions")
        raw_base = np.asarray(output["actions"])
        base = physical_baseline(raw_base, delay, c)
        state = array(obs["observation.state"], (14,), "state")
        previous = array(request["previous_command"], (14,), "last submitted command")
        gate_heights = self.fk.gate_heights(state, base, c)
        heights = {arm: gate_heights[arm] for arm in ("left", "right")}
        gate = session.select(request, rtc, state, base, gate_heights)
        trace = {**{k: request[k] for k in ("run_id", "episode_id", "mode", "request_id")},
                 **self.metadata, "epoch": epoch, "observation_id": observation_id,
                 "decision_id": f"e{epoch}:r{rid}", "gate": gate,
                 "start_tick": tick, "delay_steps": delay, "base": wire_plan(base, raw_base, delay), "state": state,
                 "physical_base": base, "physical_actions": base,
                 "previous_command": previous, "base_z_m": heights, "raw_base": raw_base,
                 "baseline_transform": c.get("baseline_transform"), "actions": wire_plan(base, raw_base, delay),
                 "wire_actions": wire_plan(base, raw_base, delay), "execution_contract_sha256": fingerprint(c),
                 "replay_eligible": not diagnostic_zero, "residual_applied": False}
        if not gate["active"]:
            return {**output, "actions": trace["wire_actions"], "pld": trace}
        arm, elapsed = gate["arm"], gate["elapsed_seconds"]
        context = planning_context(base, previous, state, gate, arm, delay, c) if smooth else None
        scale, negative_scale = feasible_bounds(base, previous, arm, delay, c, context=context)
        if gate.get("apply_residual") is False:
            scale, negative_scale = np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32)
        images = []
        for view in ("top_rgb", "left_rgb", "right_rgb"):
            rgb = np.asarray(obs[f"observation.images.{view}"])
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = rgb.transpose(1, 2, 0)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("PLD requires uint8 RGB")
            images.append(self.transform(Image.fromarray(rgb)))
        features = self.encoder(torch.stack(images).to(self.device))
        features = torch.nn.functional.normalize(features, dim=-1).flatten().cpu().numpy()
        extra = {"negative_scale": negative_scale, "context": window_context(gate, tick, heights[arm], c["action_dt_s"])} if c["gate"].get("version") == 3 else {}
        x = observation_vector(features, state, base, previous, arm, delay, elapsed, scale, self.manifest["normalization"], **extra)
        mode = request["mode"]
        if self.manifest.get("collector") and (mode == "collect" or smooth):
            ac = self.actor_contract
            ap, an = feasible_bounds(base, previous, arm, delay, ac,
                                     context=context if ac.get("trajectory") else None)
            if gate.get("apply_residual") is False:
                ap, an = np.zeros(7, np.float32), np.zeros(7, np.float32)
            ax = observation_vector(features, state, base, previous, arm, delay, elapsed, ap,
                                    self.manifest["normalization"], negative_scale=an,
                                    context=window_context(gate, tick, heights[arm], ac["action_dt_s"]))
            mean_u, _ = self.actor(torch.from_numpy(ax)[None].to(self.device), deterministic=True)
            mean_u = mean_u[0].cpu().numpy()
            offset = 0 if arm == "left" else 7
            physical_center = map_action(mean_u, ap, an) * np.asarray(ac["radius"][offset:offset+7], np.float32)
            if smooth:
                radius = np.asarray(c["radius"][offset:offset+7])
                physical_center = np.clip(physical_center, -radius*negative_scale, radius*scale)
            mu = torch.from_numpy(center_latent(physical_center, c["radius"][offset:offset+7], scale, negative_scale)).to(self.device)[None]
            std = torch.tensor(self.manifest["collector"]["latent_std"], device=self.device)[None]
            if smooth:
                noise = {"rho": 0., "previous": np.zeros(7), "innovation": np.zeros(7), "value": np.zeros(7)}
                if mode == "collect" and gate.get("apply_residual"):
                    innovation = torch.randn(7, device=self.device).cpu().numpy()
                    noise = session.noise.advance(gate["window_id"], tick, c["action_dt_s"], c["trajectory"]["noise_tau_s"], innovation)
                z = mu + std * torch.as_tensor(noise["value"], device=self.device, dtype=mu.dtype)[None]
                conditional_mu = mu + std * noise["rho"] * torch.as_tensor(noise["previous"], device=self.device, dtype=mu.dtype)[None]
                conditional_std = std * math.sqrt(1-noise["rho"]**2)
                trace.update(collector_noise=noise, trajectory_context=context)
            else:
                z = mu + std * torch.randn_like(mu)
                conditional_mu, conditional_std = mu, std
            u = torch.tanh(z)
            logp = (-.5 * (((z-conditional_mu)/conditional_std).square()+2*conditional_std.log()+math.log(2*math.pi))
                    -2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))).sum(-1)
            if smooth and (mode != "collect" or not gate.get("apply_residual")):
                logp = torch.zeros_like(logp)
            # Visual/state/plan/previous-command fields are shared with x.
            trace.update(actor_context=ax[2264:], actor_mean_u=mean_u, actor_scale=ap, actor_negative_scale=an,
                         actor_physical_center=physical_center, collector_latent_center=mu[0].cpu().numpy(),
                         collector_latent_std=std[0].cpu().numpy())
        else:
            u, logp = self.actor(torch.from_numpy(x)[None].to(self.device), deterministic=mode == "eval")
        u = u[0].cpu().numpy()
        action = map_action(u, scale, negative_scale) if mode in {"collect", "eval"} else np.zeros(7, dtype=np.float32)
        if smooth:
            if diagnostic_zero:
                # Zero target through a prefix bridge can still move. Probe
                # and shadow must bypass the bridge and execute exact Pi.
                actions, projection = base.copy(), {"factor": 0., "reason": "diagnostic_zero_residual"}
            else:
                actions, projection = trajectory_decode(base, action, arm, c, context)
            trace["trajectory_projection"] = projection
            if mode == "collect" and (context["reason"] or projection["factor"] == 0):
                session.noise = CorrelatedNoise()
                trace["collector_noise_reset_after"] = context["reason"] or projection["reason"]
        else:
            actions = decode(base, action, arm, delay, c)
        wire = wire_plan(actions, raw_base, delay)
        trace.update(arm=arm, window_id=gate["window_id"], elapsed_seconds=elapsed,
                     envelope_end_tick=tick + (delay if diagnostic_zero else context["end_index"] if smooth else delay + c["block_steps"]), x=x, scale=scale, negative_scale=negative_scale, sample_u=u,
                     action=action, log_prob_u=float(logp[0]), actions=wire, physical_actions=actions,
                     wire_actions=wire,
                     residual_applied=bool(np.any(actions != base)),
                     encoder_ms_included_total_ms=(time.monotonic() - start) * 1000)
        session.note_proposal(tick, delay, end_index=delay if diagnostic_zero else context["end_index"] if smooth else None)
        return {**output, "actions": wire, "pld": trace}


class RuntimeSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.gate = InterventionSession(runtime.contract)
        self.mode = None
        self.noise = CorrelatedNoise()

    def reset(self):
        self.gate.reset()
        self.noise = CorrelatedNoise()

    def select(self, *args):
        return self.gate.select(*args)

    def note_proposal(self, *args, **kwargs):
        return self.gate.note_proposal(*args, **kwargs)

    def __call__(self, obs, rtc, request, output):
        if request.get("mode") != self.mode:
            self.mode = request.get("mode")
            self.gate = InterventionSession(execution_contract(self.runtime.manifest, self.mode))
            self.noise = CorrelatedNoise()
        return self.runtime.infer(self, obs, rtc, request, output)
