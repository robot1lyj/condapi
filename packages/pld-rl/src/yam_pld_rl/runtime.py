"""Optional Thor postprocessor. Loading it does not enable physical collection."""

import json
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
from yam_pld_rl.block_contract import feasible_scale
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import integer
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import validate_contract
from yam_pld_rl.data import sha256
from yam_pld_rl.intervention import InterventionSession
from yam_pld_rl.kinematics import YamKinematics
from yam_pld_rl.sac import SquashedActor


class BlockRuntime:
    def __init__(self, bundle, encoder_weights, *, base_weights_sha256, device="cuda"):
        bundle = Path(bundle)
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        self.contract = self.manifest["contract"]
        validate_contract(self.contract)
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

    @property
    def metadata(self):
        return {"schema": SCHEMA, "expert": "PLD-Grasp", "snapshot_sha256": self.snapshot,
                "contract_sha256": fingerprint(self.contract), "modes": ["off", "shadow", "probe", "collect", "eval"],
                "block_steps": self.contract["block_steps"], "segmentation_owner": "server",
                "gate_owner": "thor", "recording": "native_HIL_with_full_pld_reply"}

    def new_session(self):
        """Share GPU networks, isolate causal gate state per websocket connection."""
        return RuntimeSession(self)

    @torch.inference_mode()
    def infer(self, session, obs, rtc, request, output):
        start = time.monotonic()
        c = self.contract
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
        base = np.asarray(output["actions"])
        state = array(obs["observation.state"], (14,), "state")
        previous = array(request["previous_command"], (14,), "last submitted command")
        heights = self.fk.heights(state)
        gate = session.select(request, rtc, state, base, heights)
        trace = {**{k: request[k] for k in ("run_id", "episode_id", "mode", "request_id")},
                 **self.metadata, "epoch": epoch, "observation_id": observation_id,
                 "decision_id": f"e{epoch}:r{rid}", "gate": gate,
                 "start_tick": tick, "delay_steps": delay, "base": base, "state": state,
                 "previous_command": previous, "base_z_m": heights}
        if not gate["active"]:
            return {**output, "pld": trace}
        arm, elapsed = gate["arm"], gate["elapsed_seconds"]
        scale = feasible_scale(base, previous, arm, delay, c)
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
        x = observation_vector(features, state, base, previous, arm, delay, elapsed, scale, self.manifest["normalization"])
        mode = request["mode"]
        u, logp = self.actor(torch.from_numpy(x)[None].to(self.device), deterministic=mode == "eval")
        u = u[0].cpu().numpy()
        action = scale * u if mode in {"collect", "eval"} else np.zeros(7, dtype=np.float32)
        actions = decode(base, action, arm, delay, c)
        trace.update(arm=arm, window_id=gate["window_id"], elapsed_seconds=elapsed,
                     envelope_end_tick=tick + delay + c["block_steps"], x=x, scale=scale, sample_u=u,
                     action=action, log_prob_u=float(logp[0]), actions=actions,
                     encoder_ms_included_total_ms=(time.monotonic() - start) * 1000)
        return {**output, "actions": actions, "pld": trace}


class RuntimeSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.gate = InterventionSession(runtime.contract)

    def reset(self):
        self.gate.reset()

    def __call__(self, obs, rtc, request, output):
        return self.runtime.infer(self.gate, obs, rtc, request, output)
