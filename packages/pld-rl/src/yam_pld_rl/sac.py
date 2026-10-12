"""Residual SAC in a bounded action-block MDP (frozen Pi/visual encoder).

Q sees the physically feasible normalized offset a=scale(s)*tanh(z). Entropy
is defined over u=tanh(z), not over the projected robot command distribution.
This explicit latent entropy avoids undefined densities on collapsed limits.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional

from yam_pld_rl.offline_methods import QPair
from yam_pld_rl.offline_methods import mlp


class SquashedActor(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.body = mlp(dim, 256, cfg["width"], cfg["depth"] - 1, layer_norm=True)
        self.mean = nn.Linear(256, 7)
        self.std = nn.Linear(256, 7)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.std.weight)
        nn.init.constant_(self.std.bias, math.log(cfg["initial_std"]))
        self.max_log_std = math.log(cfg["max_std"])

    def forward(self, x, *, deterministic=False):
        h = self.body(x)
        mean, log_std = self.mean(h), self.std(h).clamp(-5, self.max_log_std)
        z = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        u = torch.tanh(z)
        logp = (-0.5 * (((z - mean) / log_std.exp()).square() + 2 * log_std + math.log(2 * math.pi))
                - 2 * (math.log(2) - z - functional.softplus(-2 * z))).sum(-1)
        return u, logp


class SacLearner:
    def __init__(self, dim, cfg, device="cuda"):
        self.cfg = cfg
        self.actor = SquashedActor(dim, cfg).to(device)
        self.critic = QPair(dim, cfg).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(requires_grad=False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg["initial_alpha"]), device=device))
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=cfg["learning_rate"])
        self.qopt = torch.optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])
        self.topt = torch.optim.Adam([self.log_alpha], lr=cfg["learning_rate"])
        self.updates = 0
        self.actor_updates = 0

    @torch.no_grad()
    def backup(self, b):
        # Never evaluate a fabricated terminal next observation.
        value = torch.zeros_like(b["reward"])
        live = b["terminal"] == 0
        if live.any():
            u, logp = self.actor(b["nx"][live])
            q1, q2 = self.target(b["nx"][live], b["next_scale"][live] * u)
            value[live] = torch.minimum(q1, q2) - self.log_alpha.exp() * logp
        return b["reward"] + b["discount"] * value

    def _step(self, loss, parameters, optimizer):
        params = list(parameters)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite SAC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(params, self.cfg["gradient_clip"])
        if not torch.isfinite(grad):
            raise FloatingPointError("nonfinite SAC gradient")
        optimizer.step()
        return float(grad)

    def update(self, b, *, update_actor=True):
        target = self.backup(b)
        q1, q2 = self.critic(b["x"], b["action"])
        qloss = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
        grad = self._step(qloss, self.critic.parameters(), self.qopt)
        self.updates += 1
        log = {"loss": float(qloss.detach()), "grad_norm": grad,
               "train/q_abs_max": float(torch.maximum(q1.abs(), q2.abs()).max().detach()),
               "train/q_disagreement": float((q1 - q2).abs().mean().detach())}
        if update_actor and self.updates % self.cfg["critic_actor_ratio"] == 0:
            self.critic.requires_grad_(requires_grad=False)
            u, logp = self.actor(b["x"])
            a = b["scale"] * u
            p1, p2 = self.critic(b["x"], a)
            aloss = (self.log_alpha.exp().detach() * logp - torch.minimum(p1, p2)).mean()
            self._step(aloss, self.actor.parameters(), self.aopt)
            self.critic.requires_grad_(requires_grad=True)
            tloss = -(self.log_alpha * (logp.detach() + self.cfg["target_entropy"])).mean()
            self._step(tloss, [self.log_alpha], self.topt)
            with torch.no_grad():
                self.log_alpha.clamp_(math.log(1e-5), math.log(1))
            self.actor_updates += 1
            log.update({"train/actor_loss": float(aloss.detach()), "train/entropy": -float(logp.mean().detach()),
                        "train/residual_rms": float(a.square().mean().sqrt().detach()),
                        "train/residual_saturation": float((u.abs() > 0.95).float().mean())})
        with torch.no_grad():
            for target_p, p in zip(self.target.parameters(), self.critic.parameters(), strict=True):
                target_p.lerp_(p, self.cfg["tau"])
        log.update(alpha=float(self.log_alpha.exp().detach()), actor_updates=self.actor_updates)
        return log

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                "aopt": self.aopt.state_dict(), "qopt": self.qopt.state_dict(), "topt": self.topt.state_dict(),
                "log_alpha": self.log_alpha.detach(), "updates": self.updates, "actor_updates": self.actor_updates}

    def load_state_dict(self, state):
        for key in ("actor", "critic", "target", "aopt", "qopt", "topt"):
            getattr(self, key).load_state_dict(state[key])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.updates, self.actor_updates = state["updates"], state["actor_updates"]
