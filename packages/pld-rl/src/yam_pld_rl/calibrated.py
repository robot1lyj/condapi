"""Bounded RTC adaptations of Cal-QL SAC and PARTS-style TD3+successful BC.

These are explicit project variants, not exact paper reproductions. Conservative
density corrections and smoothing live in u-space; Q always sees feasible a.
"""

import copy
import math

import torch
from torch.nn import functional

from yam_pld_rl.sac import SacLearner


def feasible(u, positive, negative):
    return u * torch.where(u < 0, negative, positive)


def calibrated_gap(random_q, policy_q, data_q, logp, mc_return):
    """CalQL's MC lower bound is inside the policy OOD penalty, not a TD label.

    Both proposal families estimate integrals over the same [-1,1]^7 domain.
    Uniform proposals have log density -7 log 2. Policy proposals carry their
    exact tanh-Gaussian latent density. Subtracting log(2N) is a fixed constant.
    """
    corrected = torch.cat((random_q + 7 * math.log(2),
                           torch.maximum(policy_q, mc_return[:, None]) - logp.detach()), dim=1)
    return (torch.logsumexp(corrected, dim=1) - math.log(corrected.shape[1]) - data_q).mean()


class CalibratedLearner(SacLearner):
    def __init__(self, dim, cfg, device="cuda"):
        super().__init__(dim, cfg, device)
        if cfg["method"] not in ("calql", "parts_td3bc"):
            raise ValueError("unknown calibrated method")
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(requires_grad=False)
        if cfg["method"] == "parts_td3bc":
            # The deployed network remains runtime-compatible; only its mean is
            # optimized. Online collection std is held at the recorded bootstrap.
            self.actor.std.requires_grad_(requires_grad=False)

    @torch.no_grad()
    def backup(self, b):
        value = torch.zeros_like(b["reward"])
        live = b["terminal"] == 0
        if live.any():
            x = b["nx"][live]
            if self.cfg["method"] == "parts_td3bc":
                u, _ = self.actor_target(x, deterministic=True)
                noise = torch.randn_like(u) * self.cfg["target_noise"]
                u = (u + noise.clamp(-self.cfg["noise_clip"], self.cfg["noise_clip"])).clamp(-1, 1)
            else:
                u, _ = self.actor(x)
            a = feasible(u, b["next_scale"][live], b["next_negative_scale"][live])
            q1, q2 = self.target(x, a)
            value[live] = torch.minimum(q1, q2)
        # No entropy in backup: Q is on the original local-reward scale.
        return b["reward"] + b["discount"] * value

    def conservative(self, b, q1, q2):
        mask = (b["scale"] + b["negative_scale"]).sum(-1) > 0
        if not mask.any():
            return q1.sum() * 0
        n = self.cfg["cql_samples"]
        x = b["x"][mask].repeat_interleave(n, dim=0)
        positive = b["scale"][mask].repeat_interleave(n, dim=0)
        negative = b["negative_scale"][mask].repeat_interleave(n, dim=0)
        with torch.no_grad():
            u, logp = self.actor(x)
            random_u = torch.rand_like(u) * 2 - 1
        pq = self.critic(x, feasible(u, positive, negative))
        rq = self.critic(x, feasible(random_u, positive, negative))
        return sum(calibrated_gap(r.reshape(-1, n), p.reshape(-1, n), d[mask],
                                  logp.reshape(-1, n), b["mc_return"][mask])
                   for r, p, d in zip(rq, pq, (q1, q2), strict=True))

    def update(self, b, *, update_actor=True):
        mc_phase = self.updates < self.cfg["mc_warmup_steps"]
        target = b["mc_return"] if mc_phase else self.backup(b)
        q1, q2 = self.critic(b["x"], b["action"])
        td = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
        cql = self.conservative(b, q1, q2) if not mc_phase and self.cfg["method"] == "calql" else td * 0
        qloss = td + self.cfg["cql_weight"] * cql
        grad = self._step(qloss, self.critic.parameters(), self.qopt)
        self.updates += 1
        log = {"loss": float(qloss.detach()), "train/td_loss": float(td.detach()),
               "train/cql_gap": float(cql.detach()), "grad_norm": grad,
               "train/q_abs_max": float(torch.maximum(q1.abs(), q2.abs()).max().detach()),
               "train/q_disagreement": float((q1-q2).abs().mean().detach()),
               "phase/mc_initialization": int(mc_phase)}
        control = (b["scale"] + b["negative_scale"]).sum(-1) > 0
        if update_actor and not mc_phase and self.updates % self.cfg["critic_actor_ratio"] == 0 and control.any():
            self.critic.requires_grad_(requires_grad=False)
            u, logp = self.actor(b["x"][control], deterministic=self.cfg["method"] == "parts_td3bc")
            a = feasible(u, b["scale"][control], b["negative_scale"][control])
            p1, p2 = self.critic(b["x"][control], a)
            if self.cfg["method"] == "parts_td3bc":
                success = b["attempt_outcome"][control] == 1
                failure = ~success
                bc = functional.mse_loss(a[success], b["action"][control][success]) if success.any() else a.sum()*0
                anchor = a[failure].square().mean() if failure.any() else a.sum()*0
                coefficient = self.cfg["td3_bc_alpha"] / p1.abs().mean().detach().clamp_min(0.1)
                aloss = -coefficient * p1.mean() + bc + self.cfg["failure_zero_weight"] * anchor
                log.update({"train/success_bc": float(bc.detach()), "train/failure_anchor": float(anchor.detach()),
                            "train/q_actor_coefficient": float(coefficient)})
            else:
                aloss = (self.log_alpha.exp().detach()*logp-torch.minimum(p1, p2)).mean()
            self._step(aloss, self.actor.parameters(), self.aopt)
            self.critic.requires_grad_(requires_grad=True)
            if self.cfg["method"] == "calql":
                tloss = -(self.log_alpha * (logp.detach()+self.cfg["target_entropy"])).mean()
                self._step(tloss, [self.log_alpha], self.topt)
                with torch.no_grad():
                    self.log_alpha.clamp_(math.log(1e-5), math.log(1))
            self.actor_updates += 1
            log.update({"train/actor_loss": float(aloss.detach()), "train/residual_rms": float(a.square().mean().sqrt().detach()),
                        "train/residual_saturation": float((u.abs()>0.95).float().mean())})
        with torch.no_grad():
            for target_p, p in zip(self.target.parameters(), self.critic.parameters(), strict=True):
                target_p.lerp_(p, self.cfg["tau"])
            if self.actor_updates and self.updates % self.cfg["critic_actor_ratio"] == 0:
                for target_p, p in zip(self.actor_target.parameters(), self.actor.parameters(), strict=True):
                    target_p.lerp_(p, self.cfg["tau"])
        log.update(alpha=float(self.log_alpha.exp().detach()), actor_updates=self.actor_updates)
        return log

    def state_dict(self):
        return {**super().state_dict(), "actor_target": self.actor_target.state_dict()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.actor_target.load_state_dict(state["actor_target"])
