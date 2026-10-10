"""Calibrated CQL critic warm-start, followed by residual SAC.

Algorithm references: Cal-QL (Nakamoto et al., NeurIPS 2023), PLD
(arXiv:2511.00091). This is an adaptation: offline updates freeze the actor,
and there is no behavior-cloning actor prewarm. See README for readiness.
This module is only executed on an approved GPU compute node.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
import math
import socket

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional


@dataclass(frozen=True)
class LearnerConfig:
    feature_dim: int
    action_lower: tuple[float, ...]
    action_upper: tuple[float, ...]
    residual_bounds: tuple[float, ...]
    base_fingerprint: str
    feature_fingerprint: str
    decision_contract: str
    gamma_per_second: float
    approved_compute_hosts: tuple[str, ...]
    hidden: int = 256
    learning_rate: float = 3e-4
    tau: float = 0.005
    cql_weight: float = 1.0
    cql_samples: int = 10
    entropy_alpha: float = 0.01
    initial_std: float = 0.05

    def __post_init__(self):
        vectors = (self.action_lower, self.action_upper, self.residual_bounds)
        if any(len(v) != 14 or any(not math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("physical action contract must contain fourteen finite coordinates")
        if any(lo >= hi or bound <= 0 for lo, hi, bound in zip(*vectors, strict=True)):
            raise ValueError("reviewed limits and strictly positive residual bounds are required")
        if not self.base_fingerprint or not self.feature_fingerprint or not self.decision_contract:
            raise ValueError("frozen base, feature, and decision identities are required")
        if not self.approved_compute_hosts:
            raise ValueError("explicit approved server compute hostnames required")
        if not 0 < self.gamma_per_second <= 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid discount/target update")
        if self.feature_dim <= 0 or self.hidden <= 0 or self.cql_samples <= 0:
            raise ValueError("invalid network size")
        if self.learning_rate <= 0 or self.cql_weight < 0 or self.entropy_alpha < 0 or not 0 < self.initial_std <= 1:
            raise ValueError("invalid optimizer/conservative loss/exploration configuration")


def network(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


class ResidualActor(nn.Module):
    def __init__(self, feature_dim, hidden, std):
        super().__init__()
        self.net = network(feature_dim + 14, hidden, 28)
        # Deterministic deployment initially equals the frozen base policy.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[14:].fill_(math.log(std))

    def forward(self, features, base, *, deterministic=False):
        mean, log_std = self.net(torch.cat((features, base), -1)).chunk(2, -1)
        log_std = log_std.clamp(-7, 1)
        distribution = Normal(mean, log_std.exp())
        latent = mean if deterministic else distribution.rsample()
        residual = latent.tanh()
        # Stable tanh Jacobian; density is in normalized residual coordinates.
        log_prob = (distribution.log_prob(latent) - 2 * (math.log(2) - latent - functional.softplus(-2 * latent))).sum(
            -1
        )
        return residual, log_prob


class Critic(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        # Q acts on the actual combined physical target, affinely normalized
        # only for conditioning. No inferred human residual is required.
        self.net = network(features + 14, hidden, 1)

    def forward(self, features, action):
        return self.net(torch.cat((features, action), -1)).squeeze(-1)


class ResidualSAC:
    def __init__(self, config, device="cuda"):
        if socket.gethostname().split(".")[0] not in config.approved_compute_hosts:
            raise RuntimeError("this host is not an approved server compute node")
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("learner execution requires an approved CUDA compute node")
        self.config, self.device, self.steps = config, torch.device(device), 0
        lower, upper, bound = [
            torch.tensor(v, device=self.device)
            for v in (config.action_lower, config.action_upper, config.residual_bounds)
        ]
        self.center, self.scale = (lower + upper) / 2, (upper - lower) / 2
        self.bound = bound / self.scale
        self.actor = ResidualActor(config.feature_dim, config.hidden, config.initial_std).to(self.device)
        self.critics = nn.ModuleList([Critic(config.feature_dim, config.hidden).to(self.device) for _ in range(2)])
        self.targets = deepcopy(self.critics).requires_grad_(requires_grad=False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.learning_rate)

    def _normalize(self, physical):
        return (physical - self.center) / self.scale

    def _action(self, features, normalized_base, *, deterministic=False):
        residual, log_prob = self.actor(features, normalized_base, deterministic=deterministic)
        proposed = normalized_base + self.bound * residual
        # For CQL, evaluate an unclipped proposal and its proper affine density.
        # The online control bridge must reject/limit unsafe proposals; a
        # clipped distribution has atoms and is not this density.
        action_log_prob = log_prob - self.bound.log().sum()
        return proposed, log_prob, action_log_prob

    def _min_q(self, features, action, *, target=False):
        pair = self.targets if target else self.critics
        return torch.minimum(pair[0](features, action), pair[1](features, action))

    def update(self, batch, *, phase):
        if phase not in ("critic_init", "online"):
            raise ValueError("explicit learner phase required")
        b = {key: torch.as_tensor(value, device=self.device, dtype=torch.float32) for key, value in batch.items()}
        required = {
            "features",
            "next_features",
            "base_action",
            "next_base_action",
            "action",
            "dt",
            "reward",
            "terminated",
            "mc_return",
            "calibration_valid",
        }
        if not required <= b.keys() or any(not torch.isfinite(b[key]).all() for key in required):
            raise ValueError("missing/nonfinite versioned replay batch")
        if any(b[key].ndim != 1 for key in ("dt", "reward", "terminated", "mc_return", "calibration_valid")):
            raise ValueError("scalar replay columns must have shape [batch]")
        if (b["dt"] <= 0).any() or ((b["terminated"] != 0) & (b["terminated"] != 1)).any():
            raise ValueError("invalid transition duration or termination")
        features, next_features = b["features"], b["next_features"]
        if features.ndim != 2 or features.shape[-1] != self.config.feature_dim or next_features.shape != features.shape:
            raise ValueError("feature schema mismatch")
        for key in ("base_action", "next_base_action", "action"):
            if b[key].shape != (len(features), 14) or (self._normalize(b[key]).abs() > 1.00001).any():
                raise ValueError("physical targets exceed reviewed action limits")
        base, next_base, action = [self._normalize(b[key]) for key in ("base_action", "next_base_action", "action")]
        with torch.no_grad():
            next_action, log_prob, _ = self._action(next_features, next_base, deterministic=phase == "critic_init")
            next_value = self._min_q(next_features, next_action.clamp(-1, 1), target=True)
            if phase == "online":
                next_value -= self.config.entropy_alpha * log_prob
            discount = torch.pow(torch.full_like(b["dt"], self.config.gamma_per_second), b["dt"])
            target = b["reward"] + (1 - b["terminated"]) * discount * next_value
        predictions = [critic(features, action) for critic in self.critics]
        td_loss = sum(functional.mse_loss(prediction, target) for prediction in predictions)
        conservative = features.new_zeros(())
        if phase == "critic_init":
            size, samples = len(features), self.config.cql_samples
            repeated = features[:, None, :].expand(-1, samples, -1).reshape(-1, features.shape[-1])
            repeated_base = base[:, None, :].expand(-1, samples, -1).reshape(-1, 14)
            with torch.no_grad():
                candidate, _, density = self._action(repeated, repeated_base)
                random_actions = torch.rand_like(candidate) * 2 - 1
            for critic, prediction in zip(self.critics, predictions, strict=True):
                random_q = critic(repeated, random_actions).reshape(size, samples) + 14 * math.log(2)
                candidate_q = critic(repeated, candidate).reshape(size, samples)
                lower_bound = b["mc_return"][:, None]
                calibrated = torch.where(
                    b["calibration_valid"][:, None].bool(), torch.maximum(candidate_q, lower_bound), candidate_q
                )
                # Integrate only over the executable action box. Retain the
                # original proposal density and denominator for importance
                # sampling; never pretend a clipped action has that density.
                valid_proposal = (candidate.abs() <= 1).all(-1).reshape(size, samples)
                weighted = (calibrated - density.reshape(size, samples)).masked_fill(~valid_proposal, -torch.inf)
                ood = torch.cat((random_q, weighted), 1)
                conservative += (torch.logsumexp(ood, 1) - math.log(2 * samples) - prediction).mean()
        loss = td_loss + self.config.cql_weight * conservative
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        actor_loss = features.new_zeros(())
        if phase == "online":
            self.critics.requires_grad_(requires_grad=False)
            proposed, log_prob, _ = self._action(features, base)
            actor_loss = (self.config.entropy_alpha * log_prob - self._min_q(features, proposed.clamp(-1, 1))).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critics.requires_grad_(requires_grad=True)
        with torch.no_grad():
            for target_parameter, parameter in zip(self.targets.parameters(), self.critics.parameters(), strict=True):
                target_parameter.lerp_(parameter, self.config.tau)
        self.steps += 1
        return {
            "critic_loss": float(loss.detach()),
            "td_loss": float(td_loss.detach()),
            "conservative_loss": float(conservative.detach()),
            "actor_loss": float(actor_loss.detach()),
            "actor_updated": phase == "online",
            "updates": self.steps,
        }

    def state_dict(self):
        return {
            "schema": "yam_residual_sac_v1",
            "config": asdict(self.config),
            "steps": self.steps,
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "targets": self.targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        if state["schema"] != "yam_residual_sac_v1" or state["config"] != asdict(self.config):
            raise ValueError("checkpoint contract changed; open a new run")
        for key in ("actor", "critics", "targets", "actor_optimizer", "critic_optimizer"):
            getattr(self, key).load_state_dict(state[key])
        self.steps = state["steps"]
