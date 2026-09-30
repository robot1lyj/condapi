"""RLinf-style Gaussian/reference learner in the YAM residual action space.

The native token implementation is pinned separately. H50 residuals, queue state,
execution masks and variable-duration returns are YAM adaptations, not a claim
of an unchanged RLinf Franka experiment. No VLA optimizer is constructed here.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .learner import Learner as PartsLearner
from .rlt_contract import UPSTREAM_COMMIT
from .rlt_contract import load_token_manifest
from .state import NON_VISUAL_DIM
from .state import REFERENCE_SLICE
from .vendor.rlinf.rlt_token_transformer import RLTTokenEncoder


def network(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.Tanh()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims, fixed_std):
        super().__init__()
        self.network = network(state_dim, hidden_dims, 300)
        self.fixed_std = fixed_std
        # Zero residual mean is a YAM warm-start, not RLinf's full-action init.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def mean(self, state):
        return self.network(state)

    def forward(self, state):
        return torch.tanh(self.mean(state))

    def sample(self, state):
        mu = self.mean(state)
        return torch.tanh(mu + self.fixed_std * torch.randn_like(mu))


class Critic(nn.Module):
    def __init__(self, state_dim, feature_dim, hidden_dims):
        super().__init__()
        self.reference = slice(feature_dim + REFERENCE_SLICE.start, feature_dim + REFERENCE_SLICE.stop)
        # Delta dynamics depend on the reference; retain it in the critic state.
        self.first = network(state_dim + 600, hidden_dims, 1)
        self.second = network(state_dim + 600, hidden_dims, 1)

    def forward(self, state, action, mask):
        value = torch.cat((state, action * mask, mask), dim=1)
        return self.first(value), self.second(value)


class Learner(PartsLearner):
    def __init__(self, state_dim, feature_dim, recipe, mean, std, *, device):
        self.recipe, self.device, self.step = recipe, torch.device(device), 0
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("RLT updates require an approved server CUDA device")
        self.actor = Actor(state_dim, recipe["hidden_dims"], recipe["fixed_std"]).to(self.device)
        self.critic = Critic(state_dim, feature_dim, recipe["hidden_dims"]).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(requires_grad=False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(requires_grad=False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=recipe["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=recipe["critic_lr"])
        self.mean, self.std = torch.as_tensor(mean, device=self.device), torch.as_tensor(std, device=self.device)
        self.reference = slice(feature_dim + REFERENCE_SLICE.start, feature_dim + REFERENCE_SLICE.stop)

    def update(self, arrays):
        self.step += 1
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in arrays.items()}
        state = (batch["state"].float() - self.mean) / self.std
        following = (batch["next_state"].float() - self.mean) / self.std
        mask, next_mask = batch["action_mask"].float(), batch["next_action_mask"].float()
        with torch.no_grad():
            target = batch["reward"].float()[:, None].clone()
            continuing = batch["bootstrap"]
            if continuing.any():
                action = self.actor_target.sample(following[continuing]) * next_mask[continuing]
                q1, q2 = self.critic_target(following[continuing], action, next_mask[continuing])
                discount = self.recipe["gamma"] ** batch["elapsed_steps"][continuing].float()
                target[continuing] += discount[:, None] * torch.minimum(q1, q2)
        first, second = self.critic(state, batch["action"].float(), mask)
        critic_loss = functional.mse_loss(first, target) + functional.mse_loss(second, target)
        if not torch.isfinite(critic_loss):
            raise ValueError("Nonfinite RLT critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.recipe["gradient_clip"], error_if_nonfinite=True)
        self.critic_optimizer.step()
        metrics = {"critic_loss": critic_loss.item()}
        if self.step % self.recipe["policy_frequency"] == 0:
            self.critic.requires_grad_(requires_grad=False)
            try:
                actor_state = state.clone()
                dropped = torch.rand(state.shape[0], device=self.device) < self.recipe["reference_dropout"]
                actor_state[dropped, self.reference] = 0
                predicted = self.actor.sample(actor_state)
                q, _ = self.critic(state, predicted, mask)
                # Staying near the VLA reference means zero in residual coordinates.
                penalty = (predicted.square() * mask).sum() / mask.sum().clamp_min(1)
                loss = -self.recipe["q_weight"] * q.mean() + self.recipe["reference_weight"] * penalty
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite RLT actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.recipe["gradient_clip"], error_if_nonfinite=True
                )
                self.actor_optimizer.step()
            finally:
                self.critic.requires_grad_(requires_grad=True)
            with torch.no_grad():
                for source, destination in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                    for p, target_p in zip(source.parameters(), destination.parameters(), strict=True):
                        target_p.lerp_(p, self.recipe["tau"])
            metrics.update(actor_loss=loss.item(), reference_penalty=penalty.item(), q_pi=q.mean().item())
        return metrics


class ActorRunner:
    def __init__(self, bundle, *, device):
        self.device = torch.device(device)
        self.actor = Actor(bundle["state_dim"], bundle["hidden_dims"], bundle["fixed_std"]).to(self.device).eval()
        self.actor.load_state_dict(bundle["weights"])
        self.actor.requires_grad_(requires_grad=False)
        self.mean, self.std = (
            torch.as_tensor(bundle["mean"], device=self.device, dtype=torch.float32),
            torch.as_tensor(bundle["std"], device=self.device, dtype=torch.float32),
        )
        if (
            self.mean.shape != (bundle["feature_dim"] + NON_VISUAL_DIM,)
            or self.std.shape != self.mean.shape
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("Invalid RLT actor normalization")

    def _mean(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != tuple(self.mean.shape) or not np.isfinite(state).all():
            raise ValueError("Invalid RLT inference state")
        with torch.inference_mode():
            value = torch.as_tensor(state, device=self.device)[None]
            return self.actor.mean((value - self.mean) / self.std)[0].cpu().numpy().reshape(50, 6)

    def __call__(self, state):
        return np.tanh(self._mean(state)).astype(np.float32)

    def sample(self, state, rng):
        return np.tanh(self._mean(state) + rng.normal(0, self.actor.fixed_std, (50, 6))).astype(np.float32)


class TokenRunner:
    def __init__(self, manifest_path, *, device):
        self.manifest, asset = load_token_manifest(manifest_path)
        bundle = torch.load(asset, map_location="cpu", weights_only=True)
        if (
            bundle.get("schema") != "yam_rlt_token_weights_v1"
            or bundle.get("upstream_commit") != UPSTREAM_COMMIT
            or bundle.get("architecture") != self.manifest["architecture"]
        ):
            raise ValueError("Token bundle/source mismatch")
        weights = bundle["weights"]
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in weights.values()):
            raise ValueError("Finite FP32 token weights required")
        self.encoder = RLTTokenEncoder(**self.manifest["architecture"]).to(device).eval()
        self.encoder.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in weights.items() if k.startswith("encoder.")}
        )
        self.encoder.requires_grad_(requires_grad=False)

    def __call__(self, prefix, mask):
        device = next(self.encoder.parameters()).device
        with torch.inference_mode():
            value = self.encoder(prefix.to(device=device, dtype=torch.float32), mask.to(device=device))
        return value[0].reshape(-1).cpu().numpy()
