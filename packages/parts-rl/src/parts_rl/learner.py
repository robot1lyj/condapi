"""Subtask TD3+BC learner; executed only in a server model environment.

The TD3 update follows Fujimoto's TD3+BC algorithm. Success-only masked BC,
failure anchoring, H50 plan actions and variable-duration discounting implement
the project PARTS contract; these are not upstream paper hyperparameters.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .state import REFERENCE_SLICE


def mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        self.network = mlp(state_dim, hidden_dims, 300)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state):
        return torch.tanh(self.network(state))


class TwinCritic(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        self.first = mlp(state_dim + 600, hidden_dims, 1)
        self.second = mlp(state_dim + 600, hidden_dims, 1)

    def forward(self, state, action, mask):
        value = torch.cat((state, action * mask, mask), dim=1)
        return self.first(value), self.second(value)


class Learner:
    def __init__(self, state_dim, feature_dim, recipe, mean, std, *, device):
        self.recipe = recipe
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("PARTS training requires an approved server CUDA device")
        self.actor = Actor(state_dim, recipe["hidden_dims"]).to(self.device)
        self.critic = TwinCritic(state_dim, recipe["hidden_dims"]).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(requires_grad=False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(requires_grad=False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=recipe["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=recipe["critic_lr"])
        self.mean = torch.as_tensor(mean, device=self.device)
        self.std = torch.as_tensor(std, device=self.device)
        self.reference = slice(feature_dim + REFERENCE_SLICE.start, feature_dim + REFERENCE_SLICE.stop)
        self.step = 0

    def update(self, arrays):
        self.step += 1
        batch = {key: torch.as_tensor(value, device=self.device) for key, value in arrays.items()}
        state = ((batch["state"].float() - self.mean) / self.std).clone()
        following = ((batch["next_state"].float() - self.mean) / self.std).clone()
        for value in (state, following):
            drop = torch.rand(value.shape[0], device=self.device) < self.recipe["reference_dropout"]
            value[drop, self.reference] = 0
        mask, next_mask = batch["action_mask"].float(), batch["next_action_mask"].float()
        with torch.no_grad():
            noise = (torch.randn_like(batch["action"]) * self.recipe["target_noise"]).clamp(
                -self.recipe["target_noise_clip"], self.recipe["target_noise_clip"]
            )
            # Terminal placeholder states are unused, including by target nets.
            target = batch["reward"].float()[:, None].clone()
            continuing = batch["bootstrap"]
            if continuing.any():
                next_action = (self.actor_target(following[continuing]) + noise[continuing]).clamp(-1, 1)
                next_action *= next_mask[continuing]
                q1, q2 = self.critic_target(following[continuing], next_action, next_mask[continuing])
                discount = self.recipe["gamma"] ** batch["elapsed_steps"][continuing].float()
                target[continuing] += discount[:, None] * torch.minimum(q1, q2)
        first, second = self.critic(state, batch["action"].float(), mask)
        critic_loss = functional.mse_loss(first, target) + functional.mse_loss(second, target)
        if not torch.isfinite(critic_loss):
            raise ValueError("Nonfinite critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        metrics = {"critic_loss": critic_loss.item()}
        if self.step % self.recipe["policy_frequency"] == 0:
            self.critic.requires_grad_(requires_grad=False)
            try:
                predicted = self.actor(state)
                q, _ = self.critic(state, predicted, mask)
                executed = batch["executed_mask"].float()
                success_mask = executed * batch["success"].float()[:, None]
                failure_mask = executed * (~batch["success"]).float()[:, None]
                bc = (
                    (predicted - batch["action"].float()).square() * success_mask
                ).sum() / success_mask.sum().clamp_min(1)
                anchor = (predicted.square() * failure_mask).sum() / failure_mask.sum().clamp_min(1)
                actor_loss = (
                    -self.recipe["q_weight"] * q.mean()
                    + self.recipe["success_bc_weight"] * bc
                    + self.recipe["failure_anchor_weight"] * anchor
                )
                if not torch.isfinite(actor_loss):
                    raise ValueError("Nonfinite actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()
            finally:
                self.critic.requires_grad_(requires_grad=True)
            with torch.no_grad():
                for source, target_net in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                    for parameter, target_parameter in zip(source.parameters(), target_net.parameters(), strict=True):
                        target_parameter.lerp_(parameter, self.recipe["tau"])
            metrics.update(actor_loss=actor_loss.item(), success_bc=bc.item(), failure_anchor=anchor.item())
        return metrics

    def checkpoint(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "step": self.step,
            "mean": self.mean,
            "std": self.std,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }

    def restore(self, checkpoint):
        for key in ("actor", "critic", "actor_target", "critic_target"):
            getattr(self, key).load_state_dict(checkpoint[key])
        for key in ("actor_optimizer", "critic_optimizer"):
            getattr(self, key).load_state_dict(checkpoint[key])
        self.step = checkpoint["step"]
        self.mean, self.std = checkpoint["mean"].to(self.device), checkpoint["std"].to(self.device)


class ActorRunner:
    """Inference-only actor with the snapshot's frozen train normalization."""

    def __init__(self, bundle, *, device):
        self.device = torch.device(device)
        self.actor = Actor(bundle["state_dim"], bundle["hidden_dims"]).to(self.device).eval()
        self.actor.load_state_dict(bundle["weights"])
        self.actor.requires_grad_(requires_grad=False)
        self.mean = torch.as_tensor(bundle["mean"], device=self.device)
        self.std = torch.as_tensor(bundle["std"], device=self.device)
        if (
            self.mean.shape != (bundle["state_dim"],)
            or self.std.shape != self.mean.shape
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError("Invalid actor normalization")

    def __call__(self, state):
        with torch.inference_mode():
            value = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float32)[None]
            return self.actor((value - self.mean) / self.std)[0].cpu().numpy().reshape(50, 6)
