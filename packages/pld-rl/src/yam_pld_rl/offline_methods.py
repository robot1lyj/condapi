"""Paper-derived offline objectives with explicit YAM residual adaptations.

These learners never issue robot commands. Action ``u`` is a dimensionless
residual; Q consumes base + radius*u in training-action standard deviation units.
The radius is NOT a validated physical deployment limit.
"""

import copy

import torch
from torch import nn


def mlp(inputs, outputs, width, depth, layer_norm):
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(inputs, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.ReLU())
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class QPair(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.q1 = mlp(dim + 7, 1, cfg["width"], cfg["depth"], cfg["critic_layer_norm"])
        self.q2 = copy.deepcopy(self.q1)
        # Independent initialization, including the two Q heads.
        for layer in self.q2.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, action):
        xa = torch.cat((x, action), -1)
        return self.q1(xa).squeeze(-1), self.q2(xa).squeeze(-1)


class ResidualActor(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.net = mlp(dim, 7, cfg["width"], cfg["depth"], layer_norm=False)
        # Start at the existing Pi behavior, with no separate BC warmup.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if cfg["method"] == "iql":
            self.log_std = nn.Parameter(torch.zeros(7))

    def forward(self, x):
        return torch.tanh(self.net(x))

    def log_prob(self, x, u):
        # As in the official IQL normal policy: bounded mean, unsquashed Normal.
        distribution = torch.distributions.Normal(self(x), self.log_std.clamp(-5, 2).exp())
        return distribution.log_prob(u).sum(-1)


def expectile_loss(difference, expectile):
    weight = torch.where(difference > 0, expectile, 1 - expectile)
    return weight * difference.square()


def bellman_target(reward, discount, terminal, next_value):
    return reward + discount * (1 - terminal) * next_value


class OfflineLearner:
    def __init__(self, dim, cfg, device="cuda"):
        self.cfg = cfg
        self.critic = QPair(dim, cfg).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(requires_grad=False)
        self.actor = ResidualActor(dim, cfg).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(requires_grad=False)
        self.value = (
            mlp(dim, 1, cfg["width"], cfg["depth"], layer_norm=False).to(device) if cfg["method"] == "iql" else None
        )
        self.qopt = torch.optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=cfg["learning_rate"])
        self.vopt = torch.optim.Adam(self.value.parameters(), lr=cfg["learning_rate"]) if self.value else None
        self.actor_updates = 0

    def action(self, x, base):
        return base + self.cfg["residual_radius_std"] * self.actor(x)

    def _optimize(self, loss, model, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), self.cfg["gradient_clip"])
        if not torch.isfinite(grad):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        return float(grad)

    @torch.no_grad()
    def backup(self, b, *, noise=True):
        if self.value is not None:
            continuation = self.value(b["nx"]).squeeze(-1)
        else:
            u = self.actor_target(b["nx"])
            if noise:
                epsilon = (torch.randn_like(u) * self.cfg["policy_noise"]).clamp(
                    -self.cfg["noise_clip"], self.cfg["noise_clip"]
                )
                u = (u + epsilon).clamp(-1, 1)
            action = b["next_base"] + self.cfg["residual_radius_std"] * u
            q1, q2 = self.target(b["nx"], action)
            continuation = torch.minimum(q1, q2)
            if self.cfg["method"] == "rebrac":
                # ReBRAC regularizes the next-action value inside the backup.
                continuation -= self.cfg["critic_bc"] * (u - b["next_u"]).square().sum(-1)
        return bellman_target(b["reward"], b["discount"], b["terminal"], continuation)

    def update(self, b, step):
        cfg = self.cfg
        log = {}
        if self.value is not None:
            with torch.no_grad():
                t1, t2 = self.target(b["x"], b["action"])
                in_sample_q = torch.minimum(t1, t2)
            value = self.value(b["x"]).squeeze(-1)
            vl = expectile_loss(in_sample_q - value, cfg["expectile"]).mean()
            log["train/value_grad_norm"] = self._optimize(vl, self.value, self.vopt)
            log["train/value_loss"] = float(vl.detach())
        y = self.backup(b)
        q1, q2 = self.critic(b["x"], b["action"])
        loss = (q1 - y).square().mean() + (q2 - y).square().mean()
        if max(float(q1.detach().abs().max()), float(q2.detach().abs().max())) > 100:
            raise FloatingPointError("|Q|>100; preserve this failed run, do not deploy")
        log["grad_norm"] = self._optimize(loss, self.critic, self.qopt)
        log.update(
            loss=float(loss.detach()),
            **{"train/td_loss": float(loss.detach()), "q/q_mean": float(((q1 + q2) / 2).detach().mean())},
        )
        actor_step = self.value is not None or step % cfg["policy_frequency"] == 0
        if actor_step:
            if self.value is not None:
                with torch.no_grad():
                    advantage = in_sample_q - self.value(b["x"]).squeeze(-1)
                    weights = (advantage * cfg["advantage_temperature"]).clamp(max=4.605170186).exp()
                aloss = -(weights * self.actor.log_prob(b["x"], b["u"])).mean()
                log.update(
                    **{"train/advantage": float(advantage.mean()), "train/advantage_weight": float(weights.mean())}
                )
            else:
                self.critic.requires_grad_(requires_grad=False)
                try:
                    u = self.actor(b["x"])
                    action = b["base"] + cfg["residual_radius_std"] * u
                    p1, p2 = self.critic(b["x"], action)
                    policy_q = p1 if cfg["method"] == "td3bc" else torch.minimum(p1, p2)
                    scale = cfg["q_coefficient"] / policy_q.detach().abs().mean().clamp(min=1e-6)
                    squared_error = (u - b["u"]).square()
                    bc = squared_error.mean() if cfg["method"] == "td3bc" else squared_error.sum(-1).mean()
                    aloss = -scale * policy_q.mean() + cfg["actor_bc"] * bc
                    log["train/actor_bc"] = float(bc.detach())
                finally:
                    # Parameter gradients were disabled when constructing this graph;
                    # only actor gradients can flow through the policy objective.
                    self.critic.requires_grad_(requires_grad=True)
            log["train/actor_grad_norm"] = self._optimize(aloss, self.actor, self.aopt)
            log["train/actor_loss"] = float(aloss.detach())
            self.actor_updates += 1
            if self.value is None:
                self.polyak(self.actor_target, self.actor)
        if self.value is not None or actor_step:
            self.polyak(self.target, self.critic)
        return log

    @torch.no_grad()
    def polyak(self, target, current):
        for t, p in zip(target.parameters(), current.parameters(), strict=True):
            t.lerp_(p, self.cfg["tau"])

    def state_dict(self):
        names = ["critic", "target", "actor", "actor_target", "value", "qopt", "aopt", "vopt"]
        return {name: getattr(self, name).state_dict() for name in names if getattr(self, name) is not None} | {
            "actor_updates": self.actor_updates
        }

    def load_state_dict(self, state):
        for name, value in state.items():
            if name != "actor_updates":
                getattr(self, name).load_state_dict(value)
        self.actor_updates = state["actor_updates"]
