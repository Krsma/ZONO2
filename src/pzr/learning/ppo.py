"""Small PyTorch PPO implementation for continuous-control experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised when optional extra is absent.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for clipped PPO."""

    hidden_sizes: tuple[int, ...] = (128, 128)
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


class ActorCritic(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Gaussian actor-critic with a learned diagonal log standard deviation."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        _require_torch()
        super().__init__()
        hidden = tuple(int(size) for size in hidden_sizes)
        self.actor = _mlp(observation_dim, action_dim, hidden)
        self.critic = _mlp(observation_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def distribution(self, observations: Any) -> Any:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, observations: Any) -> Any:
        return self.critic(observations).squeeze(-1)

    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            dist = self.distribution(obs)
            raw_action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            value = self.value(obs)
        action = torch.clamp(raw_action, -1.0, 1.0).cpu().numpy()[0]
        return action.astype(float), float(log_prob.item()), float(value.item())


@dataclass
class RolloutBuffer:
    """In-memory rollout buffer with GAE advantage computation."""

    observations: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]

    @classmethod
    def empty(cls) -> "RolloutBuffer":
        return cls([], [], [], [], [], [])

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def to_tensors(self, *, last_value: float, config: PPOConfig) -> dict[str, Any]:
        _require_torch()
        count = len(self)
        if count == 0:
            raise ValueError("cannot build PPO tensors from an empty rollout")
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros(count, dtype=np.float32)
        last_gae = 0.0
        for index in range(count - 1, -1, -1):
            if index == count - 1:
                next_nonterminal = 1.0 - dones[index]
                next_value = float(last_value)
            else:
                next_nonterminal = 1.0 - dones[index]
                next_value = float(values[index + 1])
            delta = rewards[index] + config.gamma * next_value * next_nonterminal - values[index]
            last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
            advantages[index] = last_gae
        returns = advantages + values
        std = float(np.std(advantages))
        if std > 1e-8:
            advantages = (advantages - float(np.mean(advantages))) / (std + 1e-8)
        else:
            advantages = advantages - float(np.mean(advantages))
        return {
            "observations": torch.as_tensor(np.asarray(self.observations), dtype=torch.float32),
            "actions": torch.as_tensor(np.asarray(self.actions), dtype=torch.float32),
            "old_log_probs": torch.as_tensor(np.asarray(self.log_probs), dtype=torch.float32),
            "returns": torch.as_tensor(returns, dtype=torch.float32),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32),
        }


class PPOTrainer:
    """Clipped PPO optimizer for an :class:`ActorCritic` model."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        _require_torch()
        self.config = config or PPOConfig()
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.model = ActorCritic(observation_dim, action_dim, self.config.hidden_sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)

    def act(self, observation: np.ndarray, *, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        self.model.eval()
        return self.model.act(observation, deterministic=deterministic)

    def value(self, observation: np.ndarray) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        self.model.eval()
        with torch.no_grad():
            return float(self.model.value(obs).item())

    def update(self, rollout: RolloutBuffer, *, last_value: float = 0.0) -> dict[str, float]:
        self.model.train()
        batch = rollout.to_tensors(last_value=last_value, config=self.config)
        count = int(batch["observations"].shape[0])
        indices = np.arange(count)
        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }
        minibatch_size = min(max(1, self.config.minibatch_size), count)
        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, count, minibatch_size):
                mb = indices[start : start + minibatch_size]
                observations = batch["observations"][mb]
                actions = batch["actions"][mb]
                old_log_probs = batch["old_log_probs"][mb]
                advantages = batch["advantages"][mb]
                returns = batch["returns"][mb]

                dist = self.model.distribution(observations)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                values = self.model.value(observations)
                value_loss = 0.5 * torch.mean((returns - values) ** 2)
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()
                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.item()))
                stats["approx_kl"].append(float(approx_kl.item()))
        return {name: float(np.mean(values)) for name, values in stats.items()}

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        _require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "pzr_ppo_actor_critic",
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
                "model_state": self.model.state_dict(),
                "metadata": {} if metadata is None else dict(metadata),
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PPOTrainer":
        _require_torch()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = PPOConfig(**checkpoint["config"])
        trainer = cls(
            int(checkpoint["observation_dim"]),
            int(checkpoint["action_dim"]),
            config,
            seed=0,
        )
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.model.eval()
        return trainer


def _mlp(input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]) -> Any:
    layers: list[Any] = []
    previous = int(input_dim)
    for hidden in hidden_sizes:
        layers.append(nn.Linear(previous, int(hidden)))
        layers.append(nn.Tanh())
        previous = int(hidden)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for PPO controller training. "
            "Install the learning extra with `python -m pip install -e .[learning]`."
        )
