"""Multi-Agent PPO (MAPPO) trainer.

Centralized training with decentralized execution (CTDE).
Uses a centralized critic conditioned on global state + agent-specific features.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

from src.utils.networks import StochasticActor, CentralizedCritic
from src.utils.buffer import OnPolicyBuffer
from src.utils.evaluator import MetricsTracker


class MAPPOTrainer:
    """Trainer for MAPPO with centralized critic.

    The critic receives both global state and agent-local observation,
    enabling better credit assignment in multi-agent settings.
    """

    def __init__(
        self,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        num_agents: int = 4,
        hidden_dims: list = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        use_popart: bool = False,
        device: torch.device = torch.device("cpu"),
        actor: nn.Module = None,
        critic: nn.Module = None,
    ):
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.use_popart = use_popart

        if actor is not None:
            self.actor = actor.to(device)
        else:
            self.actor = StochasticActor(obs_dim, action_dim, hidden_dims).to(device)
        if critic is not None:
            self.critic = critic.to(device)
        else:
            self.critic = CentralizedCritic(global_obs_dim, obs_dim, hidden_dims).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # PopArt normalization for value targets
        if use_popart:
            self.value_normalizer = PopArt(1)
        else:
            self.value_normalizer = None

        self.metrics = MetricsTracker()

    def train_on_buffer(
        self, buffer: OnPolicyBuffer, writer: SummaryWriter, global_step: int
    ) -> Dict[str, float]:
        """Perform MAPPO update with centralized critic."""
        data = buffer.get_training_data()
        num_steps = buffer.step

        # Compute centralized values using global state
        obs_all_steps = data["obs"]  # (T, N, obs_dim)
        global_all_steps = data["global_state"]  # (T, N*obs_dim)

        with torch.no_grad():
            obs_t = torch.tensor(obs_all_steps, dtype=torch.float32, device=self.device)
            global_t = torch.tensor(global_all_steps, dtype=torch.float32, device=self.device)

            obs_shape = getattr(self, 'obs_shape', None)
            if obs_shape is not None:
                C, H, W = obs_shape
                obs_t = obs_t.reshape(num_steps, self.num_agents, C, H, W)
                global_t = global_t.reshape(num_steps, self.num_agents * C, H, W)

            # Batched centralized values: expand global across agent dimension
            global_expanded = global_t.unsqueeze(1).expand(-1, self.num_agents, *([-1] * (global_t.ndim - 1)))
            global_flat = global_expanded.reshape(-1, *global_t.shape[1:])
            obs_flat = obs_t.reshape(-1, *obs_t.shape[2:])
            cent_values = self.critic(global_flat, obs_flat).squeeze(-1).reshape(num_steps, self.num_agents)

            values_np = cent_values.cpu().numpy()

            # Last step values (batched)
            last_obs = torch.tensor(data["obs"][-1], dtype=torch.float32, device=self.device)
            last_global = torch.tensor(data["global_state"][-1], dtype=torch.float32, device=self.device)
            obs_shape = getattr(self, 'obs_shape', None)
            if obs_shape is not None:
                C, H, W = obs_shape
                last_obs = last_obs.reshape(self.num_agents, C, H, W)
                last_global = last_global.reshape(self.num_agents * C, H, W)
            last_global_batch = last_global.unsqueeze(0).expand(self.num_agents, *([-1] * last_global.ndim))
            last_values = self.critic(last_global_batch, last_obs).squeeze(-1)
            last_values_np = last_values.cpu().numpy()

        last_done = data["dones"][-1]

        # Update buffer values to centralized values and compute GAE
        buffer.values = values_np
        advantages, returns = buffer.compute_gae(last_values_np, last_done)

        # Flatten
        obs_flat = obs_all_steps.reshape(-1, self.obs_dim)
        actions_flat = data["actions"].reshape(-1)
        old_log_probs_flat = data["action_log_probs"].reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # Expand global state to match agent dimension
        global_expanded = np.repeat(global_all_steps, self.num_agents, axis=0)

        # Normalize advantages
        adv_mean, adv_std = advantages_flat.mean(), advantages_flat.std()
        advantages_flat = (advantages_flat - adv_mean) / (adv_std + 1e-8)

        total_batch_size = num_steps * self.num_agents

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(total_batch_size)
            for start in range(0, total_batch_size, self.mini_batch_size):
                idx = indices[start : start + self.mini_batch_size]

                obs_b = torch.tensor(obs_flat[idx], dtype=torch.float32, device=self.device)
                global_b = torch.tensor(global_expanded[idx], dtype=torch.float32, device=self.device)
                obs_shape = getattr(self, 'obs_shape', None)
                if obs_shape is not None:
                    C, H, W = obs_shape
                    obs_b = obs_b.reshape(-1, C, H, W)
                    global_b = global_b.reshape(-1, self.num_agents * C, H, W)
                act_b = torch.tensor(actions_flat[idx], dtype=torch.long, device=self.device)
                old_lp_b = torch.tensor(old_log_probs_flat[idx], dtype=torch.float32, device=self.device)
                adv_b = torch.tensor(advantages_flat[idx], dtype=torch.float32, device=self.device)
                ret_b = torch.tensor(returns_flat[idx], dtype=torch.float32, device=self.device)

                # Actor loss (same as IPPO)
                new_log_probs, entropy = self.actor.evaluate_actions(obs_b, None, act_b)
                ratio = torch.exp(new_log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (centralized)
                values_pred = self.critic(global_b, obs_b).squeeze(-1)
                if self.use_popart:
                    values_norm = self.value_normalizer.normalize(ret_b.unsqueeze(-1))
                    values_denorm = self.value_normalizer.denormalize(values_pred.unsqueeze(-1))
                    value_loss = nn.functional.mse_loss(values_denorm, ret_b.unsqueeze(-1))
                    self.value_normalizer.update(ret_b.unsqueeze(-1))
                else:
                    value_loss = nn.functional.mse_loss(values_pred, ret_b)

                # Update actor
                self.actor_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy.mean()).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
        }
        self.metrics.add(metrics, global_step)

        for k, v in metrics.items():
            writer.add_scalar(f"mappo/{k}", v, global_step)

        return metrics

    def _maybe_reshape(self, obs_t: torch.Tensor) -> torch.Tensor:
        obs_shape = getattr(self, 'obs_shape', None)
        if obs_shape is not None:
            obs_t = obs_t.reshape(-1, *obs_shape)
        return obs_t

    def get_actions(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = self._maybe_reshape(obs_t)
        with torch.no_grad():
            if deterministic and not self.actor.use_rnn:
                if hasattr(self.actor, 'get_logits'):
                    logits = self.actor.get_logits(obs_t)
                else:
                    logits = self.actor.mlp(obs_t)
                actions = logits.argmax(dim=-1)
                log_probs = torch.zeros_like(actions, dtype=torch.float32)
            else:
                actions, log_probs, _ = self.actor(obs_t)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])


class PopArt:
    """PopArt value normalization (from MAPPO paper)."""

    def __init__(self, output_dim: int, beta: float = 0.9999):
        self.register = None  # set by first update
        self.mu = torch.zeros(output_dim)
        self.sigma = torch.ones(output_dim)
        self.beta = beta

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu.to(x.device)) / (self.sigma.to(x.device).clamp(min=1e-4))

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigma.to(x.device) + self.mu.to(x.device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(0).cpu()
        batch_std = x.std(0).clamp(min=1e-4).cpu()
        self.mu = self.beta * self.mu + (1 - self.beta) * batch_mean
        self.sigma = self.beta * self.sigma + (1 - self.beta) * batch_std
