"""Independent PPO (IPPO) trainer.

Each agent runs PPO independently with a shared actor network and
agent-local critics. No centralized information is used.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

from src.utils.networks import StochasticActor, Critic
from src.utils.buffer import OnPolicyBuffer
from src.utils.evaluator import evaluate, MetricsTracker


class IPPOTrainer:
    """Trainer for Independent PPO with parameter sharing.

    All agents share a single actor and critic network.
    No centralized information — each agent uses only its own observation.
    """

    def __init__(
        self,
        obs_dim: int,
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
        use_rnn: bool = False,
        rnn_hidden: int = 128,
        device: torch.device = torch.device("cpu"),
        actor: nn.Module = None,
        critic: nn.Module = None,
    ):
        self.obs_dim = obs_dim
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

        if actor is not None:
            self.actor = actor.to(device)
        else:
            self.actor = StochasticActor(
                obs_dim, action_dim, hidden_dims, use_rnn=use_rnn, rnn_hidden=rnn_hidden
            ).to(device)
        if critic is not None:
            self.critic = critic.to(device)
        else:
            self.critic = Critic(obs_dim, hidden_dims).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.metrics = MetricsTracker()

    def train_on_buffer(
        self, buffer: OnPolicyBuffer, writer: SummaryWriter, global_step: int
    ) -> Dict[str, float]:
        """Perform PPO update using collected rollout data.

        Args:
            buffer: filled OnPolicyBuffer
            writer: TensorBoard writer
            global_step: current training step

        Returns:
            training metrics dict
        """
        data = buffer.get_training_data()
        num_steps = buffer.step
        total_batch_size = num_steps * self.num_agents

        # Compute GAE
        with torch.no_grad():
            last_obs = torch.tensor(data["obs"][-1], dtype=torch.float32, device=self.device)
            last_value = self.critic(last_obs).cpu().numpy()
            last_done = data["dones"][-1]

        advantages, returns = buffer.compute_gae(last_value, last_done)

        # Flatten agent dimension into batch
        obs_flat = data["obs"].reshape(-1, self.obs_dim)
        actions_flat = data["actions"].reshape(-1)
        old_log_probs_flat = data["action_log_probs"].reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)

        # Normalize advantages
        adv_mean, adv_std = advantages_flat.mean(), advantages_flat.std()
        advantages_flat = (advantages_flat - adv_mean) / (adv_std + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # PPO update
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(total_batch_size)
            for start in range(0, total_batch_size, self.mini_batch_size):
                idx = indices[start : start + self.mini_batch_size]

                obs_b = torch.tensor(obs_flat[idx], dtype=torch.float32, device=self.device)
                obs_b = self._maybe_reshape(obs_b)
                act_b = torch.tensor(actions_flat[idx], dtype=torch.long, device=self.device)
                old_lp_b = torch.tensor(old_log_probs_flat[idx], dtype=torch.float32, device=self.device)
                adv_b = torch.tensor(advantages_flat[idx], dtype=torch.float32, device=self.device)
                ret_b = torch.tensor(returns_flat[idx], dtype=torch.float32, device=self.device)

                # Actor loss
                new_log_probs, entropy = self.actor.evaluate_actions(obs_b, None, act_b)
                ratio = torch.exp(new_log_probs - old_lp_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values = self.critic(obs_b).squeeze(-1)
                value_loss = nn.functional.mse_loss(values, ret_b)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

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

        # Log
        for k, v in metrics.items():
            writer.add_scalar(f"ippo/{k}", v, global_step)

        return metrics

    def _maybe_reshape(self, obs_t: torch.Tensor) -> torch.Tensor:
        obs_shape = getattr(self, 'obs_shape', None)
        if obs_shape is not None:
            obs_t = obs_t.reshape(-1, *obs_shape)
        return obs_t

    def get_actions(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get actions for all agents.

        Args:
            obs: (num_agents, obs_dim) numpy array
            deterministic: use argmax if True

        Returns:
            actions: (num_agents,) numpy array
            log_probs: (num_agents,) numpy array
        """
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

    def get_values(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = self._maybe_reshape(obs_t)
        with torch.no_grad():
            values = self.critic(obs_t).squeeze(-1)
        return values.cpu().numpy()

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
