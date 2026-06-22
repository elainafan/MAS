"""QMIX trainer for Quadrapong.

Off-policy CTDE with monotonic value factorization.
Uses feed-forward Q-networks (RAM is fully observable) and a mixing network with hypernetworks.
Supports both RAM and pixel (CNN) observations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from src.utils.networks import FFQNetwork, MixingNetwork, CNNFFQNetwork, CNNEncoder
from src.utils.buffer import ReplayBuffer
from src.utils.evaluator import MetricsTracker


class QMIXTrainer:
    """QMIX with feed-forward Q-networks and monotonic mixing network.

    Supports RAM (FFQNetwork) and pixel (CNNFFQNetwork + global state encoder).
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int = 4,
        hidden_dims: list = [128],
        mixing_embed: int = 32,
        hyper_hidden: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_capacity: int = 100000,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        grad_norm_clip: float = 10.0,
        double_q: bool = True,
        device: torch.device = torch.device("cpu"),
        agent_q_net: nn.Module = None,
        global_encoder: nn.Module = None,
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.grad_norm_clip = grad_norm_clip
        self.double_q = double_q
        self.device = device

        # Networks
        if agent_q_net is not None:
            self.agent_q = agent_q_net.to(device)
        else:
            self.agent_q = FFQNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.target_agent_q = deepcopy(self.agent_q)

        self.global_encoder = global_encoder.to(device) if global_encoder is not None else None
        mixer_state_dim = self.global_encoder.feat_dim if self.global_encoder is not None else state_dim
        self.mixer = MixingNetwork(num_agents, mixer_state_dim, mixing_embed, hyper_hidden).to(device)
        self.target_mixer = deepcopy(self.mixer)

        # Optimizer (RMSprop as in PyMARL, more stable than Adam for Q-learning)
        params = list(self.agent_q.parameters()) + list(self.mixer.parameters())
        if self.global_encoder is not None:
            params += list(self.global_encoder.parameters())
        self.optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)

        # Replay buffer (no RNN state needed for FF-Q)
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, state_dim, num_agents, rnn_hidden=0)

        self.metrics = MetricsTracker()
        self.train_steps = 0

    @property
    def epsilon(self):
        return max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end) * self.train_steps / self.epsilon_decay,
        )

    def _maybe_reshape(self, obs_t: torch.Tensor) -> torch.Tensor:
        obs_shape = getattr(self, 'obs_shape', None)
        if obs_shape is not None and obs_t.dim() == 2:
            obs_t = obs_t.reshape(obs_t.shape[0], *obs_shape)
        return obs_t

    @torch.no_grad()
    def get_actions(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get epsilon-greedy actions.

        Args:
            obs: (num_agents, obs_dim) for RAM or (num_agents, C, H, W) for pixel
            deterministic: if True, use greedy actions

        Returns:
            actions: (num_agents,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = self._maybe_reshape(obs_t)
        q_values = self.agent_q(obs_t).cpu().numpy()

        if deterministic or np.random.random() > self.epsilon:
            return q_values.argmax(axis=-1)
        else:
            return np.random.randint(0, self.action_dim, size=self.num_agents)

    def push_to_buffer(
        self, obs, state, actions, rewards, next_obs, next_state, done,
    ):
        self.buffer.push(obs, state, actions, rewards, next_obs, next_state, done)

    def update(
        self, writer: Optional[SummaryWriter], global_step: int
    ) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        self.train_steps += 1
        batch = self.buffer.sample(self.batch_size)

        obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        state_b = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        actions_b = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards_b = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_b = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        next_state_b = torch.tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        dones_b = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        # Reshape for CNN: buffer stores flattened pixels, network expects 4D
        obs_shape = getattr(self, 'obs_shape', None)
        if obs_shape is not None:
            C, H, W = obs_shape
            obs_b = obs_b.reshape(self.batch_size, self.num_agents, C, H, W)
            next_obs_b = next_obs_b.reshape(self.batch_size, self.num_agents, C, H, W)
            if self.global_encoder is not None:
                state_b = state_b.reshape(self.batch_size, self.num_agents * C, H, W)
                next_state_b = next_state_b.reshape(self.batch_size, self.num_agents * C, H, W)

        # Encode global state (CNN pixel → compact feature vector)
        if self.global_encoder is not None:
            state_b = self.global_encoder(state_b)  # gradients ON for online
            with torch.no_grad():
                next_state_b = self.global_encoder(next_state_b)  # no grad for target

        # Q(s, a) for all agents
        agent_qs = []
        target_agent_qs = []
        for i in range(self.num_agents):
            q_i = self.agent_q(obs_b[:, i])
            agent_qs.append(q_i)
            tq_i = self.target_agent_q(next_obs_b[:, i])
            target_agent_qs.append(tq_i)

        agent_qs = torch.stack(agent_qs, dim=1)  # (B, N, A)
        target_agent_qs = torch.stack(target_agent_qs, dim=1)

        # Gather chosen actions
        chosen_agent_qs = agent_qs.gather(2, actions_b.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixer(chosen_agent_qs, state_b)

        # Target Q with double Q
        if self.double_q:
            next_agent_qs_online = []
            for i in range(self.num_agents):
                q_i_next = self.agent_q(next_obs_b[:, i])
                next_agent_qs_online.append(q_i_next)
            next_agent_qs_online = torch.stack(next_agent_qs_online, dim=1)
            next_actions = next_agent_qs_online.argmax(dim=-1)
            target_chosen = target_agent_qs.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
        else:
            target_chosen = target_agent_qs.max(dim=-1)[0]

        with torch.no_grad():
            target_q_tot = self.target_mixer(target_chosen, next_state_b)
            team1_reward = rewards_b[:, 0].unsqueeze(-1)
            target = team1_reward + self.gamma * target_q_tot * (1 - dones_b)

        # PER-weighted MSE loss
        weights_b = torch.tensor(batch["weights"], dtype=torch.float32, device=self.device)
        td_error = q_tot - target
        loss = (weights_b * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent_q.parameters(), self.grad_norm_clip)
        nn.utils.clip_grad_norm_(self.mixer.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        # Update PER priorities
        self.buffer.update_priorities(batch["indices"], td_error.detach().cpu().numpy().squeeze(-1))

        # Hard target update every target_update_interval steps
        if self.train_steps % self.target_update_interval == 0:
            self._copy_update(self.agent_q, self.target_agent_q)
            self._copy_update(self.mixer, self.target_mixer)

        metrics = {
            "q_loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "target_mean": target.mean().item(),
            "epsilon": self.epsilon,
        }
        self.metrics.add(metrics, global_step)

        if writer is not None:
            for k, v in metrics.items():
                writer.add_scalar(f"qmix/{k}", v, global_step)

        return metrics

    def _copy_update(self, source: nn.Module, target: nn.Module):
        target.load_state_dict(source.state_dict())

    def save(self, path: str):
        state = {
            "agent_q": self.agent_q.state_dict(),
            "target_agent_q": self.target_agent_q.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.global_encoder is not None:
            state["global_encoder"] = self.global_encoder.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_q.load_state_dict(ckpt["agent_q"])
        self.target_agent_q.load_state_dict(ckpt["target_agent_q"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_mixer.load_state_dict(ckpt["target_mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.global_encoder is not None and "global_encoder" in ckpt:
            self.global_encoder.load_state_dict(ckpt["global_encoder"])
