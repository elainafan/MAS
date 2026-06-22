"""Experience buffers for on-policy and off-policy RL algorithms."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Generator


class OnPolicyBuffer:
    """Rollout buffer for on-policy algorithms (IPPO, MAPPO).

    Stores complete trajectories and computes GAE advantages.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        max_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        global_obs_dim: int = None,
    ):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.global_obs_dim = global_obs_dim if global_obs_dim is not None else obs_dim * num_agents

        self.obs = np.zeros((max_steps, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_steps, num_agents), dtype=np.int64)
        self.action_log_probs = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.rewards = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.values = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.masks = np.ones((max_steps, num_agents), dtype=np.float32)

        self.global_state = np.zeros((max_steps, self.global_obs_dim), dtype=np.float32)

        self.step = 0

    def insert(
        self,
        obs: np.ndarray,        # (num_agents, obs_dim)
        actions: np.ndarray,    # (num_agents,)
        action_log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        masks: Optional[np.ndarray] = None,
        global_state: Optional[np.ndarray] = None,
    ):
        idx = self.step
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.action_log_probs[idx] = action_log_probs
        self.rewards[idx] = rewards
        self.values[idx] = values
        self.dones[idx] = dones
        if masks is not None:
            self.masks[idx] = masks
        if global_state is not None:
            self.global_state[idx] = global_state
        self.step += 1

    def compute_gae(
        self, next_value: np.ndarray, next_done: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        Returns:
            advantages: (max_steps, num_agents)
            returns: (max_steps, num_agents)
        """
        advantages = np.zeros((self.max_steps, self.num_agents), dtype=np.float32)
        gae = np.zeros(self.num_agents, dtype=np.float32)

        for t in reversed(range(self.step)):
            next_val = next_value if t == self.step - 1 else self.values[t + 1]
            next_d = next_done if t == self.step - 1 else self.dones[t]
            delta = self.rewards[t] + self.gamma * next_val * (1 - next_d) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_d) * gae
            advantages[t] = gae

        returns = advantages + self.values[: self.step]
        return advantages, returns

    def get_training_data(self) -> Dict[str, np.ndarray]:
        """Get all collected data as a dict."""
        s = self.step
        return {
            "obs": self.obs[:s],
            "actions": self.actions[:s],
            "action_log_probs": self.action_log_probs[:s],
            "rewards": self.rewards[:s],
            "values": self.values[:s],
            "dones": self.dones[:s],
            "masks": self.masks[:s],
            "global_state": self.global_state[:s],
        }

    def clear(self):
        self.step = 0


class ReplayBuffer:
    """Prioritized replay buffer for off-policy algorithms (QMIX).

    Transitions with non-zero reward (goals) get higher sampling priority,
    combating the sparse-reward bootstrap collapse problem.
    """

    def __init__(self, capacity: int, obs_dim: int, state_dim: int, num_agents: int, rnn_hidden: int = 0,
                 alpha: float = 0.0, beta_start: float = 0.4, beta_steps: int = 100000):
        self.capacity = capacity
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.rnn_hidden = rnn_hidden

        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.rnn_states = np.zeros((capacity, num_agents, rnn_hidden), dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.idx = 0
        self.size = 0
        self.max_prio = 1.0

    def push(self, obs, state, actions, rewards, next_obs, next_state, done, rnn_states=None):
        idx = self.idx % self.capacity
        self.obs[idx] = obs
        self.state[idx] = state
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.next_state[idx] = next_state
        self.dones[idx] = float(done)
        if rnn_states is not None:
            self.rnn_states[idx] = rnn_states
        self.priorities[idx] = self.max_prio  # new transitions get max priority
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict:
        """Sample batch with proportional prioritization."""
        if self.size < batch_size:
            return None

        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)

        # Importance sampling weights
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.size / self.beta_steps)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return {
            "obs": self.obs[indices],
            "state": self.state[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "next_state": self.next_state[indices],
            "dones": self.dones[indices],
            "rnn_states": self.rnn_states[indices],
            "indices": indices,
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        new_prio = np.abs(td_errors) + 1e-6
        self.max_prio = max(self.max_prio, new_prio.max())
        self.priorities[indices] = new_prio

    def __len__(self):
        return self.size
