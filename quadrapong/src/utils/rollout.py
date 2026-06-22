"""Rollout collection for on-policy algorithms."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from src.envs.quadrapong_env import QuadrapongWrapper
from src.utils.buffer import OnPolicyBuffer


class RolloutCollector:
    """Collects rollout data by running the policy in the environment."""

    def __init__(
        self,
        env: QuadrapongWrapper,
        agent_actors: Dict[str, torch.nn.Module],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_rnn: bool = False,
    ):
        self.env = env
        self.agent_actors = agent_actors
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        self.obs_dim = env.obs_dim
        self.use_rnn = use_rnn

        self.obs_buf = {a: [] for a in self.agents}
        self.rnn_states = {a: None for a in self.agents}

    @torch.no_grad()
    def collect(
        self,
        num_steps: int,
        critic: torch.nn.Module,
        deterministic: bool = False,
    ) -> Tuple[OnPolicyBuffer, Dict]:
        """Collect num_steps of experience.

        Args:
            num_steps: number of steps to collect
            critic: value function for computing values
            deterministic: if True, use greedy actions

        Returns:
            buffer: filled OnPolicyBuffer
            info: statistics dict
        """
        buffer = OnPolicyBuffer(
            self.num_agents, self.obs_dim, num_steps, self.gamma, self.gae_lambda
        )

        if len(self.obs_buf[self.agents[0]]) == 0:
            obs, _ = self.env.reset()
            for a in self.agents:
                self.obs_buf[a] = obs[a]
            self.rnn_states = {a: None for a in self.agents}
        else:
            obs = {a: self.obs_buf[a] for a in self.agents}

        total_rewards = np.zeros(self.num_agents)
        info = {"episode_rewards": np.zeros(self.num_agents), "episodes_done": 0}

        for step in range(num_steps):
            # Build batch for policy
            obs_batch = np.stack([obs[a] for a in self.agents])  # (num_agents, obs_dim)
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

            # Get actions and values
            actions, log_probs, values_tensor = self._get_actions_and_values(
                obs_tensor, critic, deterministic
            )

            values = values_tensor.cpu().numpy()
            actions_np = actions.cpu().numpy()
            action_log_probs_np = log_probs.cpu().numpy()

            # Step environment
            action_dict = {a: int(actions_np[i]) for i, a in enumerate(self.agents)}
            next_obs, rewards, terms, truncs, _ = self.env.step(action_dict)

            # Process
            reward_arr = np.array([rewards.get(a, 0) for a in self.agents], dtype=np.float32)
            done_arr = np.array(
                [float(terms.get(a, False) or truncs.get(a, False)) for a in self.agents],
                dtype=np.float32,
            )
            total_rewards += reward_arr

            # Build global state (concatenation of all observations)
            global_state = obs_batch.reshape(-1)

            buffer.insert(
                obs=obs_batch,
                actions=actions_np,
                action_log_probs=action_log_probs_np,
                rewards=reward_arr,
                values=values,
                dones=done_arr,
                global_state=global_state,
            )

            # Update obs buffer
            for i, a in enumerate(self.agents):
                if a in next_obs:
                    obs[a] = next_obs[a]
                self.obs_buf[a] = obs[a]

            # Handle episode termination
            if np.any(done_arr):
                info["episode_rewards"] += total_rewards
                info["episodes_done"] += 1
                obs, _ = self.env.reset()
                for a in self.agents:
                    self.obs_buf[a] = obs[a]
                total_rewards = np.zeros(self.num_agents)
                self.rnn_states = {a: None for a in self.agents}

        return buffer, info

    def _get_actions_and_values(
        self, obs: torch.Tensor, critic: torch.nn.Module, deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions, log_probs, and values from policy and critic."""
        # For parameter-shared policies, use a single actor
        agent_actor = self.agent_actors["shared"]
        actions, log_probs, _ = agent_actor(obs)
        values = critic(obs)
        return actions, log_probs, values
