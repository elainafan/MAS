"""Evaluation and video recording for Quadrapong."""

import numpy as np
import torch
import imageio
from typing import Dict, Optional
from collections import defaultdict
from pathlib import Path
from src.envs.quadrapong_env import QuadrapongWrapper


@torch.no_grad()
def evaluate(
    env: QuadrapongWrapper,
    agent_actors: Dict[str, torch.nn.Module],
    num_episodes: int = 10,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
    record_video: bool = False,
    video_path: Optional[str] = None,
    max_eval_steps: int = 5000,
    random_agent_indices: list = None,
) -> Dict:
    """Evaluate agent policies over multiple episodes.

    When record_video=True, creates a temporary rgb_array env to record
    the first episode, then continues evaluation on the original env.
    max_eval_steps caps each episode to prevent deterministic stall at max_cycles.
    random_agent_indices: list of agent indices to use random actions (e.g. [1,3] for team 2).
    """
    agents = env.possible_agents
    actor = agent_actors["shared"]
    actor.eval()

    team1_rewards = []
    team2_rewards = []
    episode_lengths = []
    team1_wins = 0
    team2_wins = 0
    draws = 0

    # For video: create a render-capable env for the first episode
    video_env = None
    if record_video:
        video_env = QuadrapongWrapper(
            obs_type=env.obs_type,
            max_cycles=env.max_cycles,
            frame_stack=env.frame_stack,
            resize_dim=env.resize_dim if hasattr(env, 'resize_dim') else None,
            render_mode="rgb_array",
        )

    for ep in range(num_episodes):
        is_recording = record_video and ep == 0 and video_env is not None
        active_env = video_env if is_recording else env
        video_frames = [] if is_recording else None

        obs, _ = active_env.reset()
        done = False
        ep_rewards = {a: 0.0 for a in agents}
        step = 0

        rnn_state_ev = None  # for DRQN-based agents (QMIX)

        while not done:
            obs_batch = np.stack([obs[a] for a in agents])
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            is_rnn = hasattr(actor, 'rnn')
            is_ppo = hasattr(actor, 'use_rnn')  # StochasticActor or CNNActor
            has_get_logits = hasattr(actor, 'get_logits')  # CNNActor
            if deterministic:
                if is_rnn:
                    if rnn_state_ev is None:
                        rnn_state_ev = torch.zeros(1, len(agents), actor.rnn_hidden, device=device)
                    q_values, rnn_state_ev = actor(obs_tensor, rnn_state_ev)
                    actions = q_values.argmax(dim=-1)
                elif has_get_logits:
                    logits = actor.get_logits(obs_tensor)
                    actions = logits.argmax(dim=-1)
                else:
                    logits = actor.mlp(obs_tensor)
                    actions = logits.argmax(dim=-1)
            else:
                if is_rnn:
                    if rnn_state_ev is None:
                        rnn_state_ev = torch.zeros(1, len(agents), actor.rnn_hidden, device=device)
                    q_values, rnn_state_ev = actor(obs_tensor, rnn_state_ev)
                    dist = torch.distributions.Categorical(logits=q_values)
                    actions = dist.sample()
                elif has_get_logits:
                    logits = actor.get_logits(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                elif is_ppo:
                    actions, _, _ = actor(obs_tensor)
                else:
                    q_values = actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=q_values)
                    actions = dist.sample()

            # Override random-agent actions if specified
            if random_agent_indices:
                for idx in random_agent_indices:
                    actions[idx] = np.random.randint(0, env.action_space.n)

            action_dict = {a: int(actions[i].item()) for i, a in enumerate(agents)}
            obs, rewards, terms, truncs, _ = active_env.step(action_dict)

            for a in agents:
                ep_rewards[a] += rewards.get(a, 0)

            done = any(terms.values()) or any(truncs.values())
            step += 1

            # Prevent deterministic stall (e.g., all agents predict NOOP)
            if step >= max_eval_steps:
                done = True

            if is_recording:
                frame = active_env.render()
                if frame is not None:
                    video_frames.append(frame)

        # Save video
        if is_recording and video_frames and video_path:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, video_frames, fps=30)

        t1_total = ep_rewards["first_0"] + ep_rewards["third_0"]
        t2_total = ep_rewards["second_0"] + ep_rewards["fourth_0"]

        team1_rewards.append(t1_total)
        team2_rewards.append(t2_total)
        episode_lengths.append(step)

        if t1_total > t2_total:
            team1_wins += 1
        elif t2_total > t1_total:
            team2_wins += 1
        else:
            draws += 1

    if video_env is not None:
        video_env.close()

    actor.train()

    return {
        "team_1_winrate": team1_wins / num_episodes,
        "team_2_winrate": team2_wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_reward_team1": np.mean(team1_rewards),
        "avg_reward_team2": np.mean(team2_rewards),
        "avg_episode_length": np.mean(episode_lengths),
        "team1_rewards": team1_rewards,
        "team2_rewards": team2_rewards,
        "episode_lengths": episode_lengths,
    }


class MetricsTracker:
    """Tracks and aggregates training metrics."""

    def __init__(self):
        self.history = defaultdict(list)

    def add(self, metrics: Dict[str, float], step: int):
        for key, value in metrics.items():
            self.history[key].append((step, value))

    def get_recent(self, key: str, n: int = 100) -> float:
        vals = self.history.get(key, [])
        if not vals:
            return 0.0
        return np.mean([v for _, v in vals[-n:]])

    def to_dict(self) -> Dict[str, list]:
        return dict(self.history)
