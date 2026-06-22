"""Quadrapong environment wrapper.

Provides a unified interface over the PettingZoo Quadrapong environment,
supporting both RAM and pixel observations with frame stacking and resize.
"""
import numpy as np
import cv2
import gymnasium as gym
from typing import Dict, Tuple, Optional, List
from collections import deque
from pettingzoo.atari import quadrapong_v4
from gymnasium import spaces


class QuadrapongWrapper:
    """Wraps PettingZoo Quadrapong into a unified training interface.

    Args:
        obs_type: 'ram' | 'rgb_image' | 'grayscale_image'
        max_cycles: max steps per episode
        frame_stack: number of frames to stack (only for pixel obs)
        resize_dim: resize pixel obs to (resize_dim, resize_dim), None = no resize
        render_mode: None | 'human' | 'rgb_array'
    """

    def __init__(
        self,
        obs_type: str = "ram",
        max_cycles: int = 100000,
        frame_stack: int = 4,
        resize_dim: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        self.obs_type = obs_type
        self.max_cycles = max_cycles
        self.frame_stack = frame_stack
        self.resize_dim = resize_dim
        self.render_mode = render_mode
        self._env = None
        self._frame_buffers: Optional[Dict[str, deque]] = None

        self.possible_agents = ["first_0", "second_0", "third_0", "fourth_0"]
        self.team_1 = ["first_0", "third_0"]
        self.team_2 = ["second_0", "fourth_0"]
        self.num_agents = 4

        self._init_spaces()

    def _init_spaces(self):
        """Initialize action and observation spaces before env creation."""
        tmp_env = quadrapong_v4.parallel_env(
            obs_type=self.obs_type, max_cycles=self.max_cycles
        )
        self.action_space = tmp_env.action_space(self.possible_agents[0])
        raw_obs_space = tmp_env.observation_space(self.possible_agents[0])
        tmp_env.close()

        if self.obs_type == "ram":
            self.obs_dim = raw_obs_space.shape[0]  # 128
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
            )
            self.use_cnn = False
        else:
            h, w, c = raw_obs_space.shape
            if self.resize_dim is not None:
                h = w = self.resize_dim
            self.obs_shape = (c * self.frame_stack, h, w)  # (C*stack, H, W)
            self.obs_dim = self.obs_shape[0] * self.obs_shape[1] * self.obs_shape[2]  # flattened pixels
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=self.obs_shape, dtype=np.float32
            )
            self.use_cnn = True

    def _maybe_init_frame_buffers(self):
        if self._frame_buffers is None and not self.obs_type == "ram":
            self._frame_buffers = {
                a: deque(maxlen=self.frame_stack) for a in self.possible_agents
            }

    def _stack_frames(self, agent: str, frame: np.ndarray) -> np.ndarray:
        """Stack consecutive frames for temporal information."""
        buf = self._frame_buffers[agent]
        if len(buf) == 0:
            for _ in range(self.frame_stack):
                buf.append(frame)
        else:
            buf.append(frame)
        stacked = np.concatenate(list(buf), axis=-1)  # HWC → stack on C
        return np.transpose(stacked, (2, 0, 1))  # CHW for PyTorch

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment. Returns (obs_dict, info_dict)."""
        if self._env is not None:
            self._env.close()

        self._env = quadrapong_v4.parallel_env(
            obs_type=self.obs_type,
            max_cycles=self.max_cycles,
            render_mode=self.render_mode,
        )
        obs, info = self._env.reset(seed=seed)
        if self._frame_buffers is not None:
            self._frame_buffers = None
        self._maybe_init_frame_buffers()

        if not self.obs_type == "ram":
            if self.resize_dim is not None:
                obs = {a: cv2.resize(o, (self.resize_dim, self.resize_dim)) for a, o in obs.items()}
                # cv2.resize drops the last dim for single-channel images
                if next(iter(obs.values())).ndim == 2:
                    obs = {a: o[..., np.newaxis] for a, o in obs.items()}
            obs = {a: self._stack_frames(a, o).astype(np.float32) / 255.0 for a, o in obs.items()}
        else:
            obs = {a: o.astype(np.float32) / 255.0 for a, o in obs.items()}

        return obs, info

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step."""
        obs, rewards, terms, truncs, info = self._env.step(actions)

        if not self.obs_type == "ram":
            if self.resize_dim is not None:
                obs = {a: cv2.resize(o, (self.resize_dim, self.resize_dim)) for a, o in obs.items() if a in obs}
                if obs and next(iter(obs.values())).ndim == 2:
                    obs = {a: o[..., np.newaxis] for a, o in obs.items()}
            obs = {a: self._stack_frames(a, o).astype(np.float32) / 255.0 for a, o in obs.items() if a in obs}
        else:
            obs = {a: o.astype(np.float32) / 255.0 for a, o in obs.items() if a in obs}

        return obs, rewards, terms, truncs, info

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()

    def get_team_rewards(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Aggregate per-agent rewards into team rewards."""
        return {
            "team_1": rewards.get("first_0", 0) + rewards.get("third_0", 0),
            "team_2": rewards.get("second_0", 0) + rewards.get("fourth_0", 0),
        }

    def get_team_agents(self, team: int) -> List[str]:
        """Get agent names for a team (1 or 2)."""
        return self.team_1 if team == 1 else self.team_2
