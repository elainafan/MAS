"""Opponent Pool for self-play training.

Manages a FIFO queue of historical policy checkpoints and samples opponents
with weighted probabilities to break position-role symmetry in self-play.
"""

import random
import torch
import numpy as np
import copy
from typing import Optional, Tuple


class OpponentPool:
    """FIFO pool of historical checkpoints with weighted opponent sampling.

    Sampling weights:
    - 60% historical checkpoint (FIFO, capacity K)
    - 20% random policy (uniform random actions)
    - 20% current training policy

    Only actor state dicts are stored (no critic/optimizer). Models are
    loaded on-demand and cached to avoid repeated disk I/O.
    """

    def __init__(
        self,
        actor_class,
        actor_kwargs: dict,
        capacity: int = 5,
        history_weight: float = 0.6,
        random_weight: float = 0.2,
        current_weight: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        assert capacity > 0
        total = history_weight + random_weight + current_weight
        self.history_weight = history_weight / total
        self.random_weight = random_weight / total
        self.current_weight = current_weight / total

        self.capacity = capacity
        self.device = device
        self.actor_class = actor_class
        self.actor_kwargs = actor_kwargs

        # FIFO: list of (step, actor_state_dict)
        self._checkpoints: list = []
        # Cache: id(state_dict) -> loaded model (prevent reloading same weights)
        self._model_cache: dict = {}

        self._rng = random.Random()

    @property
    def size(self) -> int:
        return len(self._checkpoints)

    def add_checkpoint(self, actor_state_dict: dict, step: int):
        """Save current policy state to the pool (deep copy, FIFO)."""
        state_copy = {k: v.cpu().clone() for k, v in actor_state_dict.items()}
        if len(self._checkpoints) >= self.capacity:
            removed = self._checkpoints.pop(0)
            # Clean cache entry for removed checkpoint
            removed_id = id(removed[1])
            if removed_id in self._model_cache:
                del self._model_cache[removed_id]
        self._checkpoints.append((step, state_copy))

    def sample(
        self, current_actor: torch.nn.Module
    ) -> Tuple[Optional[torch.nn.Module], str]:
        """Sample an opponent for one episode.

        Args:
            current_actor: the training policy (used for 'current' type).

        Returns:
            (opponent_model_or_None, opponent_type_str)
            - model: loaded frozen model, or None for random.
            - type: 'history', 'random', or 'current'.
        """
        r = self._rng.random()
        if r < self.history_weight and self._checkpoints:
            return self._sample_history(), "history"
        elif r < self.history_weight + self.random_weight or not self._checkpoints:
            return None, "random"
        else:
            return current_actor, "current"

    def _sample_history(self) -> torch.nn.Module:
        """Pick a random checkpoint from pool, load and cache the model."""
        step, state_dict = self._rng.choice(self._checkpoints)

        # Use Python object id as cache key (state dicts are cloned, unique per add)
        sid = id(state_dict)
        if sid not in self._model_cache:
            model = self.actor_class(**self.actor_kwargs).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self._model_cache[sid] = model

        return self._model_cache[sid]

    def pool_stats(self) -> dict:
        return {
            "pool_size": self.size,
            "capacity": self.capacity,
            "checkpoint_steps": [s for s, _ in self._checkpoints],
        }


@torch.no_grad()
def get_opponent_actions(
    opponent_model: Optional[torch.nn.Module],
    obs: np.ndarray,
    act_dim: int,
    device: torch.device,
    obs_shape: Optional[tuple] = None,
) -> np.ndarray:
    """Get actions for Team 2 from the sampled opponent.

    Args:
        opponent_model: None for random, or a frozen model.
        obs: (num_team2, obs_dim) numpy array (flat for RAM, (C*H*W,) for CNN).
        act_dim: number of discrete actions.
        device: torch device.
        obs_shape: (C, H, W) tuple for CNN observations, None for RAM.

    Returns:
        actions: (num_team2,) numpy int array.
    """
    if opponent_model is None:
        return np.random.randint(0, act_dim, size=len(obs))

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    if obs_shape is not None:
        obs_t = obs_t.reshape(-1, *obs_shape)
    result = opponent_model(obs_t)
    actions = result[0] if isinstance(result, tuple) else result.argmax(dim=-1)
    return actions.cpu().numpy()
