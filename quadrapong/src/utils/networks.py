"""Neural network modules for Quadrapong RL agents.

Includes:
- MLP policies (for RAM observations)
- CNN policies (for pixel observations)
- GRU-based recurrent policies
- QMIX mixing network and hypernetworks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from gymnasium import spaces


def make_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """Build an MLP with configurable activation and normalization."""
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation]
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(act_fn())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        out_act = {"tanh": nn.Tanh, "softmax": nn.Softmax(dim=-1)}[output_activation]
        layers.append(out_act)
    return nn.Sequential(*layers)


class StochasticActor(nn.Module):
    """Discrete action stochastic policy (for PPO).

    Outputs a categorical distribution over actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        use_rnn: bool = False,
        rnn_hidden: int = 128,
    ):
        super().__init__()
        self.use_rnn = use_rnn
        self.action_dim = action_dim

        if use_rnn:
            self.rnn = nn.GRU(obs_dim, rnn_hidden, batch_first=True)
            self.rnn_hidden = rnn_hidden
            mlp_input = rnn_hidden
        else:
            mlp_input = obs_dim

        self.mlp = make_mlp(mlp_input, hidden_dims, action_dim, activation="tanh")

    def forward(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
            rnn_state: (1, batch, rnn_hidden) if use_rnn
            mask: (batch,) or (batch, seq_len) reset flag

        Returns:
            actions: sampled actions
            log_probs: log probabilities
            new_rnn_state: updated RNN state
        """
        if self.use_rnn:
            if obs.dim() == 2:
                obs_seq = obs.unsqueeze(1)  # (B, 1, obs_dim)
            else:
                obs_seq = obs
            rnn_out, new_rnn_state = self.rnn(obs_seq, rnn_state)
            features = rnn_out.squeeze(1) if obs.dim() == 2 else rnn_out
        else:
            features = obs
            new_rnn_state = None

        logits = self.mlp(features)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, new_rnn_state

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        rnn_state: Optional[torch.Tensor],
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate given actions, returning log probs and entropy."""
        if self.use_rnn:
            if obs.dim() == 2:
                obs_seq = obs.unsqueeze(1)
            else:
                obs_seq = obs
            rnn_out, _ = self.rnn(obs_seq, rnn_state)
            features = rnn_out.squeeze(1) if obs.dim() == 2 else rnn_out
        else:
            features = obs

        logits = self.mlp(features)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy

    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Get action distribution for given observation."""
        if self.use_rnn:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            rnn_out, _ = self.rnn(obs)
            features = rnn_out.squeeze(1)
        else:
            features = obs
        logits = self.mlp(features)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    """Value function approximator."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list = [256, 256],
        use_popart: bool = False,
    ):
        super().__init__()
        self.mlp = make_mlp(obs_dim, hidden_dims, 1, activation="relu")
        self.use_popart = use_popart

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimate V(s). Shape: (batch, 1)."""
        return self.mlp(obs)


class CentralizedCritic(nn.Module):
    """Centralized critic for MAPPO — takes global state + agent observation."""

    def __init__(
        self,
        global_obs_dim: int,
        local_obs_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        input_dim = global_obs_dim + local_obs_dim
        self.mlp = make_mlp(input_dim, hidden_dims, 1, activation="relu")

    def forward(self, global_obs: torch.Tensor, local_obs: torch.Tensor) -> torch.Tensor:
        """V(s, o_i) — centralized value conditioned on global + local info."""
        x = torch.cat([global_obs, local_obs], dim=-1)
        return self.mlp(x)


class CNNEncoder(nn.Module):
    """Nature DQN CNN encoder for pixel observations.

    Supports grayscale (frame_stack × 1 channel) and RGB (frame_stack × 3 channels).
    Architecture: Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,1) → Flatten.
    """

    def __init__(self, input_channels: int, input_h: int = 84, input_w: int = 84):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_h, input_w)
            self.feat_dim = self.conv(dummy).shape[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.conv(obs)


class CNNActor(nn.Module):
    """CNN-based stochastic actor for pixel observations (IPPO/MAPPO).

    Compatible with evaluator via use_rnn=False and get_logits().
    Matches StochasticActor's interface: forward → (actions, log_probs, None).
    """

    def __init__(self, input_channels: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.use_rnn = False
        self.action_dim = action_dim
        self.encoder = CNNEncoder(input_channels)
        self.mlp = make_mlp(self.encoder.feat_dim, [hidden_dim], action_dim, activation="relu")

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        logits = self.get_logits(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), None

    def evaluate_actions(self, obs: torch.Tensor, rnn_state, actions: torch.Tensor):
        logits = self.get_logits(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def get_action_distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.get_logits(obs))


class CNNCritic(nn.Module):
    """CNN-based local critic (IPPO) — value from single agent's pixel obs."""

    def __init__(self, input_channels: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = CNNEncoder(input_channels)
        self.mlp = make_mlp(self.encoder.feat_dim, [hidden_dim], 1, activation="relu")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs))


class CNNCentralizedCritic(nn.Module):
    """CNN-based centralized critic (MAPPO).

    Encodes global pixel state (all agents' obs stacked in channel dim) + local
    agent obs, concatenates features, outputs value.
    """

    def __init__(self, input_channels: int, num_agents: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.encoder = CNNEncoder(input_channels)  # shared encoder
        self.num_agents = num_agents
        # Global: all agents stacked → (input_channels * num_agents, H, W)
        self.global_encoder = CNNEncoder(input_channels * num_agents)
        feat_in = self.encoder.feat_dim + self.global_encoder.feat_dim
        self.mlp = make_mlp(feat_in, [hidden_dim], 1, activation="relu")

    def forward(self, global_obs: torch.Tensor, local_obs: torch.Tensor) -> torch.Tensor:
        g_feat = self.global_encoder(global_obs)
        l_feat = self.encoder(local_obs)
        return self.mlp(torch.cat([g_feat, l_feat], dim=-1))


class CNNFFQNetwork(nn.Module):
    """CNN-based feed-forward Q-network for QMIX with pixel observations."""

    def __init__(self, input_channels: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = CNNEncoder(input_channels)
        self.mlp = make_mlp(self.encoder.feat_dim, [hidden_dim], action_dim, activation="relu")
        self.action_dim = action_dim

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.get_logits(obs)


# --- QMIX modules ---

class FFQNetwork(nn.Module):
    """Feed-forward Q-network for QMIX (fully observable RAM, no RNN needed)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [128]):
        super().__init__()
        self.mlp = make_mlp(obs_dim, hidden_dims, action_dim, activation="relu")
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)  # (batch, action_dim)


class DRQN(nn.Module):
    """Deep Recurrent Q-Network for QMIX agent (deprecated: use FFQNetwork)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rnn_hidden: int = 64,
        hidden_dims: list = [128],
    ):
        super().__init__()
        self.rnn = nn.GRU(obs_dim, rnn_hidden, batch_first=True)
        self.rnn_hidden = rnn_hidden
        self.mlp = make_mlp(rnn_hidden, hidden_dims, action_dim, activation="relu")
        self.action_dim = action_dim

    def forward(
        self, obs: torch.Tensor, rnn_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        rnn_out, new_state = self.rnn(obs, rnn_state)
        q_values = self.mlp(rnn_out.squeeze(1))
        return q_values, new_state


class MixingNetwork(nn.Module):
    """Monotonic mixing network for QMIX.

    Enforces ∂Q_tot/∂Q_i ≥ 0 via non-negative weights.
    Hypernetworks condition on global state to produce mixing weights.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hyper_hidden: int = 64,
        mixing_hidden: int = 32,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork for W1: state → (num_agents * embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, num_agents * embed_dim),
        )
        # Hypernetwork for b1: state → (1 * embed_dim)
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        # Hypernetwork for W2: state → (embed_dim * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        # Hypernetwork for b2: state → 1
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            agent_qs: (batch, num_agents) — per-agent Q values
            state: (batch, state_dim) — global state

        Returns:
            q_tot: (batch, 1) — total Q value
        """
        batch_size = agent_qs.shape[0]

        # Layer 1
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (B, 1, embed)

        # Layer 2
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_tot.squeeze(-1)  # (B, 1)
