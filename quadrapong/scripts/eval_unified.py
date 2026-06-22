"""Unified head-to-head evaluation for Quadrapong trained agents.

Supports IPPO, MAPPO, QMIX checkpoints with RAM or pixel (CNN) observations.
Auto-detects architecture and observation type from checkpoint weights.
Each matchup is tested with both side assignments.

Usage:
    # Two-way: A vs B, both sides
    python scripts/eval_unified.py A.pt B.pt --episodes 100 --device cuda:0

    # Three-way: all pairs among A, B, C
    python scripts/eval_unified.py A.pt B.pt C.pt --episodes 100

    # Specific matchup only
    python scripts/eval_unified.py --matchup ippo_final.pt qmix_final.pt
"""
import sys, os, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path

from src.utils.networks import (
    StochasticActor, CNNActor, FFQNetwork, CNNFFQNetwork,
)
from src.envs.quadrapong_env import QuadrapongWrapper


# ── checkpoint loading ──────────────────────────────────────────────

def _is_cnn(state_dict):
    """Check if state dict uses CNN encoder (pixel input)."""
    return any("encoder.conv" in k for k in state_dict.keys())


def _infer_mlp_dims(state_dict):
    """Infer obs_dim, hidden_dims, action_dim from an MLP state dict.
    Handles both 1-hidden-layer (QMIX) and 2-hidden-layer (IPPO/MAPPO)."""
    w0 = state_dict["mlp.0.weight"]
    obs_dim = w0.shape[1]
    # Detect number of hidden layers: mlp.4 exists → 2 hidden layers
    if "mlp.4.weight" in state_dict:
        hidden_dims = [w0.shape[0], state_dict["mlp.2.weight"].shape[0]]
        action_dim = state_dict["mlp.4.weight"].shape[0]
    else:
        hidden_dims = [w0.shape[0]]
        last_key = sorted(state_dict.keys())[-1]
        action_dim = state_dict[last_key].shape[0]
    return obs_dim, hidden_dims, action_dim


def load_ppo_actor(ckpt, device):
    """Load IPPO or MAPPO actor from loaded checkpoint.
    Returns (module, obs_type, env_kwargs)."""
    actor_state = ckpt["actor"]

    if _is_cnn(actor_state):
        input_channels = actor_state["encoder.conv.0.weight"].shape[1]
        action_dim = actor_state["mlp.2.weight"].shape[0]
        cnn_hidden = actor_state["mlp.0.weight"].shape[0]
        actor = CNNActor(input_channels, action_dim, cnn_hidden).to(device)
        actor.load_state_dict(actor_state)
        actor.eval()
        # frame_stack inferred from input_channels (grayscale C=frame_stack)
        return actor, "grayscale", {"obs_type": "grayscale_image",
                                     "frame_stack": input_channels, "resize_dim": 84}
    else:
        obs_dim, hidden_dims, action_dim = _infer_mlp_dims(actor_state)
        actor = StochasticActor(obs_dim, action_dim, hidden_dims).to(device)
        actor.load_state_dict(actor_state)
        actor.eval()
        return actor, "ram", {"obs_type": "ram"}


def load_qmix_qnet(ckpt, device):
    """Load QMIX agent Q-network from loaded checkpoint.
    Returns (module, obs_type, env_kwargs)."""
    agent_q_state = ckpt["agent_q"]

    if _is_cnn(agent_q_state):
        input_channels = agent_q_state["encoder.conv.0.weight"].shape[1]
        action_dim = agent_q_state["mlp.2.weight"].shape[0]
        cnn_hidden = agent_q_state["mlp.0.weight"].shape[0]
        q_net = CNNFFQNetwork(input_channels, action_dim, cnn_hidden).to(device)
        q_net.load_state_dict(agent_q_state)
        q_net.eval()
        return q_net, "grayscale", {"obs_type": "grayscale_image",
                                     "frame_stack": input_channels, "resize_dim": 84}
    else:
        obs_dim, hidden_dims, action_dim = _infer_mlp_dims(agent_q_state)
        q_net = FFQNetwork(obs_dim, action_dim, hidden_dims).to(device)
        q_net.load_state_dict(agent_q_state)
        q_net.eval()
        return q_net, "ram", {"obs_type": "ram"}


def load_model(path, device):
    """Auto-detect and load any checkpoint.
    Returns (module, model_type, obs_type, env_kwargs).
    Special: path="random" returns a random-policy placeholder.
    """
    if path == "random":
        return None, "random", "ram", {"obs_type": "ram"}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "agent_q" in ckpt:
        mod, obs_type, env_kwargs = load_qmix_qnet(ckpt, device)
        return mod, "qmix", obs_type, env_kwargs
    else:
        mod, obs_type, env_kwargs = load_ppo_actor(ckpt, device)
        return mod, "ppo", obs_type, env_kwargs


def get_algo_name(path):
    """Guess algorithm name from checkpoint path."""
    if path == "random":
        return "RANDOM"
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    parent_lower = parent.lower()
    # Use exact parent dir name
    known = {"ippo": "IPPO", "ippo_pixel": "IPPO_PIXEL", "ippo_pool": "IPPO_POOL",
             "mappo": "MAPPO", "mappo_pixel": "MAPPO_PIXEL", "mappo_pool": "MAPPO_POOL",
             "qmix": "QMIX", "qmix_pixel": "QMIX_PIXEL"}
    if parent_lower in known:
        return known[parent_lower]
    # Check for known algo prefixes in parent dir (longer prefixes first)
    for algo in ["ippo_pixel", "mappo_pixel", "qmix_pixel",
                 "ippo_pool", "mappo_pool", "ippo", "mappo", "qmix"]:
        if algo in parent_lower:
            return algo.upper()
    # Fallback: filename contains algo name
    fname = os.path.basename(path).lower()
    for algo in ["ippo_pixel", "mappo_pixel", "qmix_pixel",
                 "ippo_pool", "mappo_pool", "ippo", "mappo", "qmix"]:
        if algo in fname:
            return algo.upper()
    return parent.upper() if parent and parent != "." else os.path.basename(path)[:8].upper()


# ── action selection ─────────────────────────────────────────────────

def get_actions(module, model_type, obs_t, team_indices, device):
    """Get deterministic actions for specified agent indices."""
    if model_type == "random":
        return np.random.randint(0, 6, size=len(team_indices))
    with torch.no_grad():
        obs = obs_t[team_indices]
        if model_type == "qmix":
            q_values = module(obs)
            actions = q_values.argmax(dim=-1)
        else:
            if hasattr(module, 'get_logits'):
                logits = module.get_logits(obs)
            else:
                logits = module.mlp(obs)
            actions = logits.argmax(dim=-1)
    return actions.cpu().numpy()


# ── matchup runner ───────────────────────────────────────────────────

def _step_env(env, obs, actions, agents):
    """Step a single environment and return next obs, rewards, dones."""
    action_dict = {a: int(actions[i]) for i, a in enumerate(agents)}
    obs, rewards, terms, truncs, _ = env.step(action_dict)
    done = any(terms.values()) or any(truncs.values())
    return obs, rewards, done


def _obs_to_tensor(obs, agents, device):
    """Stack agent observations into a tensor."""
    obs_batch = np.stack([obs[a] for a in agents])
    return torch.tensor(obs_batch, dtype=torch.float32, device=device)


def _finalize_episode(ep_r, wins, t1_name, t2_name, t1_rewards, t2_rewards, ep_lens, term_types, step, truncated):
    t1_total = ep_r["first_0"] + ep_r["third_0"]
    t2_total = ep_r["second_0"] + ep_r["fourth_0"]
    t1_rewards.append(t1_total)
    t2_rewards.append(t2_total)
    ep_lens.append(step)
    if truncated:
        term_types["trunc_tie" if t1_total == 0 and t2_total == 0 else "trunc_lead"] += 1
    else:
        term_types["natural"] += 1
    if t1_total > t2_total:
        wins[t1_name] += 1
    elif t2_total > t1_total:
        wins[t2_name] += 1
    else:
        wins["draw"] += 1


def run_matchup(env, team1_mod, team2_mod, team1_type, team2_type,
                num_episodes, device, t1_name, t2_name, max_steps=5000):
    agents = env.possible_agents
    t1_idx = [0, 2]
    t2_idx = [1, 3]

    wins = {t1_name: 0, t2_name: 0, "draw": 0}
    t1_rewards, t2_rewards = [], []
    ep_lens, term_types, trajectories = [], {"natural": 0, "trunc_lead": 0, "trunc_tie": 0}, []

    for ep in range(num_episodes):
        seed = np.random.randint(0, 2**31 - 1)
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        ep_r = {a: 0.0 for a in agents}
        ep_traj = []  # (step, t1_score, t2_score)

        # Record initial
        ep_traj.append((0, 0.0, 0.0))

        while not done:
            obs_t = _obs_to_tensor(obs, agents, device)
            actions = np.zeros(4, dtype=int)
            actions[t1_idx] = get_actions(team1_mod, team1_type, obs_t, t1_idx, device)
            actions[t2_idx] = get_actions(team2_mod, team2_type, obs_t, t2_idx, device)
            obs, rewards, done = _step_env(env, obs, actions, agents)
            for a in agents:
                ep_r[a] += rewards.get(a, 0)
            step += 1
            # Record at 1000-step intervals
            if step % 1000 == 0:
                t1_s = ep_r["first_0"] + ep_r["third_0"]
                t2_s = ep_r["second_0"] + ep_r["fourth_0"]
                ep_traj.append((step, t1_s, t2_s))
            if step >= max_steps:
                done = True

        # Ensure final step is recorded
        t1_final = ep_r["first_0"] + ep_r["third_0"]
        t2_final = ep_r["second_0"] + ep_r["fourth_0"]
        if step % 1000 != 0:
            ep_traj.append((step, t1_final, t2_final))
        trajectories.append(ep_traj)

        truncated = step >= max_steps
        _finalize_episode(ep_r, wins, t1_name, t2_name, t1_rewards, t2_rewards, ep_lens, term_types, step, truncated)

    n = num_episodes
    return _make_result(t1_name, t2_name, wins, t1_rewards, t2_rewards, ep_lens, term_types, trajectories, n)


def run_cross_matchup(env_kwargs_a, env_kwargs_b,
                      team1_mod, team2_mod, team1_type, team2_type,
                      num_episodes, device, t1_name, t2_name, max_steps=5000):
    """Cross-observation matchup: each team uses a different obs type.

    Two ALE environments (one per obs type) run in lockstep with the same
    actions. ALE is deterministic (fixed frameskip), so both envs stay
    in sync producing identical rewards.
    """
    env_t1 = QuadrapongWrapper(max_cycles=100000, **env_kwargs_a)
    env_t2 = QuadrapongWrapper(max_cycles=100000, **env_kwargs_b)
    try:
        agents = env_t1.possible_agents
        t1_idx = [0, 2]
        t2_idx = [1, 3]

        wins = {t1_name: 0, t2_name: 0, "draw": 0}
        t1_rewards, t2_rewards = [], []
        ep_lens, term_types, trajectories = [], {"natural": 0, "trunc_lead": 0, "trunc_tie": 0}, []

        for ep in range(num_episodes):
            seed = np.random.randint(0, 2**31 - 1)
            obs_t1, _ = env_t1.reset(seed=seed)
            obs_t2, _ = env_t2.reset(seed=seed)
            done = False
            step = 0
            ep_r = {a: 0.0 for a in agents}
            ep_traj = [(0, 0.0, 0.0)]

            while not done:
                obs_t1_t = _obs_to_tensor(obs_t1, agents, device)
                obs_t2_t = _obs_to_tensor(obs_t2, agents, device)

                actions = np.zeros(4, dtype=int)
                actions[t1_idx] = get_actions(team1_mod, team1_type, obs_t1_t, t1_idx, device)
                actions[t2_idx] = get_actions(team2_mod, team2_type, obs_t2_t, t2_idx, device)

                obs_t1, rewards, done = _step_env(env_t1, obs_t1, actions, agents)
                obs_t2, rewards_t2, _ = _step_env(env_t2, obs_t2, actions, agents)
                for a in agents:
                    assert abs(rewards.get(a, 0) - rewards_t2.get(a, 0)) < 1e-6, \
                        f"Env divergence at ep {ep} step {step}: {a} reward mismatch"
                for a in agents:
                    ep_r[a] += rewards.get(a, 0)
                step += 1
                if step % 1000 == 0:
                    t1_s = ep_r["first_0"] + ep_r["third_0"]
                    t2_s = ep_r["second_0"] + ep_r["fourth_0"]
                    ep_traj.append((step, t1_s, t2_s))
                if step >= max_steps:
                    done = True

            t1_final = ep_r["first_0"] + ep_r["third_0"]
            t2_final = ep_r["second_0"] + ep_r["fourth_0"]
            if step % 1000 != 0:
                ep_traj.append((step, t1_final, t2_final))
            trajectories.append(ep_traj)

            truncated = step >= max_steps
            _finalize_episode(ep_r, wins, t1_name, t2_name, t1_rewards, t2_rewards, ep_lens, term_types, step, truncated)
    finally:
        env_t1.close()
        env_t2.close()

    n = num_episodes
    return _make_result(t1_name, t2_name, wins, t1_rewards, t2_rewards, ep_lens, term_types, trajectories, n)


def _make_result(t1_name, t2_name, wins, t1_rewards, t2_rewards, ep_lens, term_types, trajectories, n):
    return {
        "t1_name": t1_name, "t2_name": t2_name,
        "t1_wr": wins[t1_name] / n, "t2_wr": wins[t2_name] / n,
        "draw": wins["draw"] / n,
        "t1_r": np.mean(t1_rewards), "t2_r": np.mean(t2_rewards),
        "t1_r_std": np.std(t1_rewards), "t2_r_std": np.std(t2_rewards),
        "t1_min": np.min(t1_rewards), "t1_max": np.max(t1_rewards),
        "t2_min": np.min(t2_rewards), "t2_max": np.max(t2_rewards),
        "ep_len": np.mean(ep_lens),
        "term_natural": term_types["natural"],
        "term_trunc_lead": term_types["trunc_lead"],
        "term_trunc_tie": term_types["trunc_tie"],
        "trajectories": trajectories,  # list of lists of (step, t1_score, t2_score)
    }


def fmt(r):
    term = (f"end: natural={r['term_natural']} trunc(lead)={r['term_trunc_lead']} "
            f"trunc(tie)={r['term_trunc_tie']}")
    return (f"{r['t1_name']}(T1)={r['t1_wr']:.2%}  "
            f"{r['t2_name']}(T2)={r['t2_wr']:.2%}  Draw={r['draw']:.2%}  "
            f"ep_len={r['ep_len']:.0f} | {term}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified head-to-head eval")
    parser.add_argument("checkpoints", nargs="*",
                        help="One or more checkpoint paths")
    parser.add_argument("--matchup", nargs=2, metavar=("CKPT1", "CKPT2"),
                        help="Run specific matchup only")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="results/unified_matchup.txt")
    parser.add_argument("--max-steps", type=int, default=20000,
                        help="Max steps per episode before truncation (default: 20000)")
    parser.add_argument("--include-random", action="store_true",
                        help="Add random baseline agent to all matchups")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.matchup:
        pairs = [tuple(args.matchup)]
    elif len(args.checkpoints) >= 2:
        ckpts = list(args.checkpoints)
        if args.include_random:
            ckpts.append("random")
        pairs = [(ckpts[i], ckpts[j]) for i in range(len(ckpts))
                 for j in range(i + 1, len(ckpts))]
    else:
        print("Need at least 2 checkpoints or --matchup")
        sys.exit(1)

    print(f"Device: {device}, Episodes: {args.episodes}")
    print(f"Matchups: {len(pairs)}")

    model_cache = {}
    all_results = []

    for path_a, path_b in pairs:
        for p in [path_a, path_b]:
            if p not in model_cache:
                try:
                    mod, mtype, otype, ekw = load_model(p, device)
                    name = get_algo_name(p)
                    model_cache[p] = (mod, mtype, otype, ekw, name)
                except Exception as e:
                    print(f"\n❌ Failed to load {p}: {e}")
                    model_cache[p] = None

        entry_a = model_cache[path_a]
        entry_b = model_cache[path_b]
        if entry_a is None or entry_b is None:
            print(f"⚠️  Skipping {path_a} vs {path_b}: load failed")
            continue

        mod_a, type_a, obs_a, ekw_a, name_a = entry_a
        mod_b, type_b, obs_b, ekw_b, name_b = entry_b

        # Random agents can play with any obs type — match the other side
        if type_a == "random":
            effective_obs_a = obs_b
        else:
            effective_obs_a = obs_a
        if type_b == "random":
            effective_obs_b = effective_obs_a
        else:
            effective_obs_b = obs_b

        print(f"\n{'='*60}")
        if effective_obs_a != effective_obs_b:
            obs_tag = f"cross: {effective_obs_a} vs {effective_obs_b}"
        else:
            obs_tag = effective_obs_a
        print(f"  {name_a} vs {name_b}  ({obs_tag}, {type_a} vs {type_b})")
        print(f"{'='*60}")

        if effective_obs_a == effective_obs_b:
            # Use non-random agent's kwargs; if both random, use any
            env_kw = ekw_b if type_a == "random" else ekw_a
            env = QuadrapongWrapper(max_cycles=100000, **env_kw)
            try:
                r1 = run_matchup(env, mod_a, mod_b, type_a, type_b,
                                 args.episodes, device, name_a, name_b, args.max_steps)
                all_results.append(r1)
                print(f"  {fmt(r1)}")

                r2 = run_matchup(env, mod_b, mod_a, type_b, type_a,
                                 args.episodes, device, name_b, name_a, args.max_steps)
                all_results.append(r2)
                print(f"  {fmt(r2)}")
            finally:
                env.close()
        else:
            try:
                r1 = run_cross_matchup(ekw_a, ekw_b, mod_a, mod_b, type_a, type_b,
                                       args.episodes, device, name_a, name_b, args.max_steps)
                all_results.append(r1)
                print(f"  {fmt(r1)}")

                r2 = run_cross_matchup(ekw_b, ekw_a, mod_b, mod_a, type_b, type_a,
                                       args.episodes, device, name_b, name_a, args.max_steps)
                all_results.append(r2)
                print(f"  {fmt(r2)}")
            except Exception as e:
                print(f"  ❌ Cross-obs matchup failed: {e}")
                continue

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {fmt(r)}")

    # Save
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"Unified Head-to-Head Results\nEpisodes: {args.episodes}\n\n")
        for r in all_results:
            f.write(f"{r['t1_name']}(T1) vs {r['t2_name']}(T2):\n")
            f.write(f"  WR: {r['t1_wr']:.3f} vs {r['t2_wr']:.3f}, Draw: {r['draw']:.3f}\n")
            f.write(f"  Score: {r['t1_r']:.1f}±{r['t1_r_std']:.1f} [{r['t1_min']:.0f},{r['t1_max']:.0f}]"
                    f" vs {r['t2_r']:.1f}±{r['t2_r_std']:.1f} [{r['t2_min']:.0f},{r['t2_max']:.0f}], "
                    f"Ep_len: {r['ep_len']:.0f}\n")
            f.write(f"  End: natural={r['term_natural']} trunc(lead)={r['term_trunc_lead']} "
                    f"trunc(tie)={r['term_trunc_tie']}\n\n")
    print(f"\nSaved to {out}")

    # Save trajectory data for plotting
    traj_out = Path(str(out).replace(".txt", "_traj.npz"))
    traj_data = {"matchups": [], "episodes": args.episodes}
    for r in all_results:
        traj_data["matchups"].append({
            "t1_name": r["t1_name"], "t2_name": r["t2_name"],
            "t1_wr": r["t1_wr"], "t2_wr": r["t2_wr"],
            "term_natural": r["term_natural"], "term_trunc_lead": r["term_trunc_lead"],
            "term_trunc_tie": r["term_trunc_tie"],
            "t1_r": r["t1_r"], "t1_r_std": r["t1_r_std"],
            "t2_r": r["t2_r"], "t2_r_std": r["t2_r_std"],
            "trajectories": r["trajectories"],
        })
    np.savez_compressed(traj_out, data=traj_data)
    print(f"Trajectories saved to {traj_out}")


if __name__ == "__main__":
    main()
