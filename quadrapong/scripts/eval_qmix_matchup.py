"""QMIX vs IPPO, QMIX vs MAPPO head-to-head evaluation.

QMIX was trained controlling Team 1 vs Random Team 2.
Each matchup tests both side assignments.

Usage:
    python scripts/eval_qmix_matchup.py [--episodes 100] [--device cuda:0]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path

from src.envs.quadrapong_env import QuadrapongWrapper
from src.algos.ippo import IPPOTrainer
from src.algos.mappo import MAPPOTrainer
from src.utils.networks import FFQNetwork


def load_ippo(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor_state = ckpt["actor"]
    obs_dim = actor_state["mlp.0.weight"].shape[1]
    action_dim = actor_state["mlp.4.weight"].shape[0]
    hidden_dims = [actor_state["mlp.0.weight"].shape[0],
                   actor_state["mlp.2.weight"].shape[0]]
    trainer = IPPOTrainer(obs_dim=obs_dim, action_dim=action_dim, num_agents=2,
                          hidden_dims=hidden_dims, device=device)
    trainer.actor.load_state_dict(actor_state)
    trainer.actor.eval()
    return trainer.actor


def load_mappo(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor_state = ckpt["actor"]
    obs_dim = actor_state["mlp.0.weight"].shape[1]
    action_dim = actor_state["mlp.4.weight"].shape[0]
    hidden_dims = [actor_state["mlp.0.weight"].shape[0],
                   actor_state["mlp.2.weight"].shape[0]]
    global_obs_dim = obs_dim * 4
    trainer = MAPPOTrainer(obs_dim=obs_dim, global_obs_dim=global_obs_dim,
                           action_dim=action_dim, num_agents=2,
                           hidden_dims=hidden_dims, device=device)
    trainer.actor.load_state_dict(actor_state)
    trainer.actor.eval()
    return trainer.actor


def load_qmix(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent_q_state = ckpt["agent_q"]
    obs_dim = agent_q_state["mlp.0.weight"].shape[1]
    # FFQNetwork: mlp.0 = Linear(in, h), mlp.2 = Linear(h, out) — no mlp.4
    last_key = sorted(agent_q_state.keys())[-1]  # e.g. "mlp.2.weight"
    action_dim = agent_q_state[last_key].shape[0]
    hidden_dims = [agent_q_state["mlp.0.weight"].shape[0]]
    q_net = FFQNetwork(obs_dim, action_dim, hidden_dims).to(device)
    q_net.load_state_dict(agent_q_state)
    q_net.eval()
    return q_net


def get_ppo_actions(actor, obs_t, indices, device):
    """Get deterministic actions from PPO actor for given agent indices."""
    with torch.no_grad():
        logits = actor.mlp(obs_t[indices])
        return logits.argmax(dim=-1).cpu().numpy()


def get_qmix_actions(q_net, obs_t, indices, device):
    """Get greedy actions from QMIX agent Q-network."""
    with torch.no_grad():
        q_values = q_net(obs_t[indices])
        return q_values.argmax(dim=-1).cpu().numpy()


def run_matchup(env, team1_actor, team2_actor, num_episodes, device,
                team1_name, team2_name, team1_type, team2_type, max_steps=5000):
    """Run head-to-head matchup between two policies.

    team1_type, team2_type: "ppo" or "qmix" — determines action selection function.
    """
    agents = env.possible_agents
    team1_idx = [0, 2]  # first_0, third_0
    team2_idx = [1, 3]  # second_0, fourth_0

    wins = {team1_name: 0, team2_name: 0, "draw": 0}
    t1_rewards, t2_rewards = [], []
    ep_lens = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_r = {a: 0.0 for a in agents}

        while not done:
            obs_batch = np.stack([obs[a] for a in agents])
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            # Team 1 actions
            if team1_type == "ppo":
                actions_t1 = get_ppo_actions(team1_actor, obs_t, team1_idx, device)
            else:
                actions_t1 = get_qmix_actions(team1_actor, obs_t, team1_idx, device)

            # Team 2 actions
            if team2_type == "ppo":
                actions_t2 = get_ppo_actions(team2_actor, obs_t, team2_idx, device)
            else:
                actions_t2 = get_qmix_actions(team2_actor, obs_t, team2_idx, device)

            actions = np.zeros(4, dtype=int)
            actions[team1_idx] = actions_t1
            actions[team2_idx] = actions_t2

            action_dict = {a: int(actions[i]) for i, a in enumerate(agents)}
            obs, rewards, terms, truncs, _ = env.step(action_dict)

            for a in agents:
                ep_r[a] += rewards.get(a, 0)

            done = any(terms.values()) or any(truncs.values())
            step += 1
            if step >= max_steps:
                done = True

        t1_total = ep_r["first_0"] + ep_r["third_0"]
        t2_total = ep_r["second_0"] + ep_r["fourth_0"]
        t1_rewards.append(t1_total)
        t2_rewards.append(t2_total)
        ep_lens.append(step)

        if t1_total > t2_total:
            wins[team1_name] += 1
        elif t2_total > t1_total:
            wins[team2_name] += 1
        else:
            wins["draw"] += 1

    n = num_episodes
    return {
        "team1_winrate": wins[team1_name] / n,
        "team2_winrate": wins[team2_name] / n,
        "draw_rate": wins["draw"] / n,
        "avg_reward_team1": np.mean(t1_rewards),
        "avg_reward_team2": np.mean(t2_rewards),
        "avg_ep_len": np.mean(ep_lens),
        "wins": wins,
    }


def print_result(label, r):
    print(f"  {label}: T1_WR={r['team1_winrate']:.2%} T2_WR={r['team2_winrate']:.2%} "
          f"Draw={r['draw_rate']:.2%} T1_r={r['avg_reward_team1']:.1f} "
          f"T2_r={r['avg_reward_team2']:.1f} Ep_len={r['avg_ep_len']:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ippo_ckpt", type=str, default="checkpoints/ippo/ippo_final.pt")
    parser.add_argument("--mappo_ckpt", type=str, default="checkpoints/mappo/mappo_final.pt")
    parser.add_argument("--qmix_ckpt", type=str, default="checkpoints/qmix/qmix_final.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Episodes: {args.episodes}")

    print("Loading models...")
    ippo_actor = load_ippo(args.ippo_ckpt, device)
    mappo_actor = load_mappo(args.mappo_ckpt, device)
    qmix_qnet = load_qmix(args.qmix_ckpt, device)
    print("Done.")

    env = QuadrapongWrapper(obs_type="ram", max_cycles=100000)
    results = {}

    # === QMIX vs IPPO ===
    print("\n" + "=" * 60)
    print("QMIX vs IPPO")
    print("=" * 60)

    # QMIX (T1) vs IPPO (T2)
    print("\n--- QMIX (T1) vs IPPO (T2) ---")
    r = run_matchup(env, qmix_qnet, ippo_actor, args.episodes, device,
                    "QMIX", "IPPO", "qmix", "ppo")
    results["QMIX_T1_vs_IPPO_T2"] = r
    print_result("QMIX(T1) vs IPPO(T2)", r)

    # IPPO (T1) vs QMIX (T2) — swapped
    print("\n--- IPPO (T1) vs QMIX (T2) ---")
    r = run_matchup(env, ippo_actor, qmix_qnet, args.episodes, device,
                    "IPPO", "QMIX", "ppo", "qmix")
    results["IPPO_T1_vs_QMIX_T2"] = r
    print_result("IPPO(T1) vs QMIX(T2)", r)

    # === QMIX vs MAPPO ===
    print("\n" + "=" * 60)
    print("QMIX vs MAPPO")
    print("=" * 60)

    # QMIX (T1) vs MAPPO (T2)
    print("\n--- QMIX (T1) vs MAPPO (T2) ---")
    r = run_matchup(env, qmix_qnet, mappo_actor, args.episodes, device,
                    "QMIX", "MAPPO", "qmix", "ppo")
    results["QMIX_T1_vs_MAPPO_T2"] = r
    print_result("QMIX(T1) vs MAPPO(T2)", r)

    # MAPPO (T1) vs QMIX (T2) — swapped
    print("\n--- MAPPO (T1) vs QMIX (T2) ---")
    r = run_matchup(env, mappo_actor, qmix_qnet, args.episodes, device,
                    "MAPPO", "QMIX", "ppo", "qmix")
    results["MAPPO_T1_vs_QMIX_T2"] = r
    print_result("MAPPO(T1) vs QMIX(T2)", r)

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, r in results.items():
        print(f"  {k}: T1_WR={r['team1_winrate']:.2%} T2_WR={r['team2_winrate']:.2%} "
              f"Draw={r['draw_rate']:.2%}")

    # Save
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "qmix_matchup_results.txt"
    with open(out_path, "w") as f:
        f.write("QMIX vs IPPO/MAPPO Head-to-Head Results\n")
        f.write(f"Episodes per matchup: {args.episodes}\n\n")
        for k, r in results.items():
            f.write(f"{k}:\n")
            f.write(f"  Team1 WR: {r['team1_winrate']:.3f}, Team2 WR: {r['team2_winrate']:.3f}, Draw: {r['draw_rate']:.3f}\n")
            f.write(f"  Avg Reward: T1={r['avg_reward_team1']:.1f}, T2={r['avg_reward_team2']:.1f}\n")
            f.write(f"  Avg Ep Len: {r['avg_ep_len']:.0f}\n\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
