"""Head-to-head evaluation: IPPO vs MAPPO, and each vs Random.

Usage:
    python scripts/eval_matchup.py [--episodes 100] [--device cuda:0]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path

from src.envs.quadrapong_env import QuadrapongWrapper
from src.algos.ippo import IPPOTrainer
from src.algos.mappo import MAPPOTrainer


def load_model(path, trainer_cls, device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Infer architecture from checkpoint
    actor_state = ckpt["actor"]
    # Find obs_dim from first layer weight
    obs_dim = actor_state["mlp.0.weight"].shape[1]
    action_dim = actor_state["mlp.4.weight"].shape[0]
    hidden_dims = [
        actor_state["mlp.0.weight"].shape[0],
        actor_state["mlp.2.weight"].shape[0],
    ]

    if trainer_cls == IPPOTrainer:
        trainer = IPPOTrainer(
            obs_dim=obs_dim, action_dim=action_dim, num_agents=2,
            hidden_dims=hidden_dims, device=device,
        )
    else:
        global_obs_dim = obs_dim * 4
        trainer = MAPPOTrainer(
            obs_dim=obs_dim, global_obs_dim=global_obs_dim, action_dim=action_dim,
            num_agents=2, hidden_dims=hidden_dims, device=device,
        )

    trainer.actor.load_state_dict(actor_state)
    trainer.actor.eval()
    return trainer


def run_matchup(env, team1_actor, team2_mode, num_episodes, device,
                team1_name="A", team2_name="B", max_steps=5000):
    """Run head-to-head matchup.

    team1_actor: nn.Module for team 1 (agents 0,2)
    team2_mode: "random" or an nn.Module for team 2 (agents 1,3)
    """
    agents = env.possible_agents
    team1_idx = [0, 2]
    team2_idx = [1, 3]

    wins = {team1_name: 0, team2_name: 0, "draw": 0}
    t1_rewards_all = []
    t2_rewards_all = []
    ep_lens = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_r = {a: 0.0 for a in agents}

        while not done:
            obs_batch = np.stack([obs[a] for a in agents])
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)

            # Team 1: use team1_actor
            with torch.no_grad():
                logits_t1 = team1_actor.mlp(obs_t[team1_idx])
                actions_t1 = logits_t1.argmax(dim=-1).cpu().numpy()

            # Team 2
            if team2_mode == "random":
                actions_t2 = np.random.randint(0, 6, size=2)
            else:
                with torch.no_grad():
                    obs_t2 = obs_t[team2_idx]
                    logits_t2 = team2_mode.mlp(obs_t2)
                    actions_t2 = logits_t2.argmax(dim=-1).cpu().numpy()

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
        t1_rewards_all.append(t1_total)
        t2_rewards_all.append(t2_total)
        ep_lens.append(step)

        if t1_total > t2_total:
            wins[team1_name] += 1
        elif t2_total > t1_total:
            wins[team2_name] += 1
        else:
            wins["draw"] += 1

    return {
        "team1_winrate": wins[team1_name] / num_episodes,
        "team2_winrate": wins[team2_name] / num_episodes,
        "draw_rate": wins["draw"] / num_episodes,
        "avg_reward_team1": np.mean(t1_rewards_all),
        "avg_reward_team2": np.mean(t2_rewards_all),
        "avg_ep_len": np.mean(ep_lens),
        "wins": wins,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ippo_ckpt", type=str, default="checkpoints/ippo/ippo_final.pt")
    parser.add_argument("--mappo_ckpt", type=str, default="checkpoints/mappo/mappo_final.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Episodes per matchup: {args.episodes}")

    # Load models
    print("Loading IPPO...")
    ippo = load_model(args.ippo_ckpt, IPPOTrainer, device)
    print("Loading MAPPO...")
    mappo = load_model(args.mappo_ckpt, MAPPOTrainer, device)

    env = QuadrapongWrapper(obs_type="ram", max_cycles=100000)

    results = {}

    # 1. IPPO vs Random
    print("\n=== IPPO (Team 1) vs Random (Team 2) ===")
    r = run_matchup(env, ippo.actor, "random", args.episodes, device, "IPPO", "Random")
    results["IPPO_vs_Random"] = r
    print(f"  IPPO WR: {r['team1_winrate']:.2%}, Random WR: {r['team2_winrate']:.2%}, "
          f"Draw: {r['draw_rate']:.2%}, T1_reward: {r['avg_reward_team1']:.1f}, "
          f"Ep_len: {r['avg_ep_len']:.0f}")

    # 2. MAPPO vs Random
    print("\n=== MAPPO (Team 1) vs Random (Team 2) ===")
    r = run_matchup(env, mappo.actor, "random", args.episodes, device, "MAPPO", "Random")
    results["MAPPO_vs_Random"] = r
    print(f"  MAPPO WR: {r['team1_winrate']:.2%}, Random WR: {r['team2_winrate']:.2%}, "
          f"Draw: {r['draw_rate']:.2%}, T1_reward: {r['avg_reward_team1']:.1f}, "
          f"Ep_len: {r['avg_ep_len']:.0f}")

    # 3. IPPO (T1) vs MAPPO (T2)
    print("\n=== IPPO (Team 1) vs MAPPO (Team 2) ===")
    r = run_matchup(env, ippo.actor, mappo.actor, args.episodes, device, "IPPO", "MAPPO")
    results["IPPO_vs_MAPPO"] = r
    print(f"  IPPO WR: {r['team1_winrate']:.2%}, MAPPO WR: {r['team2_winrate']:.2%}, "
          f"Draw: {r['draw_rate']:.2%}, T1_reward: {r['avg_reward_team1']:.1f}, "
          f"T2_reward: {r['avg_reward_team2']:.1f}, Ep_len: {r['avg_ep_len']:.0f}")

    # 4. MAPPO (T1) vs IPPO (T2) — swapped sides
    print("\n=== MAPPO (Team 1) vs IPPO (Team 2) ===")
    r = run_matchup(env, mappo.actor, ippo.actor, args.episodes, device, "MAPPO", "IPPO")
    results["MAPPO_vs_IPPO"] = r
    print(f"  MAPPO WR: {r['team1_winrate']:.2%}, IPPO WR: {r['team2_winrate']:.2%}, "
          f"Draw: {r['draw_rate']:.2%}, T1_reward: {r['avg_reward_team1']:.1f}, "
          f"T2_reward: {r['avg_reward_team2']:.1f}, Ep_len: {r['avg_ep_len']:.0f}")

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, r in results.items():
        print(f"{k}:  T1({k.split('_')[0]}) WR={r['team1_winrate']:.2%}  "
              f"T2({k.split('_')[-1]}) WR={r['team2_winrate']:.2%}  "
              f"Draw={r['draw_rate']:.2%}")

    # Save
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "matchup_results.txt", "w") as f:
        f.write("IPPO vs MAPPO Head-to-Head Results\n")
        f.write(f"Episodes per matchup: {args.episodes}\n\n")
        for k, r in results.items():
            f.write(f"{k}:\n")
            f.write(f"  Team1 WR: {r['team1_winrate']:.3f}, Team2 WR: {r['team2_winrate']:.3f}, Draw: {r['draw_rate']:.3f}\n")
            f.write(f"  Avg Reward: T1={r['avg_reward_team1']:.1f}, T2={r['avg_reward_team2']:.1f}\n")
            f.write(f"  Avg Ep Len: {r['avg_ep_len']:.0f}\n\n")
    print(f"\nResults saved to results/matchup_results.txt")


if __name__ == "__main__":
    main()
