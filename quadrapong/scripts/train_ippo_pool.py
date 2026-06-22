"""Train IPPO with Opponent Pool self-play on Quadrapong.

Team 1 = agents [first_0, third_0] (current training policy, updating).
Team 2 = agents [second_0, fourth_0] (opponent from pool, frozen).

Usage:
    python scripts/train_ippo_pool.py [--config src/configs/ippo_pool.yaml]
"""
import sys
import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.quadrapong_env import QuadrapongWrapper
from src.algos.ippo import IPPOTrainer
from src.utils.buffer import OnPolicyBuffer
from src.utils.evaluator import evaluate
from src.utils.logger import Logger, format_duration
from src.utils.opponent_pool import OpponentPool, get_opponent_actions
from src.utils.networks import StochasticActor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config: dict):
    cfg_t = config["training"]
    cfg_e = config["eval"]
    cfg_l = config["logging"]
    cfg_p = config["pool"]

    set_seed(cfg_t["seed"])
    device = config.get("_device_override", None)
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger = Logger(
        log_dir=cfg_l["log_dir"], algo_name="ippo_pool",
        use_wandb=cfg_l.get("use_wandb", False),
        wandb_project=cfg_l.get("wandb_project", "quadrapong"), config=config,
    )
    logger.print(f"IPPO+Pool training on {device}")

    env = QuadrapongWrapper(
        obs_type=config["env"]["obs_type"],
        max_cycles=config["env"]["max_cycles"],
        frame_stack=config["env"].get("frame_stack", 4),
        resize_dim=config["env"].get("resize_dim", None),
    )
    agents = env.possible_agents
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    act_dim = env.action_space.n
    use_cnn = env.use_cnn
    obs_shape = env.obs_shape if use_cnn else None
    logger.print(f"Obs dim={obs_dim}, Actions={act_dim}, Agents={num_agents}, use_cnn={use_cnn}")

    # Build networks
    if use_cnn:
        from src.utils.networks import CNNActor, CNNCritic
        input_channels = obs_shape[0]
        cnn_hidden = config["model"].get("cnn_hidden", 512)
        actor = CNNActor(input_channels, act_dim, cnn_hidden)
        critic = CNNCritic(input_channels, cnn_hidden)
    else:
        actor, critic = None, None

    trainer = IPPOTrainer(
        obs_dim=obs_dim, action_dim=act_dim, num_agents=2,  # Only Team 1 agents
        hidden_dims=config["model"].get("actor_hidden", [128, 128]),
        lr=cfg_t["lr"], gamma=cfg_t["gamma"], gae_lambda=cfg_t["gae_lambda"],
        clip_param=cfg_t["clip_param"], entropy_coef=cfg_t["entropy_coef"],
        value_coef=cfg_t["value_coef"], max_grad_norm=cfg_t["max_grad_norm"],
        ppo_epochs=cfg_t["ppo_epochs"], mini_batch_size=cfg_t["mini_batch_size"],
        device=device, actor=actor, critic=critic,
    )
    trainer.obs_shape = obs_shape

    # Opponent Pool
    hidden_dims = config["model"].get("actor_hidden", [128, 128])
    if use_cnn:
        from src.utils.networks import CNNActor
        opponent_pool = OpponentPool(
            actor_class=CNNActor,
            actor_kwargs={"input_channels": input_channels, "action_dim": act_dim,
                          "cnn_hidden": cnn_hidden},
            capacity=cfg_p.get("capacity", 5),
            history_weight=cfg_p.get("history_weight", 0.6),
            random_weight=cfg_p.get("random_weight", 0.2),
            current_weight=cfg_p.get("current_weight", 0.2),
            device=device,
        )
    else:
        opponent_pool = OpponentPool(
            actor_class=StochasticActor,
            actor_kwargs={"obs_dim": obs_dim, "action_dim": act_dim,
                          "hidden_dims": hidden_dims},
            capacity=cfg_p.get("capacity", 5),
            history_weight=cfg_p.get("history_weight", 0.6),
            random_weight=cfg_p.get("random_weight", 0.2),
            current_weight=cfg_p.get("current_weight", 0.2),
            device=device,
        )

    ckpt_dir = Path(cfg_l["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg_t["total_steps"]
    rollout_steps = cfg_t["rollout_steps"]
    pool_save_interval = cfg_p.get("save_interval", 200000)
    global_step = 0
    best_eval = -float("inf")

    # Team 1 = agents[0], agents[2] (first_0, third_0)
    # Team 2 = agents[1], agents[3] (second_0, fourth_0)
    team1_indices = [0, 2]
    team2_indices = [1, 3]
    n_t1 = len(team1_indices)
    n_t2 = len(team2_indices)

    obs, _ = env.reset()
    ep_rewards_t1 = np.zeros(n_t1)
    ep_count = 0
    ep_lens = []
    ep_step = 0
    opp_sample_count = {"history": 0, "random": 0, "current": 0}
    opponent, opp_type = opponent_pool.sample(trainer.actor)
    opp_sample_count[opp_type] += 1

    logger.print(f"Training {total_steps:,} steps, rollout={rollout_steps}, "
                 f"pool_capacity={cfg_p.get('capacity', 5)}, "
                 f"pool_save_interval={pool_save_interval}")

    while global_step < total_steps:
        buffer = OnPolicyBuffer(n_t1, obs_dim, rollout_steps,
                                gamma=trainer.gamma, gae_lambda=trainer.gae_lambda,
                                global_obs_dim=obs_dim * num_agents)

        last_done_arr = np.zeros(n_t1, dtype=np.float32)
        for step in range(rollout_steps):
            obs_batch = np.stack([obs[a] for a in agents])  # (4, obs_dim)

            # Team 1 actions from current policy
            t1_obs = obs_batch[team1_indices]  # (2, obs_dim)
            t1_actions, t1_log_probs = trainer.get_actions(t1_obs)
            t1_values = trainer.get_values(t1_obs)

            # Team 2 actions from opponent
            t2_obs = obs_batch[team2_indices]
            t2_actions = get_opponent_actions(opponent, t2_obs, act_dim, device, obs_shape)

            actions = np.zeros(num_agents, dtype=int)
            actions[team1_indices] = t1_actions
            actions[team2_indices] = t2_actions

            action_dict = {a: int(actions[i]) for i, a in enumerate(agents)}
            next_obs, rewards, terms, truncs, _ = env.step(action_dict)

            reward_arr = np.array([rewards.get(a, 0) for a in agents], dtype=np.float32)
            t1_rewards = reward_arr[team1_indices]
            done_arr = np.array(
                [float(terms.get(a, False) or truncs.get(a, False)) for a in agents],
                dtype=np.float32,
            )
            t1_dones = done_arr[team1_indices]
            ep_rewards_t1 += t1_rewards
            last_done_arr = t1_dones
            ep_step += 1

            # Only Team 1 data goes into buffer; store full global_state for logging
            buffer.insert(
                obs=t1_obs,
                actions=t1_actions,
                action_log_probs=t1_log_probs,
                rewards=t1_rewards,
                values=t1_values,
                dones=t1_dones,
                global_state=obs_batch.reshape(-1),
            )

            obs = {a: next_obs.get(a, obs[a]) for a in agents}

            if np.any(done_arr):
                ep_count += 1
                ep_lens.append(ep_step)
                ep_step = 0
                obs, _ = env.reset()
                ep_rewards_t1 = np.zeros(n_t1)
                # Sample new opponent for next episode
                opponent, opp_type = opponent_pool.sample(trainer.actor)
                opp_sample_count[opp_type] += 1

        global_step += rollout_steps

        # GAE + train on Team 1 data
        last_obs_batch = np.stack([obs[a] for a in agents])
        last_t1_obs = last_obs_batch[team1_indices]
        last_t1_values = trainer.get_values(last_t1_obs)

        data = buffer.get_training_data()
        advantages, returns = buffer.compute_gae(last_t1_values, last_done_arr)
        metrics = _train_ppo_update(trainer, data, advantages, returns)

        # Logging
        if global_step % cfg_l["log_interval"] < rollout_steps:
            avg_ep_r = ep_rewards_t1.mean()
            avg_ep_len = np.mean(ep_lens[-100:]) if ep_lens else 0
            logger.log_scalars({
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "ep_reward_mean": avg_ep_r,
                "ep_len_mean": avg_ep_len,
                "pool_size": opponent_pool.size,
            }, global_step)
            pool_info = opponent_pool.pool_stats()
            logger.print(
                f"Step {global_step:>9,} | p_loss={metrics['policy_loss']:.4f} "
                f"v_loss={metrics['value_loss']:.4f} ent={metrics['entropy']:.4f} "
                f"ep_r={avg_ep_r:.2f} ep_len={avg_ep_len:.0f} eps={ep_count} "
                f"pool={pool_info['pool_size']}/{pool_info['capacity']} "
                f"opp=(h:{opp_sample_count['history']},r:{opp_sample_count['random']},"
                f"c:{opp_sample_count['current']}) "
                f"elapsed={format_duration(logger.get_elapsed())}"
            )

        # Evaluation
        if global_step % cfg_e["eval_interval"] < rollout_steps:
            eval_env = QuadrapongWrapper(
                obs_type=config["env"]["obs_type"],
                max_cycles=config["env"]["max_cycles"],
                frame_stack=config["env"].get("frame_stack", 4),
                resize_dim=config["env"].get("resize_dim", None),
            )

            # Eval 1: vs random opponents (standard baseline)
            eval_m_random = evaluate(
                eval_env, {"shared": trainer.actor},
                num_episodes=cfg_e["num_eval_episodes"],
                deterministic=True, device=device,
                random_agent_indices=team2_indices,
            )
            logger.print(
                f"  Eval vs Random @ {global_step:,}: T1_WR={eval_m_random['team_1_winrate']:.2f} "
                f"T2_WR={eval_m_random['team_2_winrate']:.2f} len={eval_m_random['avg_episode_length']:.0f}"
            )

            # Eval 2: vs current self (standard self-play eval)
            eval_m_self = evaluate(
                eval_env, {"shared": trainer.actor},
                num_episodes=cfg_e["num_eval_episodes"],
                deterministic=True, device=device,
            )
            logger.print(
                f"  Eval vs Self  @ {global_step:,}: T1_WR={eval_m_self['team_1_winrate']:.2f} "
                f"T2_WR={eval_m_self['team_2_winrate']:.2f} len={eval_m_self['avg_episode_length']:.0f}"
            )

            eval_env.close()

            # Use self-play winrate for best model selection (higher ceiling)
            winrate = max(eval_m_self["team_1_winrate"], eval_m_self["team_2_winrate"])
            logger.log_eval(eval_m_self, global_step)

            if winrate > best_eval:
                best_eval = winrate
                trainer.save(str(ckpt_dir / "ippo_pool_best.pt"))

        # Checkpoint to disk
        if global_step % cfg_l["save_interval"] < rollout_steps:
            trainer.save(str(ckpt_dir / f"ippo_pool_step{global_step}.pt"))
            logger.print(f"  Checkpoint saved @ {global_step:,}")

        # Checkpoint to opponent pool
        if global_step % pool_save_interval < rollout_steps:
            opponent_pool.add_checkpoint(
                {k: v.cpu() for k, v in trainer.actor.state_dict().items()},
                global_step,
            )
            logger.print(
                f"  Pool checkpoint added @ {global_step:,} "
                f"(pool={opponent_pool.size}/{cfg_p.get('capacity', 5)})"
            )

    trainer.save(str(ckpt_dir / "ippo_pool_final.pt"))
    logger.print(f"Training complete. Best eval winrate: {best_eval:.3f}")
    logger.print(f"Opponent samples: {opp_sample_count}")
    logger.close()
    env.close()


def _train_ppo_update(trainer, data, advantages, returns):
    """Standard PPO update on Team 1 data only."""
    num_steps, num_agents, obs_dim = data["obs"].shape
    obs_shape = getattr(trainer, 'obs_shape', None)

    obs_flat = data["obs"].reshape(-1, obs_dim)
    actions_flat = data["actions"].reshape(-1)
    old_lp_flat = data["action_log_probs"].reshape(-1)
    adv_flat = advantages.reshape(-1)
    ret_flat = returns.reshape(-1)

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    total_batch = num_steps * num_agents

    total_p_loss, total_v_loss, total_ent = 0.0, 0.0, 0.0
    n_updates = 0
    device = trainer.device

    for _ in range(trainer.ppo_epochs):
        indices = np.random.permutation(total_batch)
        for start in range(0, total_batch, trainer.mini_batch_size):
            idx = indices[start:start + trainer.mini_batch_size]

            obs_b = torch.tensor(obs_flat[idx], dtype=torch.float32, device=device)
            if obs_shape is not None:
                obs_b = obs_b.reshape(-1, *obs_shape)
            act_b = torch.tensor(actions_flat[idx], dtype=torch.long, device=device)
            old_lp_b = torch.tensor(old_lp_flat[idx], dtype=torch.float32, device=device)
            adv_b = torch.tensor(adv_flat[idx], dtype=torch.float32, device=device)
            ret_b = torch.tensor(ret_flat[idx], dtype=torch.float32, device=device)

            new_lp, entropy = trainer.actor.evaluate_actions(obs_b, None, act_b)
            ratio = torch.exp(new_lp - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - trainer.clip_param, 1.0 + trainer.clip_param) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            values_pred = trainer.critic(obs_b).squeeze(-1)
            value_loss = nn.functional.mse_loss(values_pred, ret_b)

            loss = (policy_loss + trainer.value_coef * value_loss
                    - trainer.entropy_coef * entropy.mean())

            trainer.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainer.actor.parameters(), trainer.max_grad_norm)
            nn.utils.clip_grad_norm_(trainer.critic.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()

            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            total_ent += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_p_loss / n_updates,
        "value_loss": total_v_loss / n_updates,
        "entropy": total_ent / n_updates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/ippo_pool.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device:
        config["_device_override"] = args.device

    main(config)
