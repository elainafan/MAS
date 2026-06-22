"""Train MAPPO on Quadrapong.

Usage:
    python scripts/train_mappo.py [--config src/configs/mappo.yaml]
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
from src.algos.mappo import MAPPOTrainer
from src.utils.buffer import OnPolicyBuffer
from src.utils.evaluator import evaluate
from src.utils.logger import Logger, format_duration


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

    set_seed(cfg_t["seed"])
    device = config.get("_device_override", None)
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger = Logger(
        log_dir=cfg_l["log_dir"], algo_name="mappo",
        use_wandb=cfg_l.get("use_wandb", False),
        wandb_project=cfg_l.get("wandb_project", "quadrapong"), config=config,
    )
    logger.print(f"MAPPO training on {device}")

    env = QuadrapongWrapper(
        obs_type=config["env"]["obs_type"],
        max_cycles=config["env"]["max_cycles"],
        frame_stack=config["env"].get("frame_stack", 4),
        resize_dim=config["env"].get("resize_dim", None),
    )
    agents = env.possible_agents
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    global_obs_dim = obs_dim * num_agents
    act_dim = env.action_space.n
    use_cnn = env.use_cnn
    obs_shape = env.obs_shape if use_cnn else None  # (C, H, W)
    logger.print(f"Obs dim={obs_dim}, use_cnn={use_cnn}, Global dim={global_obs_dim}, Actions={act_dim}")

    # Build networks
    if use_cnn:
        from src.utils.networks import CNNActor, CNNCentralizedCritic
        input_channels = obs_shape[0]
        cnn_hidden = config["model"].get("cnn_hidden", 512)
        actor = CNNActor(input_channels, act_dim, cnn_hidden)
        critic = CNNCentralizedCritic(input_channels, num_agents, cnn_hidden)
    else:
        actor, critic = None, None

    trainer = MAPPOTrainer(
        obs_dim=obs_dim, global_obs_dim=global_obs_dim, action_dim=act_dim,
        num_agents=num_agents, hidden_dims=config["model"].get("actor_hidden", [128, 128]),
        lr=cfg_t["lr"], gamma=cfg_t["gamma"], gae_lambda=cfg_t["gae_lambda"],
        clip_param=cfg_t["clip_param"], entropy_coef=cfg_t["entropy_coef"],
        value_coef=cfg_t["value_coef"], max_grad_norm=cfg_t["max_grad_norm"],
        ppo_epochs=cfg_t["ppo_epochs"], mini_batch_size=cfg_t["mini_batch_size"],
        use_popart=cfg_t.get("use_popart", False), device=device,
        actor=actor, critic=critic,
    )
    trainer.obs_shape = obs_shape

    ckpt_dir = Path(cfg_l["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg_t["total_steps"]
    rollout_steps = cfg_t["rollout_steps"]
    global_step = 0
    best_eval = -float("inf")

    obs, _ = env.reset()
    ep_rewards = np.zeros(num_agents)
    ep_count = 0
    ep_lens = []

    logger.print(f"Training {total_steps:,} steps, rollout={rollout_steps}")

    while global_step < total_steps:
        buffer = OnPolicyBuffer(num_agents, obs_dim, rollout_steps,
                                gamma=trainer.gamma, gae_lambda=trainer.gae_lambda)

        last_done_arr = np.zeros(num_agents, dtype=np.float32)
        for step in range(rollout_steps):
            obs_batch = np.stack([obs[a] for a in agents])
            obs_batch_flat = obs_batch.reshape(num_agents, -1)
            global_state = obs_batch_flat.reshape(-1)
            actions, log_probs = trainer.get_actions(obs_batch_flat)

            # Centralized values
            obs_t = torch.tensor(obs_batch_flat, dtype=torch.float32, device=device)
            global_t = torch.tensor(global_state, dtype=torch.float32, device=device)
            obs_shape = getattr(trainer, 'obs_shape', None)
            if obs_shape is not None:
                C, H, W = obs_shape
                obs_t_2d = obs_t.reshape(num_agents, C, H, W)
                global_t_2d = global_t.reshape(num_agents * C, H, W)
            else:
                obs_t_2d = obs_t
                global_t_2d = global_t
            with torch.no_grad():
                # Batched: expand global across agent dim, obs_t_2d already (N, ...)
                global_batch = global_t_2d.unsqueeze(0).expand(num_agents, *([-1] * global_t_2d.ndim))
                values = trainer.critic(global_batch, obs_t_2d).squeeze(-1)

            action_dict = {a: int(actions[i]) for i, a in enumerate(agents)}
            next_obs, rewards, terms, truncs, _ = env.step(action_dict)

            reward_arr = np.array([rewards.get(a, 0) for a in agents], dtype=np.float32)
            done_arr = np.array(
                [float(terms.get(a, False) or truncs.get(a, False)) for a in agents],
                dtype=np.float32,
            )
            ep_rewards += reward_arr
            last_done_arr = done_arr

            buffer.insert(obs=obs_batch_flat, actions=actions, action_log_probs=log_probs,
                          rewards=reward_arr, values=values.cpu().numpy(), dones=done_arr,
                          global_state=global_state)

            obs = {a: next_obs.get(a, obs[a]) for a in agents}

            if np.any(done_arr):
                ep_count += 1
                ep_lens.append(step + 1)
                obs, _ = env.reset()
                ep_rewards = np.zeros(num_agents)

        global_step += rollout_steps

        # GAE with centralized values (from rollout, no recomputation)
        data = buffer.get_training_data()

        last_obs_batch = np.stack([obs[a] for a in agents])
        last_obs_batch_flat = last_obs_batch.reshape(num_agents, -1)
        last_global_t = torch.tensor(last_obs_batch_flat.reshape(-1), dtype=torch.float32, device=device)
        last_obs_t = torch.tensor(last_obs_batch_flat, dtype=torch.float32, device=device)
        obs_shape = getattr(trainer, 'obs_shape', None)
        if obs_shape is not None:
            C, H, W = obs_shape
            last_obs_t_2d = last_obs_t.reshape(num_agents, C, H, W)
            last_global_t_2d = last_global_t.reshape(num_agents * C, H, W)
        else:
            last_obs_t_2d = last_obs_t
            last_global_t_2d = last_global_t

        with torch.no_grad():
            last_global_batch = last_global_t_2d.unsqueeze(0).expand(num_agents, *([-1] * last_global_t_2d.ndim))
            last_values = trainer.critic(last_global_batch, last_obs_t_2d).squeeze(-1)

        advantages, returns = buffer.compute_gae(last_values.cpu().numpy(), last_done_arr)

        metrics = _train_mappo_update(trainer, data, advantages, returns)

        if global_step % cfg_l["log_interval"] < rollout_steps:
            avg_ep_r = ep_rewards.mean()
            avg_ep_len = np.mean(ep_lens[-100:]) if ep_lens else 0
            logger.log_scalars({
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "ep_reward_mean": avg_ep_r,
                "ep_len_mean": avg_ep_len,
            }, global_step)
            logger.print(
                f"Step {global_step:>9,} | p_loss={metrics['policy_loss']:.4f} "
                f"v_loss={metrics['value_loss']:.4f} ent={metrics['entropy']:.4f} "
                f"ep_r={avg_ep_r:.2f} ep_len={avg_ep_len:.0f} eps={ep_count} "
                f"elapsed={format_duration(logger.get_elapsed())}"
            )

        if global_step % cfg_e["eval_interval"] < rollout_steps:
            eval_env = QuadrapongWrapper(
                obs_type=config["env"]["obs_type"],
                max_cycles=config["env"]["max_cycles"],
                frame_stack=config["env"].get("frame_stack", 4),
                resize_dim=config["env"].get("resize_dim", None),
            )
            eval_m = evaluate(eval_env, {"shared": trainer.actor},
                              num_episodes=cfg_e["num_eval_episodes"],
                              deterministic=True, device=device,
                              record_video=(global_step % (cfg_e["eval_interval"] * 5) < rollout_steps),
                              video_path=str(Path(cfg_l["log_dir"]) / f"mappo/videos/step_{global_step}.mp4"))
            eval_env.close()

            logger.log_eval(eval_m, global_step)
            winrate = max(eval_m["team_1_winrate"], eval_m["team_2_winrate"])
            logger.print(
                f"  Eval @ {global_step:,}: T1_WR={eval_m['team_1_winrate']:.2f} "
                f"T2_WR={eval_m['team_2_winrate']:.2f} len={eval_m['avg_episode_length']:.0f}"
            )
            if winrate > best_eval:
                best_eval = winrate
                trainer.save(str(ckpt_dir / "mappo_best.pt"))

        if global_step % cfg_l["save_interval"] < rollout_steps:
            trainer.save(str(ckpt_dir / f"mappo_step{global_step}.pt"))
            logger.print(f"  Checkpoint saved @ {global_step:,}")

    trainer.save(str(ckpt_dir / "mappo_final.pt"))
    logger.print(f"Training complete. Best eval winrate: {best_eval:.3f}")
    logger.close()
    env.close()


def _train_mappo_update(trainer, data, advantages, returns):
    """MAPPO PPO update with centralized critic. Handles both RAM and CNN obs."""
    num_steps, num_agents, obs_dim = data["obs"].shape
    obs_shape = getattr(trainer, 'obs_shape', None)

    obs_flat = data["obs"].reshape(-1, obs_dim)
    actions_flat = data["actions"].reshape(-1)
    old_lp_flat = data["action_log_probs"].reshape(-1)
    adv_flat = advantages.reshape(-1)
    ret_flat = returns.reshape(-1)
    global_expanded = np.repeat(data["global_state"], num_agents, axis=0)

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
            global_b = torch.tensor(global_expanded[idx], dtype=torch.float32, device=device)
            if obs_shape is not None:
                C, H, W = obs_shape
                obs_b = obs_b.reshape(-1, C, H, W)
                global_b = global_b.reshape(-1, num_agents * C, H, W)
            act_b = torch.tensor(actions_flat[idx], dtype=torch.long, device=device)
            old_lp_b = torch.tensor(old_lp_flat[idx], dtype=torch.float32, device=device)
            adv_b = torch.tensor(adv_flat[idx], dtype=torch.float32, device=device)
            ret_b = torch.tensor(ret_flat[idx], dtype=torch.float32, device=device)

            new_lp, entropy = trainer.actor.evaluate_actions(obs_b, None, act_b)
            ratio = torch.exp(new_lp - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - trainer.clip_param, 1.0 + trainer.clip_param) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            values_pred = trainer.critic(global_b, obs_b).squeeze(-1)
            if trainer.use_popart:
                ret_norm = trainer.value_normalizer.normalize(ret_b.unsqueeze(-1))
                value_loss = nn.functional.mse_loss(values_pred, ret_norm.squeeze(-1))
                trainer.value_normalizer.update(ret_b.unsqueeze(-1))
            else:
                value_loss = nn.functional.mse_loss(values_pred, ret_b)

            trainer.actor_optimizer.zero_grad()
            (policy_loss - trainer.entropy_coef * entropy.mean()).backward()
            nn.utils.clip_grad_norm_(trainer.actor.parameters(), trainer.max_grad_norm)
            trainer.actor_optimizer.step()

            trainer.critic_optimizer.zero_grad()
            (trainer.value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(trainer.critic.parameters(), trainer.max_grad_norm)
            trainer.critic_optimizer.step()

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
    parser.add_argument("--config", type=str, default="src/configs/mappo.yaml")
    parser.add_argument("--device", type=str, default=None, help="force device: cpu, cuda:0, cuda:1")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device:
        config["_device_override"] = args.device

    main(config)
