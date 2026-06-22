"""Train QMIX on Quadrapong. Supports both RAM and pixel observations.

Usage:
    python scripts/train_qmix.py [--config src/configs/qmix.yaml]
    python scripts/train_qmix.py --config src/configs/qmix_pixel.yaml
"""
import sys
import os
import argparse
import yaml
import numpy as np
import torch
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.quadrapong_env import QuadrapongWrapper
from src.algos.qmix import QMIXTrainer
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
        log_dir=cfg_l["log_dir"], algo_name="qmix",
        use_wandb=cfg_l.get("use_wandb", False),
        wandb_project=cfg_l.get("wandb_project", "quadrapong"), config=config,
    )
    logger.print(f"QMIX training on {device}")

    env = QuadrapongWrapper(
        obs_type=config["env"]["obs_type"],
        max_cycles=config["env"]["max_cycles"],
        frame_stack=config["env"].get("frame_stack", 4),
        resize_dim=config["env"].get("resize_dim", None),
    )
    agents = env.possible_agents
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    state_dim = obs_dim * num_agents
    act_dim = env.action_space.n
    use_cnn = env.use_cnn
    obs_shape = env.obs_shape if use_cnn else None  # (C, H, W)
    logger.print(f"Obs dim={obs_dim}, use_cnn={use_cnn}, State dim={state_dim}, Actions={act_dim}")

    # Build networks
    if use_cnn:
        from src.utils.networks import CNNFFQNetwork, CNNEncoder
        input_channels = obs_shape[0]
        cnn_hidden = config["model"].get("cnn_hidden", 512)
        agent_q_net = CNNFFQNetwork(input_channels, act_dim, cnn_hidden)
        # Global state encoder: (N*C, H, W) → compact feature vector
        global_encoder = CNNEncoder(input_channels * num_agents)
        logger.print(f"  CNN: input_channels={input_channels}, feat_dim={global_encoder.feat_dim}")
    else:
        agent_q_net = None
        global_encoder = None

    trainer = QMIXTrainer(
        obs_dim=obs_dim, state_dim=state_dim, action_dim=act_dim,
        num_agents=num_agents,
        hidden_dims=config["model"]["q_hidden"],
        mixing_embed=config["model"]["mixing_embed"],
        hyper_hidden=config["model"]["hyper_hidden"],
        lr=cfg_t["lr"], gamma=cfg_t["gamma"],
        batch_size=cfg_t["batch_size"],
        buffer_capacity=cfg_t["buffer_capacity"],
        target_update_interval=cfg_t["target_update_interval"],
        epsilon_start=cfg_t["epsilon_start"],
        epsilon_end=cfg_t["epsilon_end"],
        epsilon_decay=cfg_t["epsilon_decay"],
        grad_norm_clip=cfg_t["grad_norm_clip"],
        double_q=cfg_t["double_q"],
        device=device,
        agent_q_net=agent_q_net,
        global_encoder=global_encoder,
    )
    trainer.obs_shape = obs_shape

    ckpt_dir = Path(cfg_l["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps = cfg_t["total_steps"]
    global_step = 0
    best_eval = -float("inf")

    # Team 1 = agents[0], agents[2] (first_0, third_0)
    # Team 2 = agents[1], agents[3] (second_0, fourth_0) — random opponent
    team1_indices = [0, 2]
    team2_indices = [1, 3]

    obs, _ = env.reset()
    ep_rewards = np.zeros(num_agents)
    ep_count = 0
    ep_lens = []
    ep_step = 0

    logger.print(f"Training {total_steps:,} steps, buffer={cfg_t['buffer_capacity']}, team1={team1_indices}")

    while global_step < total_steps:
        obs_batch = np.stack([obs[a] for a in agents])
        # Flatten pixel obs for buffer storage (buffer expects 2D per agent)
        obs_batch_flat = obs_batch.reshape(num_agents, -1) if use_cnn else obs_batch
        state = obs_batch_flat.reshape(-1)  # flattened global state for buffer

        actions = trainer.get_actions(obs_batch_flat if use_cnn else obs_batch)
        # Team 2 uses random actions (opponent per config)
        for idx in team2_indices:
            actions[idx] = np.random.randint(0, act_dim)

        action_dict = {a: int(actions[i]) for i, a in enumerate(agents)}
        next_obs, rewards, terms, truncs, _ = env.step(action_dict)

        reward_arr = np.array([rewards.get(a, 0) for a in agents], dtype=np.float32)
        # Reward shaping: small per-step reward to combat sparse goal rewards
        # ~0.004 total per episode vs sparse ±1, preserves optimal policy
        reward_arr[team1_indices] += 0.002
        done = any(terms.values()) or any(truncs.values())
        ep_rewards += reward_arr
        ep_step += 1

        next_obs_batch = np.stack([
            next_obs.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agents
        ])
        next_obs_batch_flat = next_obs_batch.reshape(num_agents, -1) if use_cnn else next_obs_batch
        next_state = next_obs_batch_flat.reshape(-1)

        trainer.push_to_buffer(
            obs=obs_batch_flat if use_cnn else obs_batch, state=state,
            actions=actions, rewards=reward_arr,
            next_obs=next_obs_batch_flat if use_cnn else next_obs_batch,
            next_state=next_state, done=done,
        )

        obs = next_obs
        global_step += 1

        # Training update (off-policy, with optional interval to throttle CNN training)
        train_interval = cfg_t.get("train_interval", 1)
        if global_step % train_interval == 0 and len(trainer.buffer) >= cfg_t["batch_size"]:
            metrics = trainer.update(None, global_step)
        else:
            metrics = None

        if metrics:
            logger.log_scalars({
                "q_loss": metrics["q_loss"],
                "q_tot_mean": metrics["q_tot_mean"],
                "epsilon": metrics["epsilon"],
            }, global_step)

        if done:
            ep_count += 1
            ep_lens.append(ep_step)
            obs, _ = env.reset()
            ep_rewards = np.zeros(num_agents)
            ep_step = 0

        if global_step % cfg_l["log_interval"] == 0:
            avg_ep_len = np.mean(ep_lens[-50:]) if ep_lens else 0
            logger.print(
                f"Step {global_step:>9,} | eps={trainer.epsilon:.3f} "
                f"q_loss={metrics['q_loss'] if metrics else 0:.4f} "
                f"q_tot={metrics['q_tot_mean'] if metrics else 0:.4f} "
                f"ep_len={avg_ep_len:.0f} eps_done={ep_count} "
                f"buf={len(trainer.buffer)} "
                f"elapsed={format_duration(logger.get_elapsed())}"
            )

        if global_step % cfg_e["eval_interval"] == 0 and global_step > 0:
            eval_env = QuadrapongWrapper(
                obs_type=config["env"]["obs_type"],
                max_cycles=config["env"]["max_cycles"],
                frame_stack=config["env"].get("frame_stack", 4),
                resize_dim=config["env"].get("resize_dim", None),
            )
            record = global_step % (cfg_e["eval_interval"] * 5) == 0
            eval_m = evaluate(eval_env, {"shared": trainer.agent_q},
                              num_episodes=cfg_e["num_eval_episodes"],
                              deterministic=True, device=device,
                              record_video=record,
                              video_path=str(Path(cfg_l["log_dir"]) / f"qmix/videos/step_{global_step}.mp4"),
                              random_agent_indices=team2_indices)
            eval_env.close()

            logger.log_eval(eval_m, global_step)
            winrate = max(eval_m["team_1_winrate"], eval_m["team_2_winrate"])
            logger.print(
                f"  Eval @ {global_step:,}: T1_WR={eval_m['team_1_winrate']:.2f} "
                f"T2_WR={eval_m['team_2_winrate']:.2f} len={eval_m['avg_episode_length']:.0f}"
            )
            if winrate > best_eval:
                best_eval = winrate
                trainer.save(str(ckpt_dir / "qmix_best.pt"))

        if global_step % cfg_l["save_interval"] == 0:
            trainer.save(str(ckpt_dir / f"qmix_step{global_step}.pt"))
            logger.print(f"  Checkpoint saved @ {global_step:,}")

    trainer.save(str(ckpt_dir / "qmix_final.pt"))
    logger.print(f"Training complete. Best eval winrate: {best_eval:.3f}")
    logger.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/qmix.yaml")
    parser.add_argument("--device", type=str, default=None, help="force device: cpu, cuda:0, cuda:1")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device:
        config["_device_override"] = args.device

    main(config)
