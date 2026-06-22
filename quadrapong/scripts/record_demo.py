"""Record head-to-head Quadrapong demo videos.

This script reuses the model loading and action-selection helpers from
``eval_unified.py`` so the videos match the quantitative evaluation setup.

Examples:
    python scripts/record_demo.py --preset all --device cuda:0
    python scripts/record_demo.py --preset ippo_pool_vs_ippo --max-steps 20000
    python scripts/record_demo.py --team1 checkpoints/ippo/ippo_final.pt \
        --team2 random --name ippo_vs_random
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_unified import get_actions, get_algo_name, load_model, _obs_to_tensor, _step_env
from src.envs.quadrapong_env import QuadrapongWrapper


AGENTS = ["first_0", "second_0", "third_0", "fourth_0"]
TEAM1_IDX = [0, 2]
TEAM2_IDX = [1, 3]
TEAM1_AGENTS = ["first_0", "third_0"]
TEAM2_AGENTS = ["second_0", "fourth_0"]


PRESETS = {
    "ippo_vs_random": {
        "team1": "checkpoints/ippo/ippo_final.pt",
        "team2": "random",
        "caption": "Baseline IPPO policy against a random opponent.",
    },
    "ippo_pool_vs_ippo": {
        "team1": "checkpoints/ippo_pool/ippo_pool_final.pt",
        "team2": "checkpoints/ippo/ippo_final.pt",
        "caption": "Opponent-pool self-play breaks the standard IPPO side lock-in.",
    },
    "mappo_vs_qmix": {
        "team1": "checkpoints/mappo/mappo_final.pt",
        "team2": "checkpoints/qmix/qmix_final.pt",
        "caption": "QMIX is dominant when placed on Team 2 against MAPPO.",
    },
    "random_vs_mappo": {
        "team1": "random",
        "team2": "checkpoints/mappo/mappo_final.pt",
        "caption": "Random Team 1 exposes MAPPO collapse in the evaluated checkpoint.",
    },
}


def effective_obs(type_a, obs_a, type_b, obs_b):
    if type_a == "random":
        eff_a = obs_b
    else:
        eff_a = obs_a
    if type_b == "random":
        eff_b = eff_a
    else:
        eff_b = obs_b
    return eff_a, eff_b


def team_score(rewards):
    t1 = sum(rewards.get(a, 0.0) for a in TEAM1_AGENTS)
    t2 = sum(rewards.get(a, 0.0) for a in TEAM2_AGENTS)
    return float(t1), float(t2)


def overlay(frame, step, t1_name, t2_name, t1_score, t2_score, caption):
    img = np.ascontiguousarray(frame.copy())
    h, w = img.shape[:2]
    panel_h = 54 if caption else 38
    cv2.rectangle(img, (0, 0), (w, panel_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        img,
        f"{t1_name} (T1)  {t1_score:+.0f}  vs  {t2_score:+.0f}  {t2_name} (T2)   step {step}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if caption:
        cv2.putText(
            img,
            caption[:110],
            (8, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return img


def open_writer(path, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, quality=8, macro_block_size=None)


def make_env(env_kwargs, render=False):
    return QuadrapongWrapper(max_cycles=100000, render_mode="rgb_array" if render else None, **env_kwargs)


def record_same_obs(
    output,
    team1_mod,
    team2_mod,
    team1_type,
    team2_type,
    env_kwargs,
    device,
    t1_name,
    t2_name,
    caption,
    seed,
    max_steps,
    frame_stride,
    fps,
):
    env = make_env(env_kwargs, render=True)
    writer = open_writer(output, fps)
    frame_count = 0
    ep_rewards = {a: 0.0 for a in AGENTS}
    try:
        obs, _ = env.reset(seed=seed)
        step = 0
        done = False
        initial_frame = env.render()
        if initial_frame is not None:
            writer.append_data(overlay(initial_frame, step, t1_name, t2_name, 0.0, 0.0, caption))
            frame_count += 1
        while not done and step < max_steps:
            obs_t = _obs_to_tensor(obs, AGENTS, device)
            actions = np.zeros(4, dtype=int)
            actions[TEAM1_IDX] = get_actions(team1_mod, team1_type, obs_t, TEAM1_IDX, device)
            actions[TEAM2_IDX] = get_actions(team2_mod, team2_type, obs_t, TEAM2_IDX, device)
            obs, rewards, done = _step_env(env, obs, actions, AGENTS)
            for agent in AGENTS:
                ep_rewards[agent] += rewards.get(agent, 0.0)
            step += 1
            if step % frame_stride == 0 or done or step == max_steps:
                t1_score, t2_score = team_score(ep_rewards)
                frame = env.render()
                if frame is not None:
                    writer.append_data(overlay(frame, step, t1_name, t2_name, t1_score, t2_score, caption))
                    frame_count += 1
    finally:
        writer.close()
        env.close()
    return finalize(output, t1_name, t2_name, caption, seed, step, ep_rewards, frame_count, fps, frame_stride)


def record_cross_obs(
    output,
    team1_mod,
    team2_mod,
    team1_type,
    team2_type,
    env_kwargs_t1,
    env_kwargs_t2,
    device,
    t1_name,
    t2_name,
    caption,
    seed,
    max_steps,
    frame_stride,
    fps,
):
    env_t1 = make_env(env_kwargs_t1, render=False)
    env_t2 = make_env(env_kwargs_t2, render=False)
    render_env = make_env({"obs_type": "rgb_image"}, render=True)
    writer = open_writer(output, fps)
    frame_count = 0
    ep_rewards = {a: 0.0 for a in AGENTS}
    try:
        obs_t1, _ = env_t1.reset(seed=seed)
        obs_t2, _ = env_t2.reset(seed=seed)
        render_obs, _ = render_env.reset(seed=seed)
        step = 0
        done = False
        initial_frame = render_env.render()
        if initial_frame is not None:
            writer.append_data(overlay(initial_frame, step, t1_name, t2_name, 0.0, 0.0, caption))
            frame_count += 1
        while not done and step < max_steps:
            obs_t1_tensor = _obs_to_tensor(obs_t1, AGENTS, device)
            obs_t2_tensor = _obs_to_tensor(obs_t2, AGENTS, device)
            actions = np.zeros(4, dtype=int)
            actions[TEAM1_IDX] = get_actions(team1_mod, team1_type, obs_t1_tensor, TEAM1_IDX, device)
            actions[TEAM2_IDX] = get_actions(team2_mod, team2_type, obs_t2_tensor, TEAM2_IDX, device)

            obs_t1, rewards, done = _step_env(env_t1, obs_t1, actions, AGENTS)
            obs_t2, rewards_t2, _ = _step_env(env_t2, obs_t2, actions, AGENTS)
            render_obs, rewards_render, _ = _step_env(render_env, render_obs, actions, AGENTS)
            for agent in AGENTS:
                if abs(rewards.get(agent, 0.0) - rewards_t2.get(agent, 0.0)) > 1e-6:
                    raise RuntimeError(f"cross-observation env diverged for {agent} at step {step}")
                if abs(rewards.get(agent, 0.0) - rewards_render.get(agent, 0.0)) > 1e-6:
                    raise RuntimeError(f"render env diverged for {agent} at step {step}")
                ep_rewards[agent] += rewards.get(agent, 0.0)
            step += 1
            if step % frame_stride == 0 or done or step == max_steps:
                t1_score, t2_score = team_score(ep_rewards)
                frame = render_env.render()
                if frame is not None:
                    writer.append_data(overlay(frame, step, t1_name, t2_name, t1_score, t2_score, caption))
                    frame_count += 1
    finally:
        writer.close()
        env_t1.close()
        env_t2.close()
        render_env.close()
    return finalize(output, t1_name, t2_name, caption, seed, step, ep_rewards, frame_count, fps, frame_stride)


def finalize(output, t1_name, t2_name, caption, seed, steps, ep_rewards, frame_count, fps, frame_stride):
    t1_score, t2_score = team_score(ep_rewards)
    if t1_score > t2_score:
        winner = t1_name
    elif t2_score > t1_score:
        winner = t2_name
    else:
        winner = "draw"
    meta = {
        "video": str(output),
        "team1": t1_name,
        "team2": t2_name,
        "caption": caption,
        "seed": int(seed),
        "steps": int(steps),
        "team1_score": t1_score,
        "team2_score": t2_score,
        "winner": winner,
        "frames": int(frame_count),
        "fps": int(fps),
        "frame_stride": int(frame_stride),
    }
    meta_path = Path(str(output)).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {output} ({frame_count} frames, winner={winner}, score={t1_score:+.0f}:{t2_score:+.0f})")
    print(f"Saved {meta_path}")
    return meta


def record_one(args, name, team1, team2, caption):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    team1_mod, team1_type, obs1, env_kwargs1 = load_model(team1, device)
    team2_mod, team2_type, obs2, env_kwargs2 = load_model(team2, device)
    t1_name = get_algo_name(team1)
    t2_name = get_algo_name(team2)
    eff_obs1, eff_obs2 = effective_obs(team1_type, obs1, team2_type, obs2)

    output = args.output
    if output is None:
        output = str(Path(args.output_dir) / f"{name}.mp4")

    if team1_type == "random" and team2_type != "random":
        env_kwargs1 = env_kwargs2
    if team2_type == "random" and team1_type != "random":
        env_kwargs2 = env_kwargs1

    print(f"Recording {name}: {t1_name}(T1) vs {t2_name}(T2), device={device}, obs={eff_obs1}/{eff_obs2}")
    if eff_obs1 == eff_obs2:
        env_kwargs = env_kwargs2 if team1_type == "random" else env_kwargs1
        return record_same_obs(
            output, team1_mod, team2_mod, team1_type, team2_type, env_kwargs, device,
            t1_name, t2_name, caption, args.seed, args.max_steps, args.frame_stride, args.fps,
        )
    return record_cross_obs(
        output, team1_mod, team2_mod, team1_type, team2_type, env_kwargs1, env_kwargs2, device,
        t1_name, t2_name, caption, args.seed, args.max_steps, args.frame_stride, args.fps,
    )


def main():
    parser = argparse.ArgumentParser(description="Record Quadrapong head-to-head demo videos")
    parser.add_argument("--preset", choices=["all"] + sorted(PRESETS.keys()))
    parser.add_argument("--team1", help="Team 1 checkpoint path or 'random'")
    parser.add_argument("--team2", help="Team 2 checkpoint path or 'random'")
    parser.add_argument("--name", default="custom_demo")
    parser.add_argument("--caption", default="")
    parser.add_argument("--output", help="Output mp4 path for a single custom/preset video")
    parser.add_argument("--output-dir", default="results/videos")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.preset:
        names = sorted(PRESETS.keys()) if args.preset == "all" else [args.preset]
        metas = []
        for preset_name in names:
            spec = PRESETS[preset_name]
            if args.preset == "all":
                args.output = None
            metas.append(record_one(args, preset_name, spec["team1"], spec["team2"], spec["caption"]))
        manifest = Path(args.output_dir) / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(metas, indent=2), encoding="utf-8")
        print(f"Saved {manifest}")
        return

    if not args.team1 or not args.team2:
        parser.error("provide --preset or both --team1 and --team2")
    record_one(args, args.name, args.team1, args.team2, args.caption)


if __name__ == "__main__":
    main()
