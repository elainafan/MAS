"""Plot learning curves for all trained models from TensorBoard logs.

Reads event files from logs/tensorboard/<algo>/ and produces multi-panel
comparison plots.

Usage:
    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --output results/plots/learning_curves.png
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

MODELS = {
    "IPPO":          ("ippo",       "RAM",      "#1f77b4", "-"),
    "MAPPO":         ("mappo",      "RAM",      "#ff7f0e", "-"),
    "QMIX":          ("qmix",       "RAM",      "#2ca02c", "-"),
    "IPPO_POOL":     ("ippo_pool",  "RAM+Pool", "#d62728", "-"),
    "MAPPO_POOL":    ("mappo_pool", "RAM+Pool", "#9467bd", "-"),
    "IPPO pixel":    ("ippo_pixel", "Pixel",    "#1f77b4", "--"),
    "MAPPO pixel":   ("mappo_pixel","Pixel",    "#ff7f0e", "--"),
}

RAM_PPO_MODELS = ["IPPO", "MAPPO", "IPPO_POOL", "MAPPO_POOL"]
ALL_PPO = ["IPPO", "MAPPO", "IPPO_POOL", "MAPPO_POOL", "IPPO pixel", "MAPPO pixel"]


def find_latest_run(algo_dir_name):
    """Find the latest TensorBoard run for an algorithm."""
    pattern = f"logs/tensorboard/{algo_dir_name}/*/tensorboard/events.out.*"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    # Group by run directory (parent of events file)
    runs = defaultdict(list)
    for f in files:
        run_dir = os.path.dirname(f)
        runs[run_dir].append(f)
    # Use the run with most events (latest full run)
    latest = max(runs.keys(), key=lambda d: os.path.getmtime(d))
    return sorted(runs[latest])[0]


def load_scalar(event_file, tag, max_points=500):
    """Load a scalar time series from TensorBoard event file."""
    ea = EventAccumulator(event_file, size_guidance={})
    ea.Reload()
    try:
        events = ea.Scalars(tag)
    except KeyError:
        return np.array([]), np.array([])
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    # Downsample if needed
    if len(steps) > max_points:
        idx = np.linspace(0, len(steps) - 1, max_points, dtype=int)
        steps = steps[idx]
        values = values[idx]
    return steps, values


def plot_entropy(ax, models):
    """Entropy over training steps — lower = more deterministic."""
    for name, (algo_dir, label, color, ls) in models.items():
        if "QMIX" in name:
            continue
        f = find_latest_run(algo_dir)
        if not f:
            print(f"  ⚠ No events for {name}")
            continue
        steps, values = load_scalar(f, f"{algo_dir}/entropy")
        if len(steps):
            ax.plot(steps / 1e6, values, color=color, linestyle=ls, linewidth=1.2, label=name)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Entropy")
    ax.legend(fontsize=7)
    ax.set_title("Policy Entropy")


def plot_ep_len(ax, models, max_x=None):
    """Mean episode length over training steps."""
    for name, (algo_dir, label, color, ls) in models.items():
        f = find_latest_run(algo_dir)
        if not f:
            continue
        if "QMIX" in name:
            tag = f"{algo_dir}/eval/avg_episode_length"
        else:
            tag = f"{algo_dir}/ep_len_mean"
        steps, values = load_scalar(f, tag)
        if len(steps):
            ax.plot(steps / 1e6, values, color=color, linestyle=ls, linewidth=1.2, label=name)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Episode Length (steps)")
    ax.legend(fontsize=7)
    ax.set_title("Episode Length (higher = better defense)")


def plot_eval_winrate(ax, models):
    """Team 1 win rate from evaluation."""
    for name, (algo_dir, label, color, ls) in models.items():
        f = find_latest_run(algo_dir)
        if not f:
            continue
        # QMIX uses different eval tag
        algo_key = algo_dir
        steps, wr = load_scalar(f, f"{algo_key}/eval/team_1_winrate")
        if len(steps) == 0:
            steps, wr = load_scalar(f, f"{algo_key}/eval/win_rate")
        if len(steps):
            ax.plot(steps / 1e6, wr * 100, color=color, linestyle=ls, linewidth=1.2, label=name)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("T1 Win Rate (%)")
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Eval: Team 1 Win Rate")


def plot_eval_reward(ax, models):
    """Mean eval reward for Team 1."""
    for name, (algo_dir, label, color, ls) in models.items():
        f = find_latest_run(algo_dir)
        if not f:
            continue
        algo_key = algo_dir
        steps, r = load_scalar(f, f"{algo_key}/eval/avg_reward_team1")
        if len(steps):
            ax.plot(steps / 1e6, r, color=color, linestyle=ls, linewidth=1.2, label=name)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Avg Reward (T1)")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Eval: Team 1 Avg Reward")


def plot_qmix_loss(ax):
    """QMIX q_loss and q_tot."""
    f = find_latest_run("qmix")
    if not f:
        return
    steps, q_loss = load_scalar(f, "qmix/q_loss")
    steps2, q_tot = load_scalar(f, "qmix/q_tot_mean")
    if len(steps):
        ax.plot(steps / 1e6, q_loss, color="#2ca02c", linewidth=1.2, label="q_loss")
    if len(steps2):
        ax2 = ax.twinx()
        ax2.plot(steps2 / 1e6, q_tot, color="red", linewidth=1.2, alpha=0.6, label="q_tot")
        ax2.set_ylabel("Q_tot mean", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Q Loss")
    ax.set_title("QMIX: Q Loss & Q_tot")


def plot_pairwise_comparison(ax, models_a, models_b, metric_tag, ylabel, title,
                              color_a="#1f77b4", color_b="#d62728"):
    """Compare two model groups on the same metric."""
    for name, (algo_dir, label, color, ls) in models_a.items():
        f = find_latest_run(algo_dir)
        if not f:
            continue
        steps, values = load_scalar(f, metric_tag)
        if len(steps):
            ax.plot(steps / 1e6, values, color=color_a, linestyle=ls, linewidth=1.2,
                    label=f"Std {name}", alpha=0.8)
    for name, (algo_dir, label, color, ls) in models_b.items():
        f = find_latest_run(algo_dir)
        if not f:
            continue
        steps, values = load_scalar(f, metric_tag)
        if len(steps):
            ax.plot(steps / 1e6, values, color=color_b, linestyle=ls, linewidth=1.2,
                    label=f"Pool {name}", alpha=0.8)
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/plots/learning_curves.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check available
    available = {}
    for name, (algo_dir, label, color, ls) in MODELS.items():
        f = find_latest_run(algo_dir)
        if f:
            available[name] = (algo_dir, label, color, ls)
        else:
            print(f"  ⚠ {name}: no event file found")
    print(f"Found {len(available)}/7 models with logs")

    # ── Figure 1: Training metrics (5 panels) ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plot_entropy(axes[0, 0], available)
    plot_ep_len(axes[0, 1], available)
    plot_eval_winrate(axes[0, 2], available)
    plot_eval_reward(axes[1, 0], available)
    plot_qmix_loss(axes[1, 1])
    # Panel 6: pool vs standard comparison (entropy)
    std_models = {k: v for k, v in available.items() if k in ["IPPO", "MAPPO"]}
    pool_models = {k: v for k, v in available.items() if k in ["IPPO_POOL", "MAPPO_POOL"]}
    for name, (algo_dir, label, color, ls) in {**std_models, **pool_models}.items():
        f = find_latest_run(algo_dir)
        if not f:
            continue
        steps, values = load_scalar(f, f"{algo_dir}/entropy")
        if len(steps):
            style = "Pool" if "POOL" in name else "Std"
            c = "#d62728" if "POOL" in name else "#1f77b4"
            lw = 1.5 if "POOL" in name else 1.0
            axes[1, 2].plot(steps / 1e6, values, color=c, linewidth=lw,
                           linestyle="-" if "POOL" in name else "--",
                           label=name, alpha=0.8)
    axes[1, 2].set_xlabel("Steps (M)")
    axes[1, 2].set_ylabel("Entropy")
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].set_title("Pool vs Standard: Entropy")

    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nSaved {args.output}")
    plt.close(fig)

    # ── Figure 2: RAM-only comparison (cleaner view) ──
    ram_models = {k: v for k, v in available.items()
                  if v[1] in ("RAM", "RAM+Pool")}
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    plot_entropy(axes2[0], ram_models)
    plot_ep_len(axes2[1], ram_models)
    plot_eval_winrate(axes2[2], ram_models)

    ram_out = args.output.replace(".png", "_ram.png")
    fig2.tight_layout()
    fig2.savefig(ram_out, bbox_inches="tight")
    print(f"Saved {ram_out}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
