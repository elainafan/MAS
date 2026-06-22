"""Plot evaluation trajectory data from eval_unified.py output.

Reads the _traj.npz file and generates:
- Per-matchup line chart with both directions on same axes (score diff ± std)
- Summary statistics table

Usage:
    python scripts/plot_eval.py results/unified_nonpixel_traj.npz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

GRID_STEP = 1000


def load_traj_data(path):
    try:
        data = np.load(path, allow_pickle=True)["data"].item()
        return data["matchups"], data["episodes"]
    except (KeyError, ValueError, OSError) as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(1)


def align_trajectories(trajs):
    """Align variable-length trajectories to fixed-step grid using LOCF.

    For each grid point g, uses the last recorded point with step <= g.
    This preserves final scores even when episodes end between grid points.

    Returns (grid_steps, scores_t1_mean, scores_t1_std,
             scores_t2_mean, scores_t2_std, alive_frac).
    """
    n_eps = len(trajs)
    if n_eps == 0:
        return [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Derive grid from actual max step in data
    max_step = max(traj[-1][0] for traj in trajs)
    # Round up to nearest GRID_STEP
    grid_end = ((max_step + GRID_STEP - 1) // GRID_STEP) * GRID_STEP
    grid = list(range(0, grid_end + 1, GRID_STEP))

    scores_t1 = np.full((n_eps, len(grid)), np.nan)
    scores_t2 = np.full((n_eps, len(grid)), np.nan)

    for i, traj in enumerate(trajs):
        # Sort by step for LOCF
        steps_arr = np.array([t[0] for t in traj])
        s1_arr = np.array([t[1] for t in traj], dtype=float)
        s2_arr = np.array([t[2] for t in traj], dtype=float)

        # For each grid point, find the last trajectory point with step <= g
        for j, g in enumerate(grid):
            mask = steps_arr <= g
            if mask.any():
                idx = np.where(mask)[0][-1]
                scores_t1[i, j] = s1_arr[idx]
                scores_t2[i, j] = s2_arr[idx]

    mean_t1 = np.nanmean(scores_t1, axis=0)
    std_t1 = np.nanstd(scores_t1, axis=0)
    mean_t2 = np.nanmean(scores_t2, axis=0)
    std_t2 = np.nanstd(scores_t2, axis=0)
    alive = np.sum(~np.isnan(scores_t1), axis=0) / n_eps

    return grid, mean_t1, std_t1, mean_t2, std_t2, alive


def plot_pair(ax, pair_results, pair_key):
    """Plot both directions of a matchup on the same axes."""
    colors = ["#1f77b4", "#d62728"]
    line_styles = ["-", "--"]

    for idx, r in enumerate(pair_results):
        trajs = r["trajectories"]
        if not trajs:
            continue

        grid, m_t1, s_t1, m_t2, s_t2, alive = align_trajectories(trajs)
        if len(grid) == 0:
            continue

        diff_mean = m_t1 - m_t2
        diff_std = np.sqrt(s_t1**2 + s_t2**2)

        label = f"{r['t1_name']}(T1) vs {r['t2_name']}(T2)  WR:{r['t1_wr']:.0%}"
        ax.plot(grid, diff_mean, color=colors[idx], linestyle=line_styles[idx],
                linewidth=1.5, label=label)
        if diff_std.max() > 1e-6:
            ax.fill_between(grid, diff_mean - diff_std, diff_mean + diff_std,
                            color=colors[idx], alpha=0.12)
        else:
            # Mark deterministic
            ax.scatter(grid[-1], diff_mean[-1], color=colors[idx], s=20, zorder=5)

    if all(np.isclose(np.std(diff), 0) for r in pair_results
           for diff in [np.array([t[1] - t[2] for t in r["trajectories"][0]], dtype=float)]
           if r["trajectories"]):
        ax.text(0.5, 0.02, "(deterministic — all episodes identical)",
                transform=ax.transAxes, fontsize=7, ha="center", alpha=0.5)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Score diff (T1 - T2)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"{pair_key}", fontsize=10)

    # Termination summary
    term_info = " | ".join(
        f"{r['t1_name']}→T1: nat={r['term_natural']} lead={r['term_trunc_lead']} tie={r['term_trunc_tie']}"
        for r in pair_results
    )
    ax.text(0.5, -0.12, term_info, transform=ax.transAxes, fontsize=6, ha="center", alpha=0.6)


def plot_all(matchups, episodes, output_dir="results/plots"):
    """Generate one plot per matchup pair with both directions combined."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by unordered pair
    pairs = defaultdict(list)
    for m in matchups:
        key = frozenset([m["t1_name"], m["t2_name"]])
        pairs[key].append(m)

    for i, (pair_key, pair_results) in enumerate(pairs.items()):
        names = sorted(list(pair_key))
        safe_name = f"{names[0]}_vs_{names[1]}".replace("/", "_")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        plot_pair(ax, pair_results, f"{names[0]} vs {names[1]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{safe_name}.png", bbox_inches="tight")
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f"  Plotted {i+1}/{len(pairs)}")

    print(f"Saved {len(pairs)} plots to {output_dir}/")


def print_summary(matchups, episodes):
    """Print summary statistics table."""
    print(f"\n{'='*100}")
    print(f"  EVALUATION SUMMARY ({episodes} episodes per direction)")
    print(f"{'='*100}")
    header = (f"{'T1':<18} {'T2':<18} {'T1_WR':>7} {'T2_WR':>7} {'Draw':>7} "
              f"{'Nat':>6} {'Lead':>6} {'Tie':>6} "
              f"{'T1_score':>14} {'T2_score':>14}")
    print(header)
    print("-" * 100)
    for m in matchups:
        draw = m.get("draw", 1.0 - m["t1_wr"] - m["t2_wr"])
        print(f"{m['t1_name']:<18} {m['t2_name']:<18} "
              f"{m['t1_wr']:7.1%} {m['t2_wr']:7.1%} {draw:7.1%} "
              f"{m['term_natural']:>6} {m['term_trunc_lead']:>6} {m['term_trunc_tie']:>6} "
              f"{m['t1_r']:+10.1f}±{m['t1_r_std']:.1f}  {m['t2_r']:+10.1f}±{m['t2_r_std']:.1f}")
    print("-" * 100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "results/unified_matchup_full_traj.npz"
    else:
        path = sys.argv[1]

    print(f"Loading {path}...")
    matchups, episodes = load_traj_data(path)
    print(f"Loaded {len(matchups)} matchup results, {episodes} episodes each")

    print_summary(matchups, episodes)

    out_dir = Path(path).parent / "plots"
    print(f"\nGenerating plots to {out_dir}...")
    plot_all(matchups, episodes, output_dir=str(out_dir))
    print("Done.")
