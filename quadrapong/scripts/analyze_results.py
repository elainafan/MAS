"""Generate report-ready analysis artifacts from eval_unified trajectory files.

The script reads the ``*_traj.npz`` files produced by ``scripts/eval_unified.py``
and writes compact CSV/Markdown/PNG summaries for the course report.

Usage:
    python scripts/analyze_results.py results/unified_nonpixel_traj.npz
    python scripts/analyze_results.py results/unified_all_5ep_v2_traj.npz \
        --output-dir results/analysis_all
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load_data(path):
    data = np.load(path, allow_pickle=True)["data"].item()
    return data["matchups"], int(data["episodes"])


def final_scores(trajectories):
    scores = []
    for traj in trajectories:
        if not traj:
            continue
        last = traj[-1]
        scores.append((float(last[1]), float(last[2])))
    return np.asarray(scores, dtype=float)


def is_deterministic(trajectories):
    scores = final_scores(trajectories)
    if len(scores) <= 1:
        return True
    return bool(np.allclose(scores, scores[0]))


def build_rows(matchups):
    rows = []
    for m in matchups:
        scores = final_scores(m.get("trajectories", []))
        if len(scores):
            diff = scores[:, 0] - scores[:, 1]
            diff_mean = float(np.mean(diff))
            diff_std = float(np.std(diff))
        else:
            diff_mean = float("nan")
            diff_std = float("nan")

        rows.append({
            "team1": m["t1_name"],
            "team2": m["t2_name"],
            "team1_winrate": float(m["t1_wr"]),
            "team2_winrate": float(m["t2_wr"]),
            "draw_rate": float(m.get("draw", 1.0 - m["t1_wr"] - m["t2_wr"])),
            "team1_score_mean": float(m["t1_r"]),
            "team1_score_std": float(m["t1_r_std"]),
            "team2_score_mean": float(m["t2_r"]),
            "team2_score_std": float(m["t2_r_std"]),
            "score_diff_mean": diff_mean,
            "score_diff_std": diff_std,
            "natural": int(m["term_natural"]),
            "trunc_lead": int(m["term_trunc_lead"]),
            "trunc_tie": int(m["term_trunc_tie"]),
            "deterministic": is_deterministic(m.get("trajectories", [])),
        })
    return rows


def ordered_models(matchups):
    models = []
    for m in matchups:
        for name in (m["t1_name"], m["t2_name"]):
            if name not in models:
                models.append(name)
    return models


def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.0f}%"


def draw_rate(m):
    return float(m.get("draw", 1.0 - m["t1_wr"] - m["t2_wr"]))


def write_matchup_csv(rows, out_path):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(matchups, models, out_path):
    by_pair = {(m["t1_name"], m["t2_name"]): m for m in matchups}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["A \\ B"] + models)
        for a in models:
            row = [a]
            for b in models:
                if a == b:
                    row.append("-")
                    continue
                a_t1 = by_pair.get((a, b))
                a_t2 = by_pair.get((b, a))
                t1_wr = a_t1["t1_wr"] if a_t1 else None
                t2_wr = a_t2["t2_wr"] if a_t2 else None
                row.append(f"{pct(t1_wr)} / {pct(t2_wr)}")
            writer.writerow(row)


def write_matrix_md(matchups, models, out_path):
    by_pair = {(m["t1_name"], m["t2_name"]): m for m in matchups}
    lines = []
    lines.append("# 对抗矩阵")
    lines.append("")
    lines.append("每格格式为 `A 做 T1 胜率 / A 做 T2 胜率`。")
    lines.append("")
    header = "| A \\ B | " + " | ".join(models) + " |"
    sep = "|" + "---|" * (len(models) + 1)
    lines.extend([header, sep])
    for a in models:
        cells = []
        for b in models:
            if a == b:
                cells.append("-")
                continue
            a_t1 = by_pair.get((a, b))
            a_t2 = by_pair.get((b, a))
            t1_wr = a_t1["t1_wr"] if a_t1 else None
            t2_wr = a_t2["t2_wr"] if a_t2 else None
            cells.append(f"{pct(t1_wr)} / {pct(t2_wr)}")
        lines.append("| " + a + " | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_summary(rows, models):
    summary = []
    for model in models:
        as_t1 = [r for r in rows if r["team1"] == model]
        as_t2 = [r for r in rows if r["team2"] == model]
        t1_wr = np.mean([r["team1_winrate"] for r in as_t1]) if as_t1 else np.nan
        t2_wr = np.mean([r["team2_winrate"] for r in as_t2]) if as_t2 else np.nan
        t1_diff = np.mean([r["score_diff_mean"] for r in as_t1]) if as_t1 else np.nan
        t2_diff = np.mean([-r["score_diff_mean"] for r in as_t2]) if as_t2 else np.nan
        combined_wr = np.nanmean([t1_wr, t2_wr])
        combined_diff = np.nanmean([t1_diff, t2_diff])
        summary.append({
            "model": model,
            "avg_winrate_as_t1": t1_wr,
            "avg_winrate_as_t2": t2_wr,
            "avg_winrate_overall": combined_wr,
            "avg_score_diff_as_t1": t1_diff,
            "avg_score_diff_as_t2": t2_diff,
            "avg_score_diff_overall": combined_diff,
        })
    summary.sort(key=lambda x: (x["avg_winrate_overall"], x["avg_score_diff_overall"]), reverse=True)
    return summary


def write_model_summary_csv(summary, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_heatmap(matchups, models, out_path, mode="winrate"):
    by_pair = {(m["t1_name"], m["t2_name"]): m for m in matchups}
    matrix = np.full((len(models), len(models)), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if a == b:
                continue
            m = by_pair.get((a, b))
            if not m:
                continue
            if mode == "winrate":
                matrix[i, j] = m["t1_wr"]
            else:
                matrix[i, j] = m["t1_r"] - m["t2_r"]

    fig, ax = plt.subplots(figsize=(8, 6))
    if mode == "winrate":
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn")
        cbar_label = "Team 1 win rate"
        fmt_cell = lambda v: pct(v)
    else:
        max_abs = np.nanmax(np.abs(matrix)) if np.any(~np.isnan(matrix)) else 1.0
        im = ax.imshow(matrix, vmin=-max_abs, vmax=max_abs, cmap="coolwarm")
        cbar_label = "Score diff (T1 - T2)"
        fmt_cell = lambda v: f"{v:+.0f}"
    fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    ax.set_xticks(range(len(models)), labels=models, rotation=35, ha="right")
    ax.set_yticks(range(len(models)), labels=models)
    ax.set_xlabel("Team 2 model")
    ax.set_ylabel("Team 1 model")
    ax.set_title("Directed matchup " + ("win rate" if mode == "winrate" else "score diff"))
    for i in range(len(models)):
        for j in range(len(models)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt_cell(v), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def lookup(matchups, t1, t2):
    for m in matchups:
        if m["t1_name"] == t1 and m["t2_name"] == t2:
            return m
    return None


def write_summary_md(matchups, rows, models, summary, episodes, source_path, out_path):
    total = len(rows) * episodes
    natural = sum(r["natural"] for r in rows)
    trunc_lead = sum(r["trunc_lead"] for r in rows)
    trunc_tie = sum(r["trunc_tie"] for r in rows)
    directed_t1_wr = np.mean([r["team1_winrate"] for r in rows])
    directed_t2_wr = np.mean([r["team2_winrate"] for r in rows])
    directed_draw = np.mean([r["draw_rate"] for r in rows])
    deterministic = sum(1 for r in rows if r["deterministic"])

    lines = []
    lines.append("# 数据分析摘要")
    lines.append("")
    lines.append(f"数据文件：`{source_path}`")
    lines.append(f"规模：{len(rows)} 个有向对局，{episodes} episodes/方向，共 {total} 局。")
    lines.append("")
    lines.append("## 汇总统计")
    lines.append("")
    lines.append(f"- 全局 T1 平均胜率：{pct(directed_t1_wr)}；T2 平均胜率：{pct(directed_t2_wr)}；平局率：{pct(directed_draw)}。")
    lines.append(f"- 终止类型：natural={natural} ({pct(natural / total)}), trunc_lead={trunc_lead} ({pct(trunc_lead / total)}), trunc_tie={trunc_tie} ({pct(trunc_tie / total)})。")
    lines.append(f"- 确定性对局：{deterministic}/{len(rows)} 个有向 matchup 的所有 episodes 终局分数完全一致。")
    lines.append("")
    lines.append("## 模型综合排名")
    lines.append("")
    lines.append("| 模型 | T1 平均胜率 | T2 平均胜率 | 综合胜率 | 综合净胜分 |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summary:
        lines.append(
            f"| {s['model']} | {pct(s['avg_winrate_as_t1'])} | {pct(s['avg_winrate_as_t2'])} | "
            f"{pct(s['avg_winrate_overall'])} | {s['avg_score_diff_overall']:+.1f} |"
        )

    lines.append("")
    lines.append("## 可直接写进报告的观察")
    lines.append("")
    ippo_pool_t1 = lookup(matchups, "IPPO_POOL", "IPPO")
    ippo_pool_t2 = lookup(matchups, "IPPO", "IPPO_POOL")
    if ippo_pool_t1 and ippo_pool_t2:
        lines.append(
            f"- 对手池对 IPPO 有明确提升：IPPO_POOL 做 T1 胜率 {pct(ippo_pool_t1['t1_wr'])}，"
            f"做 T2 胜率 {pct(ippo_pool_t2['t2_wr'])}，实现双向压制。"
        )
    mappo_pool_t1 = lookup(matchups, "MAPPO_POOL", "MAPPO")
    mappo_pool_t2 = lookup(matchups, "MAPPO", "MAPPO_POOL")
    if mappo_pool_t1 and mappo_pool_t2:
        mappo_pool_t1_draw = draw_rate(mappo_pool_t1)
        lines.append(
            f"- 对手池对 MAPPO 的改善更大：MAPPO_POOL 做 T2 胜率 {pct(mappo_pool_t2['t2_wr'])}，"
            f"做 T1 时胜率 {pct(mappo_pool_t1['t1_wr'])}、平局率 {pct(mappo_pool_t1_draw)}；标准 MAPPO 对随机策略也明显失利。"
        )
    qmix_t2 = lookup(matchups, "MAPPO", "QMIX")
    qmix_t1 = lookup(matchups, "QMIX", "MAPPO")
    if qmix_t2 and qmix_t1:
        qmix_t1_draw = draw_rate(qmix_t1)
        lines.append(
            f"- QMIX 呈现强烈位置不对称：QMIX 作为 T2 对 MAPPO 胜率 {pct(qmix_t2['t2_wr'])}，"
            f"但作为 T1 对 MAPPO 的胜率为 {pct(qmix_t1['t1_wr'])}，且该方向平局率 {pct(qmix_t1_draw)}。"
        )
    random_vs_mappo = lookup(matchups, "RANDOM", "MAPPO")
    mappo_vs_random = lookup(matchups, "MAPPO", "RANDOM")
    if random_vs_mappo and mappo_vs_random:
        lines.append(
            f"- MAPPO 出现训练崩溃迹象：RANDOM 做 T1 对 MAPPO 的胜率 {pct(random_vs_mappo['t1_wr'])}，"
            f"MAPPO 做 T1 对 RANDOM 的胜率仅 {pct(mappo_vs_random['t1_wr'])}。"
        )
    lines.append("- 像素模型和 QMIX 的负结果应作为 limitation 展开：稀疏奖励、像素 replay buffer 成本和 off-policy bootstrap 都会放大训练难度。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `matchups.csv`：所有有向 matchup 的胜率、分数、终止类型。")
    lines.append("- `model_summary.csv`：按模型聚合的 T1/T2/综合表现。")
    lines.append("- `bidirectional_matrix.csv` 与 `bidirectional_matrix.md`：报告用对抗矩阵。")
    lines.append("- `t1_winrate_heatmap.png` 与 `score_diff_heatmap.png`：有向热力图。")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_unified trajectory output")
    parser.add_argument("traj", help="Path to *_traj.npz")
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()

    matchups, episodes = load_data(args.traj)
    rows = build_rows(matchups)
    models = ordered_models(matchups)
    summary = model_summary(rows, models)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_matchup_csv(rows, out_dir / "matchups.csv")
    write_model_summary_csv(summary, out_dir / "model_summary.csv")
    write_matrix_csv(matchups, models, out_dir / "bidirectional_matrix.csv")
    write_matrix_md(matchups, models, out_dir / "bidirectional_matrix.md")
    plot_heatmap(matchups, models, out_dir / "t1_winrate_heatmap.png", mode="winrate")
    plot_heatmap(matchups, models, out_dir / "score_diff_heatmap.png", mode="score")
    write_summary_md(matchups, rows, models, summary, episodes, args.traj, out_dir / "summary.md")

    print(f"Loaded {len(matchups)} directed matchups, {episodes} episodes each")
    print(f"Saved analysis artifacts to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        raise
