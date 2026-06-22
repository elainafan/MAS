# 数据分析摘要

数据文件：`results/unified_nonpixel_traj.npz`
规模：30 个有向对局，50 episodes/方向，共 1500 局。

## 汇总统计

- 全局 T1 平均胜率：52%；T2 平均胜率：40%；平局率：8%。
- 终止类型：natural=778 (52%), trunc_lead=608 (41%), trunc_tie=114 (8%)。
- 确定性对局：20/30 个有向 matchup 的所有 episodes 终局分数完全一致。

## 模型综合排名

| 模型 | T1 平均胜率 | T2 平均胜率 | 综合胜率 | 综合净胜分 |
|---|---:|---:|---:|---:|
| IPPO | 79% | 48% | 64% | +12.8 |
| IPPO_POOL | 78% | 44% | 61% | +5.4 |
| MAPPO_POOL | 36% | 69% | 53% | +10.7 |
| RANDOM | 69% | 33% | 51% | +5.5 |
| QMIX | 51% | 46% | 49% | +11.5 |
| MAPPO | 0% | 1% | 1% | -45.9 |

## 可直接写进报告的观察

- 对手池对 IPPO 有明确提升：IPPO_POOL 做 T1 胜率 100%，做 T2 胜率 100%，实现双向压制。
- 对手池对 MAPPO 的改善更大：MAPPO_POOL 做 T2 胜率 100%，做 T1 时胜率 0%、平局率 100%；标准 MAPPO 对随机策略也明显失利。
- QMIX 呈现强烈位置不对称：QMIX 作为 T2 对 MAPPO 胜率 100%，但作为 T1 对 MAPPO 的胜率为 0%，且该方向平局率 100%。
- MAPPO 出现训练崩溃迹象：RANDOM 做 T1 对 MAPPO 的胜率 94%，MAPPO 做 T1 对 RANDOM 的胜率仅 2%。
- 像素模型和 QMIX 的负结果应作为 limitation 展开：稀疏奖励、像素 replay buffer 成本和 off-policy bootstrap 都会放大训练难度。

## 输出文件

- `matchups.csv`：所有有向 matchup 的胜率、分数、终止类型。
- `model_summary.csv`：按模型聚合的 T1/T2/综合表现。
- `bidirectional_matrix.csv` 与 `bidirectional_matrix.md`：报告用对抗矩阵。
- `t1_winrate_heatmap.png` 与 `score_diff_heatmap.png`：有向热力图。
