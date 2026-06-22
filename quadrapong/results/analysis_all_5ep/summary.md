# 数据分析摘要

数据文件：`results/unified_all_5ep_v2_traj.npz`
规模：56 个有向对局，5 episodes/方向，共 280 局。

## 汇总统计

- 全局 T1 平均胜率：50%；T2 平均胜率：44%；平局率：6%。
- 终止类型：natural=77 (28%), trunc_lead=186 (66%), trunc_tie=17 (6%)。
- 确定性对局：43/56 个有向 matchup 的所有 episodes 终局分数完全一致。

## 模型综合排名

| 模型 | T1 平均胜率 | T2 平均胜率 | 综合胜率 | 综合净胜分 |
|---|---:|---:|---:|---:|
| RANDOM | 89% | 60% | 74% | +95.9 |
| IPPO | 86% | 60% | 73% | +98.9 |
| IPPO_POOL | 80% | 57% | 69% | +92.2 |
| MAPPO_POOL | 54% | 74% | 64% | +61.2 |
| QMIX | 60% | 60% | 60% | +96.5 |
| MAPPO | 29% | 29% | 29% | +58.2 |
| MAPPO_PIXEL | 0% | 14% | 7% | -254.0 |
| IPPO_PIXEL | 0% | 0% | 0% | -249.0 |

## 可直接写进报告的观察

- 对手池对 IPPO 有明确提升：IPPO_POOL 做 T1 胜率 100%，做 T2 胜率 100%，实现双向压制。
- 对手池对 MAPPO 的改善更大：MAPPO_POOL 做 T2 胜率 100%，做 T1 时胜率 0%、平局率 100%；标准 MAPPO 对随机策略也明显失利。
- QMIX 呈现强烈位置不对称：QMIX 作为 T2 对 MAPPO 胜率 100%，但作为 T1 对 MAPPO 的胜率为 0%，且该方向平局率 100%。
- MAPPO 出现训练崩溃迹象：RANDOM 做 T1 对 MAPPO 的胜率 100%，MAPPO 做 T1 对 RANDOM 的胜率仅 0%。
- 像素模型和 QMIX 的负结果应作为 limitation 展开：稀疏奖励、像素 replay buffer 成本和 off-policy bootstrap 都会放大训练难度。

## 输出文件

- `matchups.csv`：所有有向 matchup 的胜率、分数、终止类型。
- `model_summary.csv`：按模型聚合的 T1/T2/综合表现。
- `bidirectional_matrix.csv` 与 `bidirectional_matrix.md`：报告用对抗矩阵。
- `t1_winrate_heatmap.png` 与 `score_diff_heatmap.png`：有向热力图。
