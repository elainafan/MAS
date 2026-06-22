# 数据分析与 Demo 录制交接

本文档是我们负责的“数据分析 + Demo 录制”部分的工作台。模型训练、统一评估和基础图表已经由前序工作完成，当前继续基于以下产物整理报告材料与演示视频。

## 当前可用产物

- 评估结果：`results/unified_nonpixel.txt`、`results/unified_nonpixel_traj.npz`
- 全模型含 pixel 的小规模评估：`results/unified_all_5ep_v2.txt`、`results/unified_all_5ep_v2_traj.npz`
- 对抗曲线图：`results/plots/*_vs_*.png`
- 学习曲线：`results/plots/learning_curves.png`、`results/plots/learning_curves_ram.png`
- 最终模型：`checkpoints/ippo/*_final.pt`、`checkpoints/mappo/*_final.pt`、`checkpoints/qmix/*_final.pt`、`checkpoints/ippo_pool/*_final.pt`、`checkpoints/mappo_pool/*_final.pt`

## 数据分析

生成报告可用的 CSV、Markdown 表格和热力图：

```bash
source /home/claude/miniconda3/etc/profile.d/conda.sh
conda activate cv-lab3
cd /workspace/MAS/quadrapong

python scripts/analyze_results.py results/unified_nonpixel_traj.npz \
  --output-dir results/analysis
```

输出文件：

- `results/analysis/summary.md`：中文摘要，可直接迁移到报告结果分析章节。
- `results/analysis/bidirectional_matrix.md`：对抗矩阵，每格为 `A 做 T1 胜率 / A 做 T2 胜率`。
- `results/analysis/matchups.csv`：有向对局明细。
- `results/analysis/model_summary.csv`：模型级聚合指标。
- `results/analysis/t1_winrate_heatmap.png`：T1 有向胜率热力图。
- `results/analysis/score_diff_heatmap.png`：T1-T2 分差热力图。

若需要把 pixel 负结果也整理成同样格式：

```bash
python scripts/analyze_results.py results/unified_all_5ep_v2_traj.npz \
  --output-dir results/analysis_all_5ep
```

## 报告中建议强调的结论

1. 对手池自博弈是最清晰的创新点证据：`IPPO_POOL` 双向压制 `IPPO`，`MAPPO_POOL` 相对 `MAPPO` 提升更明显。
2. Quadrapong 的初始位置效应非常强，T1/T2 位置差异经常压过算法差异；分析时要把“算法强弱”和“side assignment”分开讨论。
3. 标准 `MAPPO` checkpoint 出现明显 collapse：对随机策略也经常失利，可作为自博弈不稳定性的反例。
4. `QMIX` 在该环境中呈现强烈位置不对称和稀疏奖励困难，适合作为负结果讨论，而不是回避。
5. pixel 模型失败的原因应诚实写成 limitation：像素观测增大样本复杂度，QMIX pixel 的 replay buffer 成本尤其高。

## Demo 录制

推荐一键录制 4 个代表性视频：

```bash
source /home/claude/miniconda3/etc/profile.d/conda.sh
conda activate cv-lab3
cd /workspace/MAS/quadrapong

python scripts/record_demo.py --preset all --device cuda:0
```

默认输出到 `results/videos/`，每个 `.mp4` 旁边会生成同名 `.json`，记录 seed、步数、终局比分和 winner。

预设视频：

- `ippo_vs_random.mp4`：展示基础 IPPO 已经学到有效策略。
- `ippo_pool_vs_ippo.mp4`：展示 opponent pool 的创新点效果。
- `mappo_vs_qmix.mp4`：展示 QMIX 作为 T2 对 MAPPO 的统治性结果。
- `random_vs_mappo.mp4`：展示标准 MAPPO checkpoint 的 collapse。

单独录制示例：

```bash
python scripts/record_demo.py --preset ippo_pool_vs_ippo --device cuda:0
python scripts/record_demo.py --team1 checkpoints/ippo/ippo_final.pt --team2 random \
  --name ippo_vs_random_short --max-steps 8000 --frame-stride 6
```

