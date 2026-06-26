# 给论文撰写同学的交接说明

这份说明面向论文撰写。数据分析和 demo 录制部分已经整理完毕，可以直接基于当前 GitHub `main` 分支写结果分析与展示部分。

## 先看这些文件

- `report/RESULTS_AND_DEMO_DRAFT.md`：结果分析段落草稿，可以直接改写进论文。
- `report/DATA_ANALYSIS_AND_DEMO.md`：数据分析与 demo 的工作台说明。
- `results/analysis/summary.md`：自动生成的数据分析摘要。
- `results/analysis/bidirectional_matrix.md`：双向对抗矩阵，每格表示 `A 做 T1 胜率 / A 做 T2 胜率`。
- `results/analysis/matchups.csv`：所有有向 matchup 明细。
- `results/analysis/model_summary.csv`：按模型聚合后的 T1/T2/综合表现。
- `results/analysis/t1_winrate_heatmap.png`、`results/analysis/score_diff_heatmap.png`：报告可用热力图。
- `results/videos/`：四个 demo mp4 及对应 JSON 元数据。

## 推荐写进论文的结论

1. 对手池自博弈是最清晰的创新点证据。`IPPO_POOL` 对 `IPPO` 呈现双向优势，`MAPPO_POOL` 相对标准 `MAPPO` 也有明显改善。
2. Quadrapong 的 T1/T2 不是简单可互换阵营。训练中会形成角色化策略，当前定性观察是 Team 2 更偏防守/守成，所以论文里要把算法能力和 side assignment 分开讨论。
3. `MAPPO` checkpoint 出现明显 collapse，甚至会输给随机策略。这个可以作为多智能体自博弈训练不稳定性的负结果分析。
4. `QMIX` 的表现高度依赖位置/角色。它作为 T2 对 `MAPPO` 很强，但反向不代表同样强，不能写成全局最强算法。
5. pixel 模型和 QMIX 的负结果建议诚实写成 limitation：像素观测提高样本复杂度，稀疏奖励和 off-policy bootstrap 会放大训练难度。

## Demo 讲解顺序

1. `ippo_vs_random.mp4`：IPPO baseline 已经学到基本击球和得分能力。
2. `ippo_pool_vs_ippo.mp4`：展示 opponent pool 对标准 IPPO 的提升。
3. `mappo_vs_qmix.mp4`：展示 QMIX 作为 T2 在防守/反击角色下对 MAPPO 的优势。
4. `random_vs_mappo.mp4`：展示标准 MAPPO 的 collapse，用作负结果说明。

视频顶部的 `return +x:-x` 是 Team1/Team2 累计 reward，不是 Atari 画面自带的大号比分。论文和展示里建议统一称为 return。

## 表述时请注意

- 不要只引用单方向胜率；优先引用双向评估或明确写清楚 `T1/T2`。
- 不要把 `QMIX as T2` 的强表现写成 QMIX 全局胜出。
- 不要把 `RANDOM` 的局部胜利解释成随机策略真的更聪明，它主要暴露了环境位置效应和部分训练策略 collapse。
- `results/video_frames/` 是本地抽帧检查目录，不是正式交付内容；正式 demo 以 `results/videos/` 为准。
