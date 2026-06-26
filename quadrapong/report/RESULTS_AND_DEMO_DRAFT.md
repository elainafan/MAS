# Results and Demo Draft

本文档整理数据分析与 Demo 录制部分的报告草稿，可直接拆分到报告的 `results`、`innovation` 和 demo 讲解稿中。

## 1. 数据来源

- 主评估：`quadrapong/results/unified_nonpixel.txt`
- 主评估轨迹：`quadrapong/results/unified_nonpixel_traj.npz`
- 分析摘要：`quadrapong/results/analysis/summary.md`
- 对抗矩阵：`quadrapong/results/analysis/bidirectional_matrix.md`
- 热力图：`quadrapong/results/analysis/t1_winrate_heatmap.png`、`quadrapong/results/analysis/score_diff_heatmap.png`
- 学习曲线：`quadrapong/results/plots/learning_curves.png`、`quadrapong/results/plots/learning_curves_ram.png`
- Demo 视频：`quadrapong/results/videos/`

主评估覆盖 6 个非 pixel 策略实体：IPPO、MAPPO、QMIX、IPPO_POOL、MAPPO_POOL、RANDOM。评估为 30 个有向对局，每个方向 50 局，共 1500 局。

## 2. 核心统计

- 全局 T1 平均胜率为 52%，T2 平均胜率为 40%，平局率为 8%。
- 终止类型为 natural 778 局、trunc_lead 608 局、trunc_tie 114 局。
- 20/30 个有向 matchup 的终局分数在 50 个 episode 中完全一致，说明确定性策略之间的对局随机性很低。
- 综合胜率排序为 IPPO、IPPO_POOL、MAPPO_POOL、RANDOM、QMIX、MAPPO，但该排序必须结合位置效应解释，不能简单等同于算法绝对强弱。

## 3. 报告可写结论

### 3.1 对手池自博弈有效缓解位置固化

标准 IPPO 在自博弈中容易形成固定角色分工，导致策略对 T1/T2 初始位置非常敏感。加入 opponent pool 后，IPPO_POOL 对 IPPO 呈现双向优势：IPPO_POOL 做 T1 时胜率 100%，做 T2 时胜率 100%。这说明历史策略、随机策略和当前策略混合采样能提升对不同对手和位置的泛化能力。

MAPPO_POOL 相对标准 MAPPO 也有明显改善。尤其当 MAPPO_POOL 作为 T2 对战 MAPPO 时胜率达到 100%；反向作为 T1 时虽然没有取胜，但能稳定守成平局。这个结果比标准 MAPPO 对随机策略也频繁失利的表现强很多。

### 3.2 Quadrapong 中位置效应很强

随机策略作为 T1 时，对多个训练模型都有较高胜率，例如 RANDOM(T1) vs MAPPO(T2) 胜率为 94%。这说明 Quadrapong 的 spawn 和发球机制会显著影响对局结果，报告中需要把“算法能力”和“side assignment”分开讨论。

因此我们采用双向评估：每一对模型都交换 T1/T2 位置各评估 50 局。报告中的对抗矩阵每格写作 `A 做 T1 胜率 / A 做 T2 胜率`，避免单方向结果误导。

同时，T1/T2 训练出的策略不能简单理解为同一策略的镜像换边。根据训练观察，Team 2 更偏防守/守成，因此同一算法在 T1 与 T2 的表现差异既包含 spawn/发球机制影响，也包含训练过程中形成的角色化策略。

### 3.3 MAPPO 出现 collapse

标准 MAPPO checkpoint 表现最弱。MAPPO(T1) vs RANDOM(T2) 胜率仅 2%，RANDOM(T1) vs MAPPO(T2) 胜率达到 94%。这可以作为多智能体自博弈训练不稳定的负例：集中式 critic 并不必然带来更强策略，若训练动态 collapse，最终策略甚至会弱于随机 baseline。

### 3.4 QMIX 呈现强烈位置不对称

QMIX 的表现高度依赖位置。MAPPO(T1) vs QMIX(T2) 中，QMIX 以 100% 胜率和 +42:-42 的 return 优势获胜；但 QMIX(T1) vs MAPPO(T2) 时胜率为 0%，平局率为 100%。这说明 QMIX 学到的策略可能更适配某个固定位置/发球模式和 T2 防守/守成角色，而不是均衡的双边策略。

### 3.5 Pixel 与 QMIX 负结果需要诚实讨论

Pixel 模型整体没有形成有效策略，QMIX pixel 训练也因为 replay buffer 成本和稀疏奖励问题提前终止。报告中建议把这部分作为 limitation，而不是弱化处理：像素观测显著增加样本复杂度，off-policy TD bootstrap 在稀疏奖励下更容易被大量零奖励 transition 主导。

## 4. Demo 视频清单

| 视频 | 目的 | 终局 |
|---|---|---|
| `ippo_vs_random.mp4` | 展示基础 IPPO 已学到有效策略 | IPPO 胜，+26:-26，12350 步 |
| `ippo_pool_vs_ippo.mp4` | 展示 opponent pool 创新点 | IPPO_POOL 胜，+4:-4，20000 步 |
| `mappo_vs_qmix.mp4` | 展示 QMIX 作为 T2 的防守/反击优势 | QMIX 胜，-42:+42，10860 步 |
| `random_vs_mappo.mp4` | 展示标准 MAPPO collapse | RANDOM 胜，+16:-16，20000 步 |

录屏文件已经在 `quadrapong/results/videos/` 下，每个 mp4 旁边都有同名 JSON 元数据。

本地 Lenovo 当前可以直接播放和剪辑这些 mp4；若要本地重新录制，需要先安装 `gymnasium`、`pettingzoo`、`multi-agent-ale-py`、`torch`、`opencv-python`、`imageio` 等运行依赖。为减少环境折腾，重新录制建议继续在远程训练机的 `cv-lab3` 环境中完成。

## 5. Demo 讲解顺序

1. 先放 `ippo_vs_random.mp4`：说明 IPPO baseline 至少学到了稳定击球和得分能力。
2. 再放 `ippo_pool_vs_ippo.mp4`：强调创新点不是新网络，而是训练分布改造；opponent pool 打破标准自博弈的位置固化。
3. 放 `mappo_vs_qmix.mp4`：展示 QMIX 在特定 side assignment 下可以极强，但这不是全局胜利，需要结合反向评估解释。
4. 最后放 `random_vs_mappo.mp4`：用随机策略击败 MAPPO 的例子说明负结果和训练 collapse，也是报告分析的一部分。

## 6. 可直接复现命令

重新生成主分析产物：

```bash
python quadrapong/scripts/analyze_results.py \
  quadrapong/results/unified_nonpixel_traj.npz \
  --output-dir quadrapong/results/analysis
```

重新生成含 pixel 的 5ep 分析产物：

```bash
python quadrapong/scripts/analyze_results.py \
  quadrapong/results/unified_all_5ep_v2_traj.npz \
  --output-dir quadrapong/results/analysis_all_5ep
```

在远程训练环境中重新录制 Demo：

```bash
cd /workspace/MAS/quadrapong
source /home/claude/miniconda3/etc/profile.d/conda.sh
conda activate cv-lab3
python scripts/record_demo.py --preset all --device cuda:0
```
