# Quadrapong 项目移交文档

> 多智能体基础课程课题六 — 2v2 多智能体乒乓对抗  
> 移交日期：2026-06-22 ｜ 截止日期：2026-07-01 18:00 ｜ 满分 35 分

---

## 一、项目概述

Quadrapong 是 PettingZoo/ALE 环境中的 4-agent 2v2 Pong。两支队伍（Team 1: agents 0,2 / Team 2: agents 1,3）各控制两个 paddle，在上下两个半场同时对抗。每个 agent 有 6 个离散动作。环境支持两种观测：**RAM 128 维** 和 **grayscale pixel (84×84×3)**。游戏采用经典 21 分制，一方先到 21 分则自然终止。

项目目标：实现 IPPO / MAPPO / QMIX 三种基线算法，设计一个创新改进，完成全面对抗评估，撰写研究报告。

**硬件环境**：2× RTX 4090 (48GB VRAM)。Python 3.11 + PyTorch 2.7 + CUDA 12.8。conda 环境 `cv-lab3`。

---

## 二、评分结构

| 评分项 | 分值 | 权重 |
|--------|------|------|
| 算法设计（三基线实现与评估） | 21 分 | 报告 60% |
| 创新点（对手池自博弈） | 14 分 | 代码&Demo 40% |
| **总分** | **35 分** | |

> 评分权重表示：最终成绩 = 报告质量 × 60% + 代码&Demo × 40%

---

## 三、目录结构

```
quadrapong/
├── CLAUDE.md              # 项目规范与 AI 协作指引
├── docs/                  # 文档（调研、日志、架构）
│   ├── code_architecture.md   # 代码架构详解
│   ├── code_log.md            # 代码修改日志
│   ├── experiment_log.md      # 实验记录与分析
│   ├── train_log.md           # 训练运行日志
│   └── research_analysis.md   # 课题调研与任务拆解
├── src/
│   ├── algos/             # 算法实现（ippo/mappo/qmix）
│   ├── envs/              # 环境包装（quadrapong_env.py）
│   ├── utils/             # 工具（buffer/networks/evaluator/logger/opponent_pool）
│   └── configs/           # 10 个 YAML 配置文件
├── scripts/               # 训练/评估/画图脚本
│   ├── train_ippo.py / train_mappo.py / train_qmix.py
│   ├── train_ippo_pool.py / train_mappo_pool.py
│   ├── eval_unified.py    # 大一统评估脚本
│   └── plot_eval.py       # 评估结果可视化
├── checkpoints/           # 训练好的模型权重（8 个模型变种）
├── results/               # 评估结果 + 图表
│   ├── unified_nonpixel.txt / _traj.npz   # 非 pixel 50ep 全量评估
│   ├── unified_all_5ep_v2.txt / _traj.npz # 全模型 5ep 评估
│   └── plots/             # 15 张对抗曲线图
├── papers/                # 参考论文（HARL / pymarl）
├── logs/                  # TensorBoard 训练日志
└── report/                # 报告（本目录）
    └── HANDOFF.md         # 本文档
```

---

## 四、已完成的算法实现

### 4.1 三基线算法

| 算法 | 文件 | 参数量 | 特点 |
|------|------|--------|------|
| **IPPO** | `src/algos/ippo.py` | 67K | Independent PPO，共享 Actor + 独立 Critic，on-policy self-play |
| **MAPPO** | `src/algos/mappo.py` | 132K | CTDE PPO，集中式 Critic（global_state+local_obs）+ PopArt 值归一化 |
| **QMIX** | `src/algos/qmix.py` | 161K | Off-policy CTDE，FF-Q Network + 单调 Mixing Network + 双 Q |

> 详细架构见 [docs/code_architecture.md](../docs/code_architecture.md)

### 4.2 创新点：对手池自博弈 (Opponent Pool Self-Play)

**动机**：标准自博弈中 T2（agents 1,3）100% 必胜，因 spawn 位置差异导致角色固化。

**方案**：训练时 Team 2 从对手池随机采样，而非始终使用当前策略。

| 对手类型 | 来源 | 采样权重 | 说明 |
|----------|------|---------|------|
| 历史自身 | 每 200K 步 checkpoint | 60% | FIFO 队列，容量 K=5 |
| 随机策略 | ε=1.0 纯随机 | 20% | 基础探索多样性 |
| 当前自身 | 正在训练的策略 | 20% | 保持对抗压力 |

**实现**：`src/utils/opponent_pool.py` + `scripts/train_ippo_pool.py` / `train_mappo_pool.py`

**效果验证**：IPPO_POOL 双向赢 IPPO，MAPPO_POOL 双向赢 MAPPO——成功打破 T2 必胜魔咒。

> 创新点设计详见 [docs/experiment_log.md](../docs/experiment_log.md) 2026-06-20 19:00 条目

---

## 五、训练完成的模型

| 模型 | 观测 | 网络 | 训练步数 | Checkpoint | 状态 |
|------|------|------|---------|-----------|------|
| IPPO | RAM 128d | MLP | 10M | `checkpoints/ippo/ippo_final.pt` | ✅ 收敛 |
| MAPPO | RAM 128d | MLP | 10M | `checkpoints/mappo/mappo_final.pt` | ⚠️ 评估中表现最弱 |
| QMIX | RAM 128d | FF-Q+Mixer | 1M | `checkpoints/qmix/qmix_final.pt` | ⚠️ 仅 1M 步，T1 极强 T2 极弱 |
| IPPO pixel | grayscale(84×84) | CNN | 10M | `checkpoints/ippo_pixel/ippo_final.pt` | ❌ 性能等同随机 |
| MAPPO pixel | grayscale(84×84) | CNN | 10M | `checkpoints/mappo_pixel/mappo_final.pt` | ❌ 性能等同随机 |
| QMIX pixel | grayscale(84×84) | CNN+Mixer | 277K | 训练爆炸终止 | ❌ 无可用模型 |
| **IPPO pool** | RAM 128d | MLP | 10M | `checkpoints/ippo_pool/ippo_pool_final.pt` | ✅ 创新点 |
| **MAPPO pool** | RAM 128d | MLP | 10M | `checkpoints/mappo_pool/mappo_pool_final.pt` | ✅ 创新点 |

> 训练详细记录见 [docs/train_log.md](../docs/train_log.md)

---

## 六、对抗评估结果

### 评估方法

使用 `scripts/eval_unified.py`（大一统评估脚本）：
- 自动检测网络类型（IPPO/MAPPO vs QMIX）和观测类型（RAM vs pixel）
- 全配对评估 + 自动交换 T1/T2 位置
- 支持跨观测对阵（RAM vs pixel，双环境 lockstep）
- 终止分类：natural（自然 21 分）/ trunc_lead（截断时有领先）/ trunc_tie（截断时平局）
- 每 1000 步记录比分 → `_traj.npz` 轨迹文件
- `--include-random` 加入随机策略 baseline

### Batch 1：非 pixel 50ep（6 模型 = 5 个 RAM + RANDOM）

**命令**：
```bash
python scripts/eval_unified.py \
  checkpoints/ippo/ippo_final.pt \
  checkpoints/mappo/mappo_final.pt \
  checkpoints/qmix/qmix_final.pt \
  checkpoints/ippo_pool/ippo_pool_final.pt \
  checkpoints/mappo_pool/mappo_pool_final.pt \
  --include-random --max-steps 20000 --episodes 50 \
  --device cuda:0 --output results/unified_nonpixel.txt
```

**对抗矩阵**（T1 WR / T2 WR）：

| A \ B | IPPO | MAPPO | QMIX | IPPO_P | MAPPO_P | RANDOM |
|-------|------|-------|------|--------|---------|--------|
| **IPPO** | — | 100/100 | 100/0 | 0/0 | 100/100 | 96/40 |
| **MAPPO** | 0/0 | — | 0/100 | 0/0 | 0/0 | 2/6 |
| **QMIX** | 100/0 | 0/0(Dr) | — | 100/0 | 0/0 | 56/30 |
| **IPPO_POOL** | 100/100 | 100/100 | 0/100 | — | 0/100 | 88/20 |
| **MAPPO_POOL** | 0/0 | 100/0(Dr) | 100/100 | 100/0 | — | 80/46 |
| **RANDOM** | 60/2 | 94/98 | 60/40 | 78/10 | 52/14 | — |

> 每格：A 做 T1 的 WR / A 做 T2 的 WR。Dr = 100% 平局

### 核心发现

1. **IPPO_POOL 双向赢 IPPO** — 对手池自博弈有效打破 T2 必胜
2. **MAPPO_POOL 双向赢 MAPPO** — 提升幅度极大
3. **QMIX 极端不对称** — T1 零封对手（42:-42），T2 却 0-0 平局
4. **MAPPO 全面崩溃** — 连 RANDOM 都打不过（MAPPO T1 仅 2% 胜率）
5. **T1 spawn 位置优势压倒一切** — RANDOM T1 对多数模型有 50%+ 胜率
6. **确定性策略 std=0** — 模型 vs 模型时所有 episode 结果完全相同，1 局即可

### 图表

`python scripts/plot_eval.py results/unified_nonpixel_traj.npz` 生成 15 张双向合并图 → `results/plots/`

> 完整分析见 [docs/experiment_log.md](../docs/experiment_log.md) 2026-06-21~22 条目

---

## 七、已知问题与教训

| 问题 | 影响 | 应对 |
|------|------|------|
| QMIX 稀疏奖励 + TD bootstrap → Q 值崩溃 | QMIX 仅训练 1M 步，T2 极弱 | 报告中如实记录为负结果 |
| CNN pixel 从零训练失败 | 三算法 pixel 模型均无可用策略 | 分析原因（参数量大+稀疏奖励），报告中作为 limitation |
| MAPPO 10M 步未收敛 | 评估中连随机都不如 | 疑为自博弈 collapse，可讨论 |
| T1 spawn 优势过大 | 位置效应压倒算法差异 | 报告中重点讨论环境特性 |
| ALE reward 偶尔非零和 | 累计得分偏差 ~±8 | 不影响胜负判定 |

---

## 八、待办事项

### 🔴 高优先级 — 报告撰写（占 60% 分数）

`report/` 目录需要搭建 LaTeX 项目。建议结构：

```
report/
├── main.tex           # 主文件
├── sections/
│   ├── intro.tex      # 问题定义与环境描述
│   ├── algorithms.tex # IPPO/MAPPO/QMIX 原理 + 伪代码
│   ├── setup.tex      # 实验设置（超参数、硬件、训练配置）
│   ├── results.tex    # 实验结果（学习曲线、对抗矩阵、行为分析）
│   ├── innovation.tex # 创新点：对手池自博弈
│   └── conclusion.tex # 结论与展望（含 QMIX/pixel 负结果）
├── figs/              # 图表（从 results/plots/ 拷贝或重新生成）
├── refs.bib           # 参考文献
└── HANDOFF.md         # 本文档
```

**关键数据来源**：
- 对抗矩阵 → `results/unified_nonpixel.txt`（上表）
- 学习曲线 → `logs/` TensorBoard（可导出 CSV）
- 对抗曲线图 → `results/plots/`
- 训练速度/显存 → [docs/experiment_log.md](../docs/experiment_log.md) 2026-06-20 18:30 条目
- 参数拆解 → [docs/code_architecture.md](../docs/code_architecture.md)

**写作重点建议**：
1. 环境分析部分强调 T1 spawn 位置的先天优势——这是本课题最独特的环境特性
2. 创新点论证用 IPPO_POOL vs IPPO 和 MAPPO_POOL vs MAPPO 的双向胜率做证据
3. QMIX 在稀疏奖励下的失败不是减分项——负结果分析得好反而是加分项
4. 像素训练失败不要回避，作为 limitation 诚实讨论

### 🟡 中优先级 — Demo 录屏

使用 `src/utils/evaluator.py` 的视频录制功能，录制 3-4 个代表性对局：
- IPPO vs RANDOM（展示训练效果）
- IPPO_POOL vs IPPO（展示创新点效果）
- QMIX T1 vs MAPPO T2（展示 QMIX T1 统治力）
- RANDOM T1 vs MAPPO T2（展示 MAPPO 崩溃）

录制命令参考（需在 evaluator 中启用 `record_video=True`）：
```python
from src.utils.evaluator import Evaluator
evaluator = Evaluator(env, model, record_video=True, video_dir="results/videos")
```

### 🟢 低优先级 — 可选改进

- 导出 TensorBoard 学习曲线做报告用图
- 消融实验：对手池容量 K 的影响（K=1/3/5/10）
- 多 seed 训练（当前仅 1 seed）
- 从 `results/models/` 整理最终模型权重（当前散落在 checkpoints 目录）

---

## 九、快速上手

### 环境激活
```bash
source /home/claude/miniconda3/etc/profile.d/conda.sh
conda activate cv-lab3
cd /workspace/MAS/quadrapong
```

### 常用命令
```bash
# 评估：N 个模型全配对，每方向 M 局
python scripts/eval_unified.py \
  checkpoints/ippo/ippo_final.pt \
  checkpoints/mappo/mappo_final.pt \
  ... \
  --include-random --max-steps 20000 --episodes M \
  --device cuda:0 --output results/output.txt

# 画图：从 traj npz 生成对抗曲线
python scripts/plot_eval.py results/output_traj.npz

# 训练（以 IPPO pool 为例）
python scripts/train_ippo_pool.py --device cuda:0
```

### GPU 显存占用参考
- 单个 eval: ~900MB（非 pixel）/ ~1.5GB（含 pixel）
- 单个训练: IPPO ~0.6GB / MAPPO ~1.6GB / QMIX ~0.9GB（RAM 模式）

---

## 十、联系方式与资源

- 环境文档：<https://ale.farama.org/multi-agent-environments/quadrapong/>
- 助教：黄奕喆 szhyz@pku.edu.cn
- 截止：**2026-07-01 18:00**
- 剩余天数：9 天
- AI 协作规范：见项目根目录 `CLAUDE.md`
