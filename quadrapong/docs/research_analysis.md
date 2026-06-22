# Quadrapong 2v2 多智能体乒乓对抗 — 调研分析与任务拆解

## 1. 课题要求摘要

| 项目 | 内容 |
|------|------|
| 课题 | 课题六：2v2 乒乓赛（Quadrapong） |
| 助教 | 黄奕喆 (szhyz@pku.edu.cn) |
| 目标 | 控制 4 个智能体（两两一队）进行乒乓球对抗 |
| 满分 | 35 分（算法设计 21 + 创新 14） |
| 评分 | 报告 60% + 代码 & demo 40% |
| 截止 | 2026-07-01 18:00（本学期不毕业） |
| 提交 | 研究报告 PDF + 完整代码&模型 + Demo 录屏 |

**必做基线**：IPPO、MAPPO、QMIX 三种算法，两两对比。

**硬件**：2× RTX 4090 (24GB VRAM each)，240 CPU 核，容器可长时运行。

---

## 2. 环境分析：Quadrapong

### 2.1 环境概况

Quadrapong 是 Atari 2600 Pong 的 4 人变体，由 PettingZoo/ALE 提供。

```
环境: PettingZoo → ALE → multi_agent_ale_py
ROM: pong.bin, mode=33
```

### 2.2 智能体与队伍

| 队伍 | 成员 | 防守区域 |
|------|------|----------|
| Team 1 | `first_0`, `third_0` | 两个相邻得分区 |
| Team 2 | `second_0`, `fourth_0` | 另两个相邻得分区 |

**物理布局**：4 个 paddle 分别防守屏幕四边，同一队的两个 paddle 防守相邻的两条边。

### 2.3 观测空间

| 类型 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| RAM | `(128,)` | `uint8` (0–255) | Atari 128 字节内存，低维，训练快 |
| RGB | `(210, 160, 3)` | `uint8` (0–255) | 彩色图像，需 CNN 编码 |
| Grayscale | `(210, 160, 1)` | `uint8` (0–255) | 灰度图 |

**推荐使用 RAM 观测**：128 维向量，训练效率高，适合 RL 算法快速迭代。考虑使用 CNN 处理像素作为创新点之一。

### 2.4 动作空间

6 个离散动作（精简动作集，`full_action_space=False`）：

| 值 | 动作 | 描述 |
|----|------|------|
| 0 | NOOP | 无操作 |
| 1 | FIRE | 发球 |
| 2 | UP | 上移 |
| 3 | RIGHT | 右移 |
| 4 | LEFT | 左移 |
| 5 | DOWN | 下移 |

### 2.5 奖励结构

- **得分**：进球一方队伍 **+1**，对方队伍 **-1**
- **发球超时**：持球 2 秒未发出，该队 **-1**，计时器重置
- **整体接近零和博弈**，但发球超时惩罚使其不是严格零和

**MARL 关键问题**：
- 同一队内：**合作**（共享奖励）
- 队之间：**对抗**（零和奖励）
- 信用分配：如何区分每个 agent 对团队得分的贡献

### 2.6 终止条件

- 游戏结束（一方失去所有生命）→ `termination=True`
- `max_cycles`（默认 100000）→ `truncation=True`

**关键参数**：
```python
quadrapong_v4.parallel_env(
    obs_type='ram',         # 'ram' | 'rgb_image' | 'grayscale_image'
    max_cycles=100000,       # 最大帧数
    full_action_space=False, # 精简 6 动作 vs 完整 18 动作
)
```

---

## 3. 算法分析

### 3.1 IPPO（Independent PPO）

**核心思想**：每个 agent 独立运行 PPO，没有信息共享。

```
Architecture:
  Agent_i:
    obs_i → [共享 Actor 网络] → action_i, log_prob_i
    obs_i → [独立 Critic V_i]   → value_i

  Loss:
    L_clip = min(r*A_i, clip(r, 1-ε, 1+ε)*A_i)  (标准 PPO clip)
    L_value = (V_i - R_i)²
    L_entropy = H(π_i)

  Advantage: GAE (λ=0.95, γ=0.99), agent-local
```

**特点**：
- 完全分布式，没有集中式信息
- 参数共享：所有 4 个 agent 共享同一 Actor 网络，减少参数量
- 非平稳性问题：每个 agent 视其他 agent 为环境一部分
- 作为下界基线

**参考**：Schulman et al., "Proximal Policy Optimization Algorithms", 2017

### 3.2 MAPPO（Multi-Agent PPO）

**核心思想**：CTDE 框架，集中式 Critic 获取全局状态，Actor 仅用局部观测。

```
Architecture:
  Agent_i:
    obs_i → [共享 Actor π_θ] → action_i, log_prob_i
    (obs_i, global_state) → [集中式 Critic V_φ] → value_i

  Critic 输入配置（5 种选择）:
    - EP: 仅全局状态 s
    - AS: s + agent 特定特征（推荐）
    - FP: AS 的裁剪版
```

**与 IPPO 的关键差异**：
| | IPPO | MAPPO |
|---|---|---|
| Critic | 独立，仅局部 obs | 集中式，全局 state + 局部 obs |
| 优势函数 | 局部 GAE | 集中式 GAE |
| 信息利用 | agent 私有的 | 全局共享的 |

**关键工程细节（来自论文）**：
1. **Value Normalization**（PopArt）：稳定值函数学习
2. **PPO Clipping ε = 0.2**：控制多智能体非平稳性
3. **训练数据复用**：5–15 epochs，1 mini-batch
4. **Death Masking**：Quadrapong 中所有 agent 始终活跃，不需要

**参考**：Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022

### 3.3 QMIX（Monotonic Value Function Factorisation）

**核心思想**：Off-policy CTDE。每个 agent 学习独立的 Q 函数，通过单调混合网络合并为全局 Q_tot。

```
Architecture:
  Agent_i: τ_i → [DRQN] → Q_i(τ_i, a_i)   (GRU + MLP)
  
  Mixing Network:
    [Q_1, ..., Q_4] → [MLP with ≥0 weights] → Q_tot
                            ↑
                     Hypernetworks(s) → W, b
                     (|W| 保证单调性)
  
  约束: ∂Q_tot / ∂Q_i ≥ 0  (单调性，保证 IGM 原则)
```

**训练过程（Off-policy）**：
```
1. ε-greedy 探索：a_i = argmax Q_i (1-ε) 或 random (ε)
2. 存入 replay buffer：⟨s, u, r, s', done⟩
3. 采样 mini-batch，TD 学习：
   L = [Q_tot(s, u) - (r + γ * max_u' Q_tot(s', u'; θ⁻))]²
4. 软更新目标网络 θ⁻ ← τ*θ + (1-τ)*θ⁻
```

**关键点**：
- 单调性约束保证 `argmax Q_tot = (argmax Q_1, ..., argmax Q_n)`，分布式执行最优
- 适用于合作场景，但 Quadrapong 有对抗元素
- **需要处理对抗问题**：可考虑将对手视为环境，或使用多组 QMIX

**参考**：Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL", ICML 2018

### 3.4 算法对比总结

| 维度 | IPPO | MAPPO | QMIX |
|------|------|-------|------|
| 类型 | On-policy | On-policy | Off-policy |
| 框架 | 独立 | CTDE | CTDE |
| Critic | 局部 V(s) | 集中 V(s, o_i) | Q_i + Mixing Network |
| 参数共享 | Actor 共享 | Actor 共享 | 可选 |
| 样本效率 | 低 | 中 | 高 |
| 训练速度 | 快 | 中 | 慢 |
| 信用分配 | 隐式 | 隐式 | 显式（混合权重） |
| Quadrapong 适用性 | 弱 | 中 | 需改造 |

### 3.5 算法对抗适配（Quadrapong 特殊处理）

Quadrapong 的 2v2 对抗需要特殊处理：

**方案 A：队友合作，对手视为环境**
- 对 Team 1 训练一个合作策略，Team 2 使用固定策略（随机/预训练）
- 每个队独立训练 IPPO/MAPPO/QMIX
- 缺点：不会学到对抗策略

**方案 B：自博弈（Self-Play）**
- 两队使用相同算法但互为对手
- 过去版本的策略作为对手（历史模型池）
- 适合 IPPO/MAPPO（on-policy 自博弈）
- QMIX 可结合自博弈训练

**方案 C：混合训练**
- Team 1 使用 MAPPO，Team 2 使用 QMIX
- 两支队伍相互对抗、共同进化
- 创新点：异质算法对抗

**推荐**：先实现方案 A 完成基线，方案 B/C 作为创新。

---

## 4. 任务阶段拆解

### Phase 1：基础框架搭建（6/19 – 6/21，3 天）

**目标**：可运行的环境+训练+评估框架，随机策略能跑通。

| # | 任务 | 产出 | 负责人建议 |
|---|------|------|-----------|
| 1.1 | 环境包装：统一 PettingZoo→自定义接口，支持 RAM/像素双模式 | `src/envs/quadrapong_env.py` | A |
| 1.2 | 网络模块：MLP Policy、CNN Encoder、GRU 支持 | `src/utils/networks.py` | B |
| 1.3 | 经验缓冲：On-policy Buffer、Replay Buffer | `src/utils/buffer.py` | B |
| 1.4 | Rollout 采集器：并行环境交互、数据收集 | `src/utils/rollout.py` | A |
| 1.5 | 评估器：定期评估、录像保存、指标计算 | `src/utils/evaluator.py` | C |
| 1.6 | Logger：TensorBoard + WandB 日志 | `src/utils/logger.py` | C |
| 1.7 | 配置文件系统：YAML config for each algo | `src/configs/` | C |

### Phase 2：IPPO 实现与调优（6/21 – 6/23，3 天）

**目标**：IPPO 双队自博弈跑通，得到 baseline 结果。

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | 实现 PPO Actor（共享参数 CNN/MLP 策略）| `src/algos/ippo.py` |
| 2.2 | 实现 PPO Critic + GAE 优势估计 | 同上 |
| 2.3 | 自博弈训练循环（两队交替更新）| `scripts/train_ippo.py` |
| 2.4 | 超参数搜索（lr, clip_eps, entropy_coef, γ, λ）| 实验记录 |
| 2.5 | 训练 10M+ steps，记录学习曲线 | `results/` |

### Phase 3：MAPPO 实现与调优（6/23 – 6/25，3 天）

**目标**：MAPPO 跑通，与 IPPO 对比。

| # | 任务 | 产出 |
|---|------|------|
| 3.1 | 实现集中式 Critic（多输入模式：EP/AS/FP）| `src/algos/mappo.py` |
| 3.2 | Value Normalization（PopArt）| `src/utils/popart.py` |
| 3.3 | 训练循环 + 自博弈 | `scripts/train_mappo.py` |
| 3.4 | 超参数调优 | 实验记录 |
| 3.5 | IPPO vs MAPPO 对比实验 | 学习曲线、胜率、奖励分布 |

### Phase 4：QMIX 实现与适配（6/25 – 6/27，3 天）

**目标**：QMIX 适配 Quadrapong，完成三个基线。

| # | 任务 | 产出 |
|---|------|------|
| 4.1 | 实现 DRQN Agent 网络（GRU + MLP）| `src/algos/qmix.py` |
| 4.2 | 实现 Mixing Network + Hypernetworks | 同上 |
| 4.3 | 实现 Replay Buffer + TD 训练 | 同上 |
| 4.4 | 对抗场景适配（自博弈 / 对手池）| `scripts/train_qmix.py` |
| 4.5 | QMIX 超参数调优 | 实验记录 |
| 4.6 | 三算法完整对比 | 全面分析 |

### Phase 5：创新点实验（6/27 – 6/29，3 天）

**目标**：提出并验证至少 1–2 个创新点。

**可选创新方向**：

| 方向 | 具体方案 | 难度 |
|------|----------|------|
| A. 算法改进 | 引入注意力机制的 Critic（Multi-Head Attention 编码全局信息）| 中 |
| B. 算法改进 | 将 QMIX 改造为 Qatten/QPLEX 等更 expressive 的分解 | 中 |
| C. 算法改进 | WQMIX（加权 QMIX）解决单调性限制 | 中 |
| D. 训练方式 | 课程学习：从 2v1 简化场景渐进到 2v2 | 低 |
| E. 训练方式 | League Training（AlphaStar 风格联赛训练） | 高 |
| F. 观测改进 | RAM→CNN 像素观测 + 帧堆叠 + 注意力 | 中 |
| G. 环境改造 | 增加随机扰动、风力、延迟等环境变量 | 低 |
| H. 对抗鲁棒性 | 训练对多种对手策略的鲁棒性 | 中 |
| I. 异质对抗 | Team 1 MAPPO vs Team 2 QMIX，分析各算法优劣势 | 低 |

**推荐组合**：I（异质对抗，实验成本低）+ D（课程学习）+ A（注意力 Critic）

### Phase 6：报告撰写与 Demo（6/29 – 7/1，2 天）

**目标**：完成 LaTeX 报告 + 录屏。

| # | 任务 |
|---|------|
| 6.1 | LaTeX 模板搭建（ICML 2022）|
| 6.2 | 实验图表：学习曲线（3 算法 × N seeds）、奖励分布、胜率矩阵、行为分析 |
| 6.3 | 核心算法伪代码 |
| 6.4 | 实验结果分析（对比表、消融实验） |
| 6.5 | 创新点论证 |
| 6.6 | 参考文献整理 |
| 6.7 | Demo 录屏 |
| 6.8 | 最终检查、打包提交 |

---

## 5. 项目目录结构

> **详细代码架构与各文件功能见 [`code_architecture.md`](code_architecture.md)**。以下为目录概览。

```
quadrapong/
├── CLAUDE.md              # 项目规范
├── docs/
│   ├── research_analysis.md  # 本文档（调研分析）
│   ├── code_architecture.md  # 代码架构详解
│   ├── code_log.md           # 代码修改日志
│   ├── train_log.md          # 训练日志
│   └── experiment_log.md     # 实验分析与结论
├── src/
│   ├── algos/             # ippo.py, mappo.py, qmix.py
│   ├── envs/              # quadrapong_env.py
│   ├── utils/             # buffer.py, networks.py, evaluator.py, logger.py, rollout.py
│   └── configs/           # ippo.yaml, mappo.yaml, qmix.yaml (+ light 版)
├── scripts/               # train_ippo.py, train_mappo.py, train_qmix.py
├── checkpoints/           # 模型检查点
├── logs/tensorboard/      # TensorBoard 日志
├── papers/                # 参考实现 (HARL, pyMARL)
├── results/               # 最终结果 (models/, plots/, videos/)
└── report/                # LaTeX 报告源文件（待创建）
```

---

## 6. 环境配置

| 工具 | 版本 | 说明 |
|------|------|------|
| Python | 3.11.15 | conda env `cv-lab3` |
| PyTorch | 2.7.0+cu128 | GPU 训练 |
| PettingZoo | 1.24.3 | Quadrapong 环境 |
| multi-agent-ale-py | 0.1.12 | ALE 后端 |
| Gymnasium | 1.3.0 | 环境接口 |
| TensorBoard | 2.20.0 | 训练日志 |
| WandB | 0.27.2 | 实验追踪（可选）|

**激活环境**：`conda activate cv-lab3`

---

## 7. 团队分工建议（3 人）

| 成员 | 主要职责 | 备注 |
|------|----------|------|
| A | 环境包装 + 训练框架 + IPPO | 框架搭建先行 |
| B | 网络模块 + MAPPO + QMIX 混和网络 | 算法核心 |
| C | 评估/日志 + 调参实验 + 创新点 | 实验与分析 |

交叉审查：每个算法至少两人 review 代码和实验结果。

---

## 8. 关键风险与应对

| 风险 | 应对 |
|------|------|
| 训练不稳定/不收敛 | 减小 lr，增加 value normalization，调 γ/λ |
| QMIX 在对抗场景效果差 | 使用自博弈，考虑 WQMIX 改进 |
| 时间不够（GPU 排队）| 使用 RAM 观测（训练快），夜间跑长实验 |
| WandB 网络问题 | 使用本地 TensorBoard 替代 |
| 队友间信用分配困难 | 尝试 individual reward shaping |

---

## 9. 参考文献

1. Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
2. Yu, C. et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS, 2022.
3. Rashid, T. et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL." ICML, 2018.
4. Mnih, V. et al. "Human-level control through deep reinforcement learning." Nature, 2015.
5. Sunehag, P. et al. "Value-Decomposition Networks For Cooperative Multi-Agent Learning." AAMAS, 2018.
