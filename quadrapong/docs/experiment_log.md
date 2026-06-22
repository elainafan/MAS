# 实验日志

## 2026-06-20 19:00 — 创新点方案：对手池自博弈

### 问题诊断
标准自博弈（self-play）评估中 T2(agents 1,3) 必胜 100%。根因：初始 spawn 位置固定，策略学习到"某个位置更容易进攻/防守"的单一模式，形成角色固化。策略缺乏对不同对手、不同位置的泛化能力。

### 方案设计
训练时 Team 2 从对手池随机采样，而非始终使用当前策略。

**对手池组成**：
| 对手 | 来源 | 采样权重 | 机制 |
|------|------|---------|------|
| 历史自身 | 每 200K 步保存的 past checkpoint | 60% | FIFO 队列，容量 K=5 |
| 随机策略 | ε=1.0 纯随机 | 20% | 基础探索多样性 |
| 当前自身 | 正在训练的 policy | 20% | 保持对抗压力 |

**训练流程**：
1. 每个 episode 开始：加权随机从池中选一个对手
2. Team 1 = 当前训练策略（正常更新）
3. Team 2 = 对手策略（推理模式，冻结参数）
4. 每 200K 步：当前策略 checkpoint 入池（FIFO）
5. 仅对 IPPO/MAPPO 实施，QMIX 不适用

### 预期结果
- T1/T2 双向胜率不再固化
- 对抗评估中双方胜率趋于均衡
- 可能提升对未见对手的泛化能力

## 2026-06-20 18:30 — 三算法 RAM 对抗评估结果

**结果路径**: `results/unified_matchup.txt`（由 `scripts/eval_unified.py` 生成）

### 对抗结果 (100 episodes/组, max 5000 steps)

```
A(T1) vs B(T2)    IPPO       MAPPO      QMIX
IPPO               —          T2 100%    T2 100%
MAPPO              T2 100%    —          T2 100%
QMIX               T1 100%    平局 100%   —
```

- 所有对局 ep_len=5000 — 三算法均学会不丢球（打满 eval 上限）
- **T2 必胜效应**：IPPO/MAPPO 对抗中 T2(agents 1,3)总赢 — 自博弈导致位置角色分化
- **QMIX 双向通吃**：不管放 T1/T2，对 IPPO 均 100% 胜 — 共享 Q-network 学到无方向控球策略
- **MAPPO vs QMIX 不对称**：QMIX(T2) 16-0 大胜，QMIX(T1) 0-0 平局 — T2 spawn 位置有进攻优势

### 三算法时空开销

**训练速度与 GPU 显存**：

| 算法 | 参数量 | GPU 显存 | RAM 训练速度 | Pixel 训练速度 | RAM 训练耗时 |
|------|--------|----------|-------------|---------------|-------------|
| IPPO | 67K | ~0.6GB | 333/s (CPU) | 159/s (GPU) | 8.3h (10M) |
| MAPPO | 132K | ~1.6GB | 233/s (GPU) | 90/s (GPU) | 11.9h (10M) |
| QMIX | 161K | ~0.9GB | 60/s (GPU) | 22/s (GPU) | 5.0h (1M) |

**运行时 CPU 内存实测** (单次训练，含 buffer/optimizer/中间激活)：

| 算法 | RAM 训练 | Pixel 训练 | 内存瓶颈 |
|------|---------|-----------|---------|
| IPPO | ~0.8 GB | **3.5 GB** | OnPolicyBuffer (2048×4×obs_dim) + CNN 激活 |
| MAPPO | ~1.2 GB | **3.5 GB** | 同上 + global_state buffer |
| QMIX | ~2.1 GB | **34.3 GB** | ReplayBuffer 100K×(N×obs_dim + state_dim) float32 |

> **QMIX pixel 34GB 原因**：ReplayBuffer 存储 100K 条 `next_state`（4agents×21168维×4B = 338KB/条 × 100K = **33.6 GB**）。全观测 off-policy 用 replay buffer 是对 RAM 设计的，像素模式下 state_dim 扩大 165 倍（512→84672），内存爆炸。如需节省可减小 buffer_capacity 或用 CNN 编码后的 state。

详细参数拆解：

| 组件 | IPPO RAM | MAPPO RAM | QMIX RAM | IPPO Pixel | MAPPO Pixel | QMIX Pixel |
|------|----------|-----------|----------|------------|-------------|------------|
| Actor/Agent-Q | 33,798 | 33,798 | 17,286 | 1,685,158 | 1,685,158 | 1,685,158 |
| Critic/Mixer | 33,153 | 98,689 | 143,873 | 1,682,593 | 3,382,593 | 815,617 |
| Global Encoder | — | — | — | — | — | 94,368 |
| **总计** | **66,951** | **132,487** | **161,159** | **3,367,751** | **5,067,751** | **2,595,143** |

每步计算量（CNN forward 次数）：

| 算法 | Rollout 每步 | 训练每次 | 备注 |
|------|-------------|---------|------|
| IPPO RAM/CNN | 2 | 每2048步1次更新 | on-policy |
| MAPPO RAM/CNN | 1+4=5 | 每2048步1次更新 | 集中式critic |
| QMIX RAM/CNN | 1 | 每N步1次更新 | off-policy, train_interval=4 |

**速度瓶颈**：QMIX 为 off-policy 每步训练，每步 backward 开销大。IPPO/MAPPO 为 on-policy，每 2048 步才训练一次，rollout 阶段仅 forward。

## 2026-06-20 00:31 — QMIX 稀疏奖励诊断（无效）

**问题**: q_loss 在 25K 时健康 (0.178)，到 50K 时归零 (0.00006)。q_tot 从 2.6 跌至 0.9。

**诊断**: Agent 实测 10 局，进球率 0.20%。batch=32 时 6.3% 批次有信号，batch=256 时 40.6%。

**修复**: batch 32→256, eps_decay 50K→300K, tau 0.005→0.001

**结果**: 失败。225K 时 q_loss 再次归零 0.0005，q_tot→0.13，eval 胜率从 65% 跌至 0%。batch 增大只延缓崩溃，不根除。

## 2026-06-20 02:33 — QMIX Bootstrap Collapse 根除

**根本原因**: 99.8% transition reward=0，`target = γ·Q_target(s')` 自举锁死 Q→0 是数学必然。99.8% 样本的梯度要求 Q=0，淹没 0.2% 进球信号。batch_size 无关——信号比例恒为 0.2%。

**修复**:
1. FFQNetwork（纯 MLP）替代 DRQN，移除 RNN 噪音
2. Reward shaping: Team 1 +0.002/step，提供密集梯度
3. gamma 0.99→0.95，缩短自举链
4. 代码大幅简化（移除所有 RNN state 逻辑）

**结果**: 失败。shaping 提供密集梯度，但 gamma=0.95 有效视界仅 ~30 步。Q 值衰减到 shaping 下限 0.08。470K 时 winrate 从 80%→10%。

## 2026-06-20 09:44 — QMIX γ=0.99 + PER + 硬更新

**根因**: gamma 太低 + 均匀采样稀释信号。

**修复**:
1. gamma 0.95→0.99（视界 30→400 步，shaping 下限 0.08→1.0）
2. PER（按 TD error 优先采样 goal transition）
3. 软更新→硬更新（消除目标滞后偏置）
4. shaping 0.002→0.005

## 2026-06-21 23:00 — 全量对抗评估

**分两批**（pixel 模型仅 5 局象征性；非 pixel 完整 50 局）：

```bash
# 第一批：非 pixel 模型（C(6,2)=15 组对阵）+ RANDOM, 50 episodes
python scripts/eval_unified.py \
  checkpoints/ippo/ippo_final.pt \
  checkpoints/mappo/mappo_final.pt \
  checkpoints/qmix/qmix_final.pt \
  checkpoints/ippo_pool/ippo_pool_final.pt \
  checkpoints/mappo_pool/mappo_pool_final.pt \
  --include-random --max-steps 20000 --episodes 50 \
  --device cuda:0 --output results/unified_nonpixel.txt

# 第二批：全部 7 模型 + RANDOM, 5 episodes（覆盖所有 pixel 对阵）
python scripts/eval_unified.py \
  checkpoints/ippo/ippo_final.pt ... (全部 7 个 checkpoint) \
  --include-random --max-steps 20000 --episodes 5 \
  --device cuda:0 --output results/unified_pixel.txt
```

**参数说明**:
- `--max-steps 20000`：每局上限（经验：多数对局 15-20K 步自然结束）
- `--include-random`：追加 RANDOM baseline（均匀随机 6 动作）
- 每局独立 seed，跨 obs 双环境同 seed 锁步 + reward 同步断言
- 每 1000 步记录比分 → `_traj.npz` 轨迹文件
- 终止分类：`natural`(到 21 分) / `trunc(lead)`(截断有优势) / `trunc(tie)`(截断 0-0)
- 画图：`scripts/plot_eval.py results/unified_nonpixel_traj.npz` → `results/plots/`

**评估模型**（8 实体）:

| 模型 | obs | 算法 | episodes |
|------|-----|------|----------|
| IPPO | RAM 128d | IPPO | 50 |
| MAPPO | RAM 128d | MAPPO | 50 |
| QMIX | RAM 128d | QMIX | 50 |
| IPPO pixel | grayscale(3,84,84) | IPPO | 5（象征） |
| MAPPO pixel | grayscale(3,84,84) | MAPPO | 5（象征） |
| IPPO pool | RAM 128d | IPPO+Pool | 50 |
| MAPPO pool | RAM 128d | MAPPO+Pool | 50 |
| RANDOM | — | uniform | 50 |

**数据路径**: `results/unified_nonpixel.txt` + `results/unified_nonpixel_traj.npz`（batch 1）<br>
`results/unified_all_5ep_v2.txt` + `results/unified_all_5ep_v2_traj.npz`（batch 2 v2, 已修复标签bug）

### Batch 1 结果（非pixel 50ep, 6模型, 15组对阵, 2h58m, GPU 0）

**对抗矩阵** (T1 WR vs T2 WR, 50ep/方向, max_steps=20000):

| A(T1) vs B(T2) | IPPO | MAPPO | QMIX | IPPO_POOL | MAPPO_POOL | RANDOM |
|-----------------|------|-------|------|-----------|------------|--------|
| **IPPO** | — | 100%/100% | 100%/0% | 0%/0% | 100%/100% | 96%/40% |
| **MAPPO** | 0%/0% | — | 0%/100% | 0%/0% | 0%/0% | 2%/6% |
| **QMIX** | 100%/0% | 0%/0%(Dr) | — | 100%/0% | 0%/0% | 56%/30% |
| **IPPO_POOL** | 100%/100% | 100%/100% | 0%/100% | — | 0%/100% | 88%/20% |
| **MAPPO_POOL** | 0%/0% | 100%/0%(Dr) | 100%/100% | 100%/0% | — | 80%/46% |
| **RANDOM** | 60%/2% | 94%/98% | 60%/40% | 78%/10% | 52%/14% | — |

> 每格：T1方向WR / T2方向WR。100%/0% = A做T1必胜,B做T1必败（A强）。Dr=平局。

**核心发现**:

1. **IPPO_POOL 双向赢 IPPO**（对手池有效）：IPPO_POOL 在 T1 和 T2 位置都能赢 IPPO，打破了标准自博弈的 T2 必胜魔咒
2. **MAPPO_POOL 双向赢 MAPPO**：提升幅度极大——MAPPO_POOL T2 对 MAPPO T1 净胜 52 分（最反直觉的结果：T2弱势位赢T1强势位）
3. **MAPPO 全面崩溃**：连 RANDOM 都打不过（MAPPO T1 仅 2% 胜率对 RANDOM T2）
4. **QMIX 极端不对称**：T1 可零封 MAPPO（42:-42），T2 对 MAPPO 却 0-0 平局打满 20K 步
5. **T1 spawn 优势压倒一切**：RANDOM T1 对大部分模型有 50%+ 胜率

**终止类型统计**：15组30方向共1500局中，Natural=74.7%, TruncLead=23.0%, TruncTie=2.3%。

### 诊断发现：确定性策略的 std=0 现象

经实验验证：训练好的 IPPO/MAPPO 使用 argmax（确定性）策略时，不同 random seed 产生**完全相同**的对局结果（分数、步数均一致）。3 次独立验证 IPPO vs MAPPO 均为 T1=28:-28, 18772 步。

- 模型 vs 模型：50 局全部相同（std=0），1 局即代表全部
- 模型 vs RANDOM：std > 0（随机策略引入方差）
- **影响**：50 局对模型间对抗纯属浪费，实际 1 局即可

### ALE Quadrapong 奖励非零和

环境在少数步中分配 partial reward（如仅给 T2 两 agent 各 -1，T1 两 agent 得 0），导致 cumulative reward 最多偏移 ~±8。胜负判定（T1_total > T2_total）不受影响。

### Batch 2 v2（全模型 5ep, 8模型含 pixel, 28组）

Batch 2 初版存在 `get_algo_name` bug——`ippo_pixel`/`mappo_pixel` 被 `ippo`/`mappo` 子串误匹配，导致 pixel 模型被错标。v2 已修复，输出到 `results/unified_all_5ep_v2.txt`。

### 图表

`python scripts/plot_eval.py results/unified_nonpixel_traj.npz` → `results/plots/`（15张，双向合并）

## 2026-06-21 00:35 — QMIX pixel 训练终止 ❌

**结论**: QMIX 在 pixel 和 RAM 两种观测下均无法收敛。根因是 Quadrapong 稀疏奖励（~0.2% 进球率）与 TD bootstrap 的组合产生数学必然的崩溃——99.8% 零奖励 transition 的自举不动点是 Q=0，0.2% 进球信号通过 γ 链放大为 Q 值爆炸。所有修复尝试（γ 调参、PER、RMSprop、硬更新、reward shaping、FF-Q 替代 DRQN、batch_size）均只延缓不根除。

**QMIX RAM 尝试历史**: 7 次重启，γ=0.95/0.97/0.99、PER/均匀、soft/hard update、shaping 0.002/0.005、batch 32/256 — 全部最终崩溃。

**QMIX pixel 最终状态**: 277K/1M，q_loss 48~39,665 振荡，q_tot 18-50，eval 对随机仅 25% 胜率，内存 134 GB。

**报告中处理**: 如实记录为负结果，分析稀疏奖励 + off-policy + CTDE 在该任务上的根本困难。

## 2026-06-19 15:30 — IPPO/MAPPO 首批训练（已停止）

**结论**：方向正确但价值有限，指标趋势清晰后提前终止，直接进入全规模训练。

### IPPO 轻量训练

| 指标 | 初始 | 最终 (1.05M / 2M) | 趋势 |
|------|------|------|------|
| entropy | 1.756 | 1.688 | ↓ 策略更确定 |
| ep_len | 0 (无终止) | 599 | ↑ 学会让游戏结束 |
| episodes | 0 | 44 | ↑ |
| ep_reward | 0 | -289 | Team 1 落后 |
| 运行时间 | | ~3h | |
| 累计速率 | | ~99 steps/s | (受 eval 拖累) |

### MAPPO 轻量训练

| 指标 | 初始 | 最终 (700K / 2M) | 趋势 |
|------|------|------|------|
| entropy | 1.753 | 1.599 | ↓ 收敛快于 IPPO |
| ep_len | 0 (无终止) | 431 | ↑ MAPPO 游戏更短 |
| episodes | 0 | 29 | ↑ |
| ep_reward | 0 | -289 | Team 1 落后 |
| 运行时间 | | ~3h | |
| 累计速率 | | ~66 steps/s | (集中式 Critic 更重) |

### 分析

- **方向正确**：熵下降、episode 从无法终止到 400-600 步结束，说明策略在学
- **MAPPO vs IPPO**：MAPPO 熵下降更快 (1.60 vs 1.69)，ep_len 更短 (431 vs 599)，集中式 Critic 有加速收敛迹象
- **value_loss 不可信**：GAE bug 导致 critic collapse，loss 值无参考意义
- **自博弈困境**：两队使用相同策略且随机初始化相同，Team 1 一直在输（ep_r=-289），存在对称破缺但缓慢
- **确定性 eval 无效**：未收敛策略用 deterministic 模式所有 agent 选 NOOP，eval 全跑满 max_cycles

### 对外规模训练的启示

1. 全规模训练需用不同 seed 初始化两队 actor（打破对称）
2. MAPPO 收敛更快，10M 步可能不需要跑满
3. eval 需要 max_eval_steps 截断（已修复）
4. 需要训练足够步数以观察 ep_len 是否继续下降
