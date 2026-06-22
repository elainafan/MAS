# 代码架构文档

> 维护规则：新增/修改模块时同步更新本文档。其他 docs 如引用了具体文件，改为指向本文档的对应章节。

## 目录总览

```
quadrapong/
├── src/
│   ├── envs/quadrapong_env.py   # 环境包装（PettingZoo → 统一接口）
│   ├── algos/
│   │   ├── ippo.py              # IPPO trainer（独立 PPO + 参数共享）
│   │   ├── mappo.py             # MAPPO trainer（集中式 Critic + PopArt）
│   │   └── qmix.py              # QMIX trainer（DRQN + MixingNetwork）
│   ├── utils/
│   │   ├── networks.py          # 神经网络模块（Actor, Critic, DRQN, MixingNet）
│   │   ├── buffer.py            # 经验缓冲（OnPolicyBuffer, ReplayBuffer）
│   │   ├── evaluator.py         # 评估 + 录像 + MetricsTracker
│   │   ├── logger.py            # 统一日志（TensorBoard + WandB + file）
│   │   └── rollout.py           # Rollout 采集器（未使用，被训练脚本内联替代）
│   └── configs/                 # YAML 超参数配置
│       ├── ippo.yaml / ippo_light.yaml
│       ├── mappo.yaml / mappo_light.yaml
│       └── qmix.yaml
├── scripts/
│   ├── train_ippo.py            # IPPO 训练入口
│   ├── train_mappo.py           # MAPPO 训练入口
│   └── train_qmix.py            # QMIX 训练入口
├── checkpoints/                 # 模型检查点
├── logs/tensorboard/            # TensorBoard 事件文件
└── docs/                        # 项目文档
```

---

## 1. 环境层 — `src/envs/quadrapong_env.py`

### `class QuadrapongWrapper`
统一包装 PettingZoo `quadrapong_v4.parallel_env`，提供类 Gym 接口。

| 方法 | 功能 |
|------|------|
| `__init__(obs_type, max_cycles, frame_stack, render_mode)` | 创建临时 env 获取 obs/action space 元数据 |
| `reset(seed)` | 重建 env（PettingZoo 限制），清空帧缓冲，返回 `(obs_dict, info)` |
| `step(actions)` | 执行动作字典 `{"first_0": 0, ...}`，归一化像素到 [0,1] |
| `_stack_frames(agent, frame)` | 为 CNN 模式堆叠连续帧 |
| `get_team_rewards(rewards)` | rewards → `{"team_1": ..., "team_2": ...}` 聚合 |
| `get_team_agents(team)` | 返回 Team 1 或 2 的 agent 名列表 |

**属性**: `possible_agents = ["first_0", "second_0", "third_0", "fourth_0"]`, `team_1 = ["first_0", "third_0"]`, `team_2 = ["second_0", "fourth_0"]`, `num_agents = 4`, `obs_dim = 128`（RAM）。

**数据流**: RAM obs `(128,) uint8` → `/255.0` → `float32 [0,1]`。像素 obs `(210,160,C)` → 帧堆叠 → `(C*stack, 210, 160)`。

---

## 2. 网络模块 — `src/utils/networks.py`

所有网络都是 `nn.Module` 子类，通过 `make_mlp()` 工厂函数构建 MLP。

### `make_mlp(input_dim, hidden_dims, output_dim, activation, output_activation, use_layer_norm)`
构建 `Linear → [LayerNorm?] → Activation → ... → Linear → [OutputActivation?]` 序列。

### `class StochasticActor`
离散动作随机策略（IPPO/MAPPO 共用）。

| 参数 | 说明 |
|------|------|
| `obs_dim, action_dim` | 输入输出维度 |
| `hidden_dims` | MLP 隐藏层，默认 `[128,128]` |
| `use_rnn, rnn_hidden` | 可选 GRU 前置（当前未启用） |

| 方法 | 返回 |
|------|------|
| `forward(obs, rnn_state, mask)` | `(actions, log_probs, new_rnn_state)` — 采样动作 |
| `evaluate_actions(obs, rnn_state, actions, mask)` | `(log_probs, entropy)` — PPO 更新用 |
| `get_action_distribution(obs)` | `Categorical` 分布 |

### `class Critic`
局部价值函数（IPPO 用）。输入 `obs` → MLP → `V(s)` 标量。

### `class CentralizedCritic`
集中式 Critic（MAPPO 用）。输入 `(global_state, local_obs)` 拼接 → MLP → `V(s, o_i)`。

### `class CNNEncoder`
Nature DQN 卷积编码器。输入 `(B, C, H, W)` → Conv(32,8,4)+ReLU → Conv(64,4,2)+ReLU → Conv(64,3,1)+ReLU → Flatten → `(B, feat_dim)`。自动计算 `feat_dim`。

### `class CNNActor`
CNN 随机策略（IPPO/MAPPO 像素模式用）。`encoder(CNNEncoder) + mlp`。`use_rnn=False`，提供 `get_logits(obs)` 供 evaluator 使用。

### `class CNNCritic`
CNN 局部价值函数（IPPO 像素模式用）。`encoder → mlp → V(s)`。

### `class CNNCentralizedCritic`
CNN 集中式 Critic（MAPPO 像素模式用）。共享 encoder 编码 global（所有 agent 像素通道堆叠）和 local obs，拼接特征后输出 V(s, o_i)。

### `class CNNFFQNetwork`
CNN FF-Q 网络（QMIX 像素模式用）。`encoder → mlp → Q(s,a)`。

### `class FFQNetwork`
Feed-forward Q-network（QMIX agent 用，RAM 全观测无需 RNN）。

| 参数 | 说明 |
|------|------|
| `obs_dim, action_dim` | 128, 6 |
| `hidden_dims` | MLP 隐藏层，默认 `[128]` |

`forward(obs)` → `(batch, action_dim)` — 单返回值。

### `class DRQN`（已弃用）
Deep Recurrent Q-Network。保留供参考，当前 QMIX 使用 FFQNetwork。

### `class MixingNetwork`
QMIX 单调混合网络。两个超网络（hypernetwork）从全局 state 生成权重，保证 `∂Q_tot/∂Q_i ≥ 0`。

**结构**:
```
agent_qs (B,4) → [W1≥0, b1] → ELU → [W2≥0, b2] → Q_tot (B,1)
                    ↑              ↑
              hyper_w1(state)  hyper_w2(state)
```

---

## 3. 缓冲模块 — `src/utils/buffer.py`

### `class OnPolicyBuffer`
存储 rollout 数据，计算 GAE。供 IPPO/MAPPO 使用。

| 方法 | 功能 |
|------|------|
| `insert(obs, actions, ..., global_state)` | 存一步转移 |
| `compute_gae(last_value, last_done)` | 向量化 GAE，返回 `(advantages, returns)` |
| `get_training_data()` | 返回所有数据的 dict |

**GAE 公式**: `δ_t = r_t + γ·V_{t+1}·(1-done_t) - V_t`，`A_t = δ_t + γλ·(1-done_t)·A_{t+1}`。

### `class ReplayBuffer`
Off-policy 回放缓冲（QMIX 用）。定长循环存储，随机采样。

| 存储 | 形状 |
|------|------|
| `obs, next_obs` | `(cap, N, obs_dim)` |
| `state, next_state` | `(cap, N*obs_dim)` |
| `actions` | `(cap, N)` |
| `rewards` | `(cap, N)` |
| `dones` | `(cap, 1)` |
| `rnn_states` | `(cap, N, rnn_hidden)` — DRQN 时序状态 |

---

## 4. 算法层 — `src/algos/`

### 4.1 `ippo.py` — `class IPPOTrainer`

独立 PPO，参数共享。所有 agent 共用同一 Actor 和 Critic。无集中式信息。

| 方法 | 功能 |
|------|------|
| `get_actions(obs, deterministic)` | 前向 Actor 获取动作 |
| `get_values(obs)` | 前向 Critic 获取值 |
| `train_on_buffer(buffer, writer, step)` | PPO 更新（未被训练脚本调用，被内联 `_train_ppo_update` 替代） |
| `save/load(path)` | 模型持久化 |

**训练流程**（在 `train_ippo.py` 中内联）:
1. Rollout 2048 步，存 OnPolicyBuffer
2. 计算 GAE（per-agent advantage）
3. PPO 更新：`L = L_clip + value_coef·L_value - entropy_coef·H`
4. 共用 optimizer 更新 actor + critic

### 4.2 `mappo.py` — `class MAPPOTrainer`

集中式 Critic，输入全局 state + 局部 obs。Actor 同 IPPO。

| 方法 | 功能 |
|------|------|
| `get_actions(obs, deterministic)` | 同 IPPO |
| `train_on_buffer(...)` | 含 PopArt 的 PPO 更新（未被调用，被 `_train_mappo_update` 替代） |
| `save/load(path)` | 模型持久化 |

**与 IPPO 差异**: Critic 是 `CentralizedCritic(global, local)`，优势函数使用集中式 value。PopArt 在 `_train_mappo_update` 中（`use_popart=True` 时生效）。

### `class PopArt`
价值归一化。维护 running mean/std，`normalize(x)` 归一化目标，`denormalize(x)` 还原预测值，`update(x)` 指数移动平均更新统计量。

### 4.3 `qmix.py` — `class QMIXTrainer`

Off-policy CTDE。每个 agent 一个 DRQN，MixingNetwork 合并为 Q_tot。

| 方法 | 功能 |
|------|------|
| `get_actions(obs, rnn_states, deterministic)` | ε-greedy，返回 `(actions, new_rnn_states)` |
| `push_to_buffer(..., rnn_states)` | 存转移 + RNN hidden state |
| `update(writer, step)` | 采样 → Q_tot → TD target → MSE loss → soft update |
| `save/load(path)` | 模型持久化 |

**训练流程**（`train_qmix.py`）:
1. 每步 ε-greedy 采样（Team 2 强制随机）
2. 存入 ReplayBuffer（含 RNN hidden state）
3. 每步采样 batch，计算：`L = MSE(Q_tot(s,u), r_team1 + γ·Q_tot_target(s', u'))`
4. 双 Q 学习：online 选动作，target 评估
5. 软更新目标网络 `θ⁻ ← τθ + (1-τ)θ⁻`
6. Team 1 reward = `rewards_b[:, 0]`（first_0 视角）

---

## 5. 工具模块 — `src/utils/`

### 5.1 `evaluator.py`

#### `evaluate(env, agent_actors, ..., random_agent_indices=None)`
评估任意策略。通过 `hasattr(actor, 'rnn')` 自动检测 DRQN vs MLP 分支。支持录像（imageio）、max_eval_steps 防卡死、`random_agent_indices` 指定特定 agent 用随机动作。返回胜率/奖励/episode 长度。

#### `class MetricsTracker`
`add(metrics, step)` 追加指标，`get_recent(key, n)` 取最近 n 次均值。

### 5.2 `logger.py`

#### `class Logger`
统一日志。创建 `{algo}_{timestamp}/` 目录，包含 TensorBoard events + `train.log` + 可选 WandB。

#### `format_duration(seconds)`
秒数 → `"XhXXmXXs"` 可读格式。

### 5.3 `rollout.py` — `class RolloutCollector`
通用 rollout 采集器。**未被当前训练脚本使用**（IPPO/MAPPO 训练脚本内联了 rollout 逻辑）。保留供未来可能的统一重构。

---

## 6. 训练脚本 — `scripts/`

三个脚本结构高度对称：

```
1. set_seed(42) + argparse 解析 config + --device
2. 创建 QuadrapongWrapper + 对应 Trainer + Logger
3. 主循环: rollout → GAE → update → eval → save
```

### 6.1 `train_ippo.py`
- 4 agent 自博弈（参数共享）
- 2048 步 rollout → 每步存 local value + obs/action/reward/done
- `_train_ppo_update()`: PPO clipped objective, 10 epochs, mini_batch=64
- 每 50K 步 eval 20 episodes

### 6.2 `train_mappo.py`
- 同 4 agent 自博弈
- 差异：rollout 时用 `CentralizedCritic(global, local)` 算 value
- `_train_mappo_update()`: 同 PPO + `use_popart` 分支（当前 `false`）

### 6.3 `train_qmix.py`
- **仅控制 Team 1**（agent 0和2），Team 2 随机
- 每步 ε-greedy，存 ReplayBuffer（含 RNN hidden state）
- `trainer.update()`: QMIX TD 学习
- Eval 时 Team 2 同样随机

---

## 7. 配置文件 — `src/configs/`

| 文件 | 用途 | 关键参数 |
|------|------|----------|
| `ippo.yaml` | IPPO RAM 全规模 | 10M, rollout=2048, [128,128], lr=3e-4 |
| `ippo_light.yaml` | IPPO RAM 轻量验证 | 2M, rollout=1024, [128,128] |
| `ippo_pixel.yaml` | IPPO 像素（灰度）| 10M, CNN(512), frame_stack=4 |
| `mappo.yaml` | MAPPO RAM 全规模 | 同 IPPO + use_popart=false |
| `mappo_light.yaml` | MAPPO RAM 轻量验证 | 同 IPPO light |
| `mappo_pixel.yaml` | MAPPO 像素（灰度）| 10M, CNN(512), frame_stack=4 |
| `qmix.yaml` | QMIX RAM | 1M, FF-Q, q_hidden=[128], batch=256, γ=0.97, RMSprop, hard update, eps=300K, shaping=0.002 |

**obs_type 选项**: `ram`（128 维向量）| `grayscale_image`（灰度，C=frame_stack×1）| `rgb_image`（彩色，C=frame_stack×3）。像素模式自动选择 CNN 网络。

所有算法共用字段: `env.{obs_type, max_cycles}`, `training.{seed, total_steps, lr, gamma}`, `eval.{eval_interval, num_eval_episodes}`, `logging.{log_interval, save_interval, checkpoint_dir, log_dir}`。

---

## 8. 数据流总览

### IPPO / MAPPO（On-policy）

```
┌──────────┐   obs_batch (N,128)   ┌─────────────┐
│  Env     │ ──────────────────→  │ Actor/Critic │
│ (4 agents)│ ←────────────────── │              │
└──────────┘   actions (N,)       └─────────────┘
      │                                  │
      │ rewards, next_obs                │ values
      ▼                                  ▼
┌──────────────┐                 ┌────────────────┐
│ OnPolicyBuffer │ ←── insert ── │ Rollout loop   │
│ (2048 steps)   │              │ per 2048 steps  │
└──────┬───────┘                 └────────────────┘
       │ GAE(advantages, returns)
       ▼
┌──────────────┐
│ PPO Update   │  10 epochs × mini_batch=64
│ L = clip + V │  → actor.backward + critic.backward
└──────────────┘
```

### QMIX（Off-policy）

```
┌──────────┐   obs_batch (N,128)   ┌─────────────┐
│  Env     │ ──────────────────→  │ DRQN (ε-greedy)│
│ Team1: QMIX│ ←───────────────── │ + random Team2 │
│ Team2: rand│   actions (N,)       └─────────────┘
└──────────┘                               │
      │                          rnn_states (N,64)
      │ rewards, next_obs, done           │
      ▼                                    ▼
┌──────────────┐                 ┌─────────────────┐
│ ReplayBuffer │ ←── push ────── │ per-step loop   │
│ (100K trans)  │   (rnn_states) │                 │
└──────┬───────┘                 └─────────────────┘
       │ sample batch (32)
       ▼
┌──────────────────────┐
│ QMIX Update          │
│ Q_i(s,a) ← DRQN(o, rnn_stored) │
│ Q_tot ← Mixer(Q_i, state)     │
│ L = MSE(Q_tot, r_team1 + γ·Q_tot_target) │
│ soft update target nets        │
└──────────────────────┘
```

---

## 9. 关键设计决策

| 决策 | 原因 |
|------|------|
| IPPO/MAPPO 参数共享 Actor | 减少参数量，加速训练；观测含位置信息可区分 agent |
| 自博弈（IPPO/MAPPO）| 标准做法，互为对手共同进化 |
| QMIX 仅控制 Team 1 | QMIX 设计为纯合作，对抗需指定对手 |
| QMIX ReplayBuffer 存 RNN state | DRQN 需时序上下文，零初始化会导致训练/评估不一致 |
| RAM 观测（非像素）| 128 维，训练快，适合算法迭代 |
| `[128,128]` 网络 | 参考 HARL 默认配置，34K 参数适合简单任务 |
| `cudnn.deterministic=True` | 确保 GPU 训练可复现 |
| Config 驱动 | 所有超参数通过 YAML 传入，不硬编码 |
