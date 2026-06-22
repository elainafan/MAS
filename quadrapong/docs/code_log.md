# 代码撰写日志

> 代码架构与各文件功能详见 [`code_architecture.md`](code_architecture.md)。

## 2026-06-22 11:00 — Bug 修复 + plot_eval.py 重构

### 修复
- **`get_algo_name` bug** (`eval_unified.py:114-132`): `ippo_pixel`/`mappo_pixel` 被 `ippo`/`mappo` 子串误匹配。修复：添加精确匹配字典，fallback 循环中长前缀优先
- **分数非零和诊断**：ALE Quadrapong 环境在少数步中仅分配 partial reward（如两 agent 各-1，另两 agent 得 0），导致 50 局累计偏差 ~±8。非代码 bug，胜负判定不受影响

### plot_eval.py 重构
- **双向合并**：forward/reverse 两方向合并到同一张图（不同颜色/线型），从 30 张图缩减为 15 张
- **轨迹对齐修复**：改用 LOCF（前向填充）替代精确网格匹配，episode 最终帧不再丢失
- **max_steps 自适应**：从数据中推导，不再硬编码 20000
- **确定性提示**：std=0 的对阵标注 "(deterministic)"
- 修复 `draw` key 缺失（traj npz 不存 draw 字段，从 WR 反算）

### 烟测
- `get_algo_name`: 所有 8 目录名测试通过
- `plot_eval.py`: batch 1 数据 15 张图正常生成

**新增**:
- `run_cross_matchup()` — 跨观测对抗评估。双环境锁步运行（相同 seed + 相同动作 → ALE 确定性保证状态一致），每个模型从其专属环境获取对应观测类型
- `_step_env()` / `_obs_to_tensor()` / `_finalize_episode()` / `_make_result()` — 复用辅助函数，减少 `run_matchup` 和 `run_cross_matchup` 间的重复
- Reward 散度运行时断言：每步校验两个环境产生的 reward 完全一致
- 跨观测路径 try/except 异常保护
- `get_algo_name()` 支持 `ippo_pool`/`mappo_pool`/`qmix_pixel` 等目录名

**验证**: 两个环境（RAM + pixel）2000 步零 reward 偏差，确认 ALE 确定性同步

**Subagent 审查**: 发现 reward 散度无检测、跨观测路径缺异常保护 → 已修复，二次烟测通过

## 2026-06-21 10:30 — 对手池自博弈实现

**新增文件**:
- `src/utils/opponent_pool.py` — OpponentPool 类 + `get_opponent_actions()` 辅助函数
  - FIFO 容量 K=5，加权采样（60%历史 / 20%随机 / 20%当前）
  - 模型缓存避免反复加载，深拷贝保证状态独立
  - `get_opponent_actions` 支持 RAM 和 CNN（通过 `obs_shape` 参数）
- `scripts/train_ippo_pool.py` — IPPO 对手池训练
- `scripts/train_mappo_pool.py` — MAPPO 对手池训练
- `src/configs/ippo_pool.yaml`、`src/configs/mappo_pool.yaml`

**设计要点**:
- Team 1 (agents 0,2) 使用当前训练策略，正常更新
- Team 2 (agents 1,3) 使用 pool 采样的对手策略（推理模式，冻结参数）
- Buffer 仅存储 Team 1 数据（num_agents=2），全局状态保留完整 4-agent 观测
- 每个 episode 结束时重新采样对手
- 每 200K 步保存 actor 到对手池（FIFO）
- 评估：同时打 random 对手和 self-play 参照
- 保留原始训练脚本接口不变

**修改文件**:
- `src/utils/buffer.py` — OnPolicyBuffer 新增 `global_obs_dim` 参数（向后兼容，默认 `obs_dim * num_agents`）

**烟测**: 两轮通过（运行 60s 无报错）

**Subagent 审查**: PASS，仅发现 CNN 模式下 `get_opponent_actions` 潜在 bug（已修复）和初始终 opponent 采样问题（已修复）

## 2026-06-20 17:35 — QMIX pixel 训练支持

- `qmix.py`: 新增 `agent_q_net`/`global_encoder` 可选参数，`_maybe_reshape` 方法，`update`/`get_actions`/`save`/`load` 支持 CNN
- `train_qmix.py`: pixel env 构建、CNN 网络创建、obs flatten 存入 buffer、eval env 传参
- `networks.py`: `CNNFFQNetwork` 新增 `get_logits()` 方法（evaluator 兼容）
- 修复：global_encoder 梯度被 `no_grad` 阻断（移到外部）、evaluator 对 CNNFFQNetwork 崩溃
- 新增 `qmix_pixel.yaml`，日志文件夹 `qmix_pixel/`

## 2026-06-20 17:00 — 大一统 eval 脚本

- `eval_unified.py` — 替换 `eval_matchup.py` + `eval_qmix_matchup.py`
- 自动检测网络类型（IPPO/MAPPO vs QMIX）和输入类型（RAM 128 vs CNN pixel）
- 自动检测 MLP 层数（1-hidden QMIX vs 2-hidden IPPO/MAPPO）
- 支持任意数量 checkpoint 的全配对 + 自动交换双方位置
- 审查修复：2层检测改用 `mlp.4.weight`、加载时复用 ckpt dict、env try/finally、algo name 检测增强、import 移到顶部

## 2026-06-20 15:24 — MAPPO critic 批处理优化

- `train_mappo.py`: rollout 和 last-value 的集中式 critic 前向从 `for i in range(4)` 改为单次 `expand + batched forward`
- `mappo.py` `train_on_buffer`: 同步修改（死代码，不参与训练）
- 数值等价性验证：新旧差异 < 1e-8
- CPU 烟测: 9.2→21.4/s (2.3x), GPU 实测: 96→106/s (+10%)

## 2026-06-20 14:29 — 84×84 resize + frame_stack 3 加速

### 改动
- `quadrapong_env.py`: 新增 `resize_dim` 参数，`cv2.resize` 在帧堆叠前将 210×160 缩至 84×84。修复 cv2 单通道灰度图丢维度问题（`.ndim==2` → 加回 `np.newaxis`）
- `networks.py`: `CNNEncoder` 默认 `input_h/input_w` 从 210/160 改为 84/84，feat_dim 从 23,936 降至 3,136
- `evaluator.py`: 视频环境传递 `resize_dim` 参数
- `train_ippo.py/mappo.py`: 所有 `QuadrapongWrapper` 调用传递 `resize_dim`
- 配置文件: `ippo_pixel.yaml`, `mappo_pixel.yaml` — `frame_stack: 4→3`, 新增 `resize_dim: 84`
- 类默认值 `resize_dim=None`（由 config 显式控制）

### 加速效果
| 算法 | 210×160 CPU | 84×84 CPU | CPU加速 | 210×160 GPU | 84×84 GPU | GPU加速 |
|------|------------|-----------|---------|-------------|-----------|---------|
| IPPO pixel | 6.4/s | 29/s | 4.5x | 58/s | 210/s | 3.6x |
| MAPPO pixel | 3.9/s | 9.2/s | 2.4x | 15/s | 96/s | 6.4x |

IPPO pixel 10M ETA: 48h→13h, MAPPO pixel 10M ETA: 7.6天→29h

## 2026-06-20 12:19 — CNN 像素输入烟测通过 + 训练启动

### 代码审查发现与修复
- **I1 (CRITICAL)**: `quadrapong_env.py` — `obs_dim` 在像素分支未赋值，导致 `AttributeError`。修复：`self.obs_dim = C*H*W`（展平像素维度）。
- **I2**: `train_mappo.py:141` — buffer insert 传入 4D `obs_batch` 而非 2D `obs_batch_flat`。修复为 `obs_batch_flat`。
- **I3**: `ippo.py` `train_on_buffer` — CNN 模式下缺 reshape，传入 MLP 的 tensor 维度错误。添加 `_maybe_reshape`。
- **I4**: `mappo.py` `train_on_buffer` — 同上 + `global_b` 也需要 reshape。
- **I5**: `quadrapong_env.py` — 像素 obs 缺 `/255.0` 归一化（RAM 分支有归一化但像素分支漏掉）。修复：`_stack_frames(...).astype(np.float32) / 255.0`。

### 烟测结果
| 测试 | Steps | 耗时 | 结果 |
|------|-------|------|------|
| IPPO pixel (CPU) | 1000/200 | 2m36s | PASS |
| MAPPO pixel (CPU) | 1000/200 | 4m14s | PASS |

## 2026-06-20 00:31 — QMIX 稀疏奖励修复

**诊断**: Quadrapong 进球率仅 0.2%，batch_size=32 时 94% 批次无奖励信号，TD 自举导致 Q→0 不动点。

**修改**:
- `src/configs/qmix.yaml`: `batch_size` 32→256, `epsilon_decay` 50K→300K, `target_update_tau` 0.005→0.001
- 见 `docs/train_log.md` 和 `docs/experiment_log.md` 详细分析

## 2026-06-20 02:33 — QMIX FF-Q + Reward Shaping 重构

**根因**: Bootstrap collapse — 99.8% transition reward=0，TD 自举锁死 Q→0。batch=256 仅延缓，不根除。

**修改**:
- `src/utils/networks.py` — 新增 `FFQNetwork`（纯 MLP，单返回值），`DRQN` 保留为已弃用
- `src/algos/qmix.py` — 全面重写：`FFQNetwork` 替代 `DRQN`，移除所有 RNN 逻辑，`get_actions` 和 `update` 大幅简化
- `scripts/train_qmix.py` — 移除 RNN 追踪，添加 reward shaping（Team 1 +0.002/step），移除 `rnn_hidden` trainer 参数
- `src/utils/evaluator.py` — 新增 `is_ppo` 检测，支持 FFQNetwork 单返回值
- `src/configs/qmix.yaml` — gamma 0.99→0.95

## 2026-06-20 09:44 — QMIX γ=0.99 + PER + 硬更新

**根因**: gamma=0.95 有效视界仅 ~30 步，shaping 只提供常数偏移。epsilon→0 后缓冲区 goal transition 被稀释。

**修改**:
- `src/configs/qmix.yaml`: gamma 0.95→0.99, `target_update_tau`→`target_update_interval: 200`
- `src/utils/buffer.py` — `ReplayBuffer` 重写为 PER：优先级采样 + importance weights + `update_priorities()`
- `src/algos/qmix.py` — 软更新→硬更新；loss 使用 PER weights；移除 `_soft_update`，增加 `_copy_update`
- `scripts/train_qmix.py` — shaping 0.002→0.005, `target_update_interval` 替换 `target_update_tau`

## 2026-06-20 10:30 — QMIX shaping 回退 + PER alpha↓

**问题**: shaping=0.005 + PER(α=0.6) + γ=0.99 导致梯度爆炸 (q_loss=5.6e19)。
**修复**: shaping 0.005→0.002, PER alpha 0.6→0.4。保持 γ=0.99 + hard update。

## 2026-06-19 15:00 — 初始框架搭建

### 环境包装
- `src/envs/quadrapong_env.py`：QuadrapongWrapper
  - 封装 PettingZoo quadrapong_v4，统一接口
  - 支持 ram / rgb_image / grayscale_image 三种观测
  - 像素模式支持帧堆叠（frame_stack=4）
  - 提供团队接口：get_team_rewards(), get_team_agents()
  - RAM obs 归一化到 [0,1]

### 网络模块
- `src/utils/networks.py`：
  - StochasticActor：离散动作随机策略（MLP + 可选 GRU），用于 PPO
  - Critic：独立值函数
  - CentralizedCritic：集中式 Critic（全局状态 + 局部观测），用于 MAPPO
  - CNNActor：像素观测的 CNN 策略
  - DRQN：Deep Recurrent Q-Network，用于 QMIX
  - MixingNetwork：单调混合网络 + Hypernetworks，强制 ∂Q_tot/∂Q_i ≥ 0
  - make_mlp()：通用 MLP 工厂函数

### 经验缓冲
- `src/utils/buffer.py`：
  - OnPolicyBuffer：PPO rollout 缓冲，支持 GAE 计算
  - ReplayBuffer：QMIX 经验回放，定长循环

### 评估与日志
- `src/utils/evaluator.py`：多轮评估，胜率/奖励/episode 长度统计，支持视频录制
- `src/utils/logger.py`：统一日志（TensorBoard + 文件 + 可选 WandB），时间戳运行目录

### 算法实现
- `src/algos/ippo.py`：独立 PPO，参数共享 Actor，独立 Critic
- `src/algos/mappo.py`：CTDE PPO，集中式 Critic + PopArt 值归一化
- `src/algos/qmix.py`：Off-policy CTDE，DRQN + MixingNetwork + Double Q-learning

### 配置与脚本
- `src/configs/ippo.yaml`、`mappo.yaml`、`qmix.yaml`：各算法独立配置
- `scripts/train_ippo.py`、`train_mappo.py`、`train_qmix.py`：训练入口

### 环境
- conda env: cv-lab3 (Python 3.11.15, PyTorch 2.7.0+cu128)
- 依赖：pettingzoo 1.24.3, multi-agent-ale-py 0.1.12, gymnasium 1.3.0
- ROM: pong.bin (AutoROM installed)

### Bug 修复
- `quadrapong_env.py:reset()` — 修复 render_mode 未传入内部 PettingZoo 环境的问题
- `src/utils/buffer.py:compute_gae()` — **P0**: GAE off-by-one, `dones[t+1]` → `dones[t]`。直接导致 IPPO value_loss 坍缩为 0
- `src/algos/qmix.py:update()` — **P0**: Double Q 使用 `agent_q(next_obs)` 在线网络 Q(s') 选动作
- `src/utils/evaluator.py:evaluate()` — **P0**: deterministic 路径检测网络类型 (mlp vs DRQN)，QMIX eval 不再崩溃

### P1/P2 修复（2026-06-19）
- `src/utils/evaluator.py` — **#7**: 加 `max_eval_steps` 截断，防 deterministic 卡死 100K 帧
- `scripts/train_ippo.py` — **#8**: `last_done_arr` 跟踪最后一步 done_arr
- `scripts/train_mappo.py` — **#8**: 同上；**#9**: 移除 MAPPO rollout 后重复的 value 重算
- `scripts/train_qmix.py` — **#11**: 加 `--device` 参数
- `src/algos/mappo.py:PopArt` — **#4**: `update()` 中 `.cpu()` 防止设备不匹配
- `src/envs/quadrapong_env.py` — **#5**: `_frame_buffers.clear()` → `= None`；**#10**: obs space 修正为 float32 [0,1]

### 已知影响
当前 IPPO/MAPPO 轻量训练（PID 928387/928388）在修复前启动，其 value_loss 和 GAE 受旧版 bug 影响。

### 设计决策
- 使用 RAM 观测（128维）而非像素，优先训练速度
- 所有 agent 共享 Actor 网络参数（参数共享）
- 三算法独立训练脚本（非统一入口），便于各自调参
- QMIX 的 state 定义为所有 agent obs 的拼接（128*4=512维）
- 训练脚本内置训练循环而非调用 trainer 方法，便于灵活修改

## 2026-06-19 16:20 — 训练加速
- 降低 total_steps (10M→2M)、rollout_steps (2048→1024)、ppo_epochs (10→5)、hidden_dims ([256,256]→[128,128])
- 有效速率从 ~20 提升至 ~500-600 steps/s

## 2026-06-19 16:30 — CPU/GPU 测速 & 训练脚本加 --device
- `scripts/train_ippo.py`、`train_mappo.py` 增加 `--device` 参数
- IPPO 跑 CPU (600 steps/s)，MAPPO 跑 GPU 1 (320 steps/s)，GPU 0 留给 QMIX
