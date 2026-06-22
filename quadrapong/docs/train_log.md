# 训练日志

## 2026-06-21 19:10 — IPPO pool 训练完成 ✅

**最终状态**: 10M 步, 7h35m, 350/s, ent=1.09, p_loss=-0.025, vs Random T1=75%, vs Self T1=100%
**对手分布**: history:261, random:95, current:85
**checkpoint**: `checkpoints/ippo_pool/ippo_pool_final.pt`

vs 原版 IPPO RAM (ent=1.57, vs Random T1=65%)：熵更低、vs Random 胜率更高、位置固化被打破。

## 2026-06-21 14:20 — 对手池训练中期状态

**IPPO pool** (PID 560840, CPU): 3.96M/10M (40%), 350/s, ent=1.13, p_loss=-0.025, 3h10m
**MAPPO pool** (PID 560904, GPU 1): 2.65M/10M (27%), 240/s, ent=1.25, p_loss=-0.021, vs Random T1=60%, 3h09m
**MAPPO pixel** (PID 395215, GPU 0): 7.61M/10M (76%), 91/s, ent=1.77, 22h54m — 策略随机

对手池机制显著加速收敛：entropy 从 1.79 降至 1.13-1.25（原版需全 10M 步），p_loss 信号强度是原版 2-3 倍。

## 2026-06-21 11:09 — 对手池训练重启（修复 log_eval bug）

修复 `log_eval` 不支持 `prefix` 参数的 bug 后重启。

| 算法 | PID | 设备 | obs | 网络 | 目标 | 备注 |
|------|-----|------|-----|------|------|------|
| IPPO pool | 560840 | CPU | RAM [128,128] | MLP | 10M | |
| MAPPO pool | 560904 | GPU 1 | RAM [128,128] | MLP | 10M | |

## 2026-06-21 10:38 — 对手池自博弈训练启动

**方案**: 对手池（60%历史/20%随机/20%当前），每 200K 步入池，每 episode 采样对手。Team 1 正常训练，Team 2 冻结推理。

| 算法 | PID | 设备 | obs | 网络 | 目标 | 备注 |
|------|-----|------|-----|------|------|------|
| IPPO pool | 559756 | CPU | RAM [128,128] | MLP | 10M | buffer 仅存 Team 1 (2 agents) |
| MAPPO pool | 559757 | GPU 1 | RAM [128,128] | MLP | 10M | 同上 |

MAPPO pixel (PID 395215, GPU 0) 仍在跑但无用，等待 kill。

## 2026-06-21 00:35 — QMIX pixel 训练终止 ❌

**终止原因**: 训练爆炸，不可恢复。q_loss 在 48~39,665 间振荡，q_tot 在 18-50 徘徊。eval 对随机对手仅 25% 胜率。内存占用 134 GB（ReplayBuffer 存储 100K×84K 维 pixel state）。继续跑无意义。

**最终状态**: 277K/1M (27.7%)，耗时 6h14m，速率 ~12/s。

QMIX 在 pixel 和 RAM 两种观测下均因稀疏奖励 + TD bootstrap 无法收敛。视为负结果，报告中如实记录。

## 2026-06-20 17:38 — QMIX pixel 训练启动

| 算法 | PID | 设备 | obs | 网络 | 目标 | 备注 |
|------|-----|------|-----|------|------|------|
| QMIX pixel | 468899 | GPU 0 | grayscale(3,84,84) | CNNFFQ+GlobalEnc+Mixer | 1M | 新日志文件夹 qmix_pixel/ |

## 2026-06-20 15:25 — MAPPO critic 批处理 + 重启

批处理优化后 MAPPO pixel 重启 (PID 395208, GPU0): 106/s, ETA ~26h。
IPPO pixel 未中断 (PID 255742, GPU1): 163/s, 551K/10M, ETA ~17h。

## 2026-06-20 14:29 — 84×84 resize 像素训练重启 (已被15:25重启替代)

84×84 resize + frame_stack 4→3 加速后重启训练。

| 算法 | PID | 设备 | obs | 网络 | 目标 | 速率 | ETA |
|------|-----|------|-----|------|------|------|-----|
| IPPO pixel | 255735 | GPU 1 | grayscale(3,84,84) | CNN | 10M | 210/s | ~13h → 6/21 凌晨 |
| MAPPO pixel | 256086 | GPU 0 | grayscale(3,84,84) | CNN | 10M | 96/s | ~29h → 6/21 傍晚 |
| QMIX | 3649157 | GPU 0 | RAM(128) | FF-Q+Mixer | 1M | 59/s | ~2h → 16:30 |

vs 旧配置加速：IPPO 3.6x GPU, MAPPO 6.4x GPU

## 2026-06-20 12:20 — CNN 像素训练启动 (旧210×160, 已废弃)

| 算法 | PID | 设备 | obs | config | 网络 | 目标步数 | 速率 | 预估 |
|------|-----|------|-----|--------|------|----------|------|------|
| IPPO pixel | 3917639 | GPU 1 | grayscale(4,210,160) | ippo_pixel.yaml | CNNActor+CNNCritic | 10M | 58/s | ~48h |
| MAPPO pixel | 3918376 | GPU 0 | grayscale(4,210,160) | mappo_pixel_4m | CNNActor+CNNCentralizedCritic | 4M | 15/s | ~74h |
| QMIX | 3649157 | GPU 0 | ram(128,) | qmix.yaml | FFQNetwork+MixingNet | 1M | ~358/s | 171K/1M |

GPU 0 同时跑 MAPPO pixel(~3.0GB) + QMIX(~0.9GB)。MAPPO 从 10M 降至 4M（集中式critic速度慢，10M 需 ~7.6 天不合理）。

### 烟测结果 (CPU)
- IPPO pixel: 1000 steps / 2m36s ≈ 6.4 step/s ✅
- MAPPO pixel: 1000 steps / 4m14s ≈ 3.9 step/s ✅

### GPU 实测速度
- IPPO pixel GPU: 58 steps/s（CPU 6.4/s → ~9x 加速，瓶颈仍在 ALE 仿真）
- MAPPO pixel GPU: 15 steps/s（集中式critic每步需5次CNN前向：1 actor + 4 critic）

## 2026-06-19 16:43–19:41 — 轻量训练（已完成，用于收敛验证）

### IPPO 轻量
- **配置**: `src/configs/ippo_light.yaml`
- **设备**: CPU, PID 928387
- **步数**: 1,050,624 / 2M (52.5%), 运行 ~3h
- **指标**: entropy 1.76→1.69, ep_len 0→599, episodes 44
- **状态**: 已停止。指标见 `docs/experiment_log.md`

### MAPPO 轻量
- **配置**: `src/configs/mappo_light.yaml`
- **设备**: CPU, PID 928388 (实际跑了 CUDA)
- **步数**: 700,416 / 2M (35%), 运行 ~3h
- **指标**: entropy 1.75→1.60, ep_len 0→431, episodes 29
- **状态**: 已停止

### 结论
方向正确，策略在学。全规模训练使用正式配置。

## Config 命名规范化
- `ippo_light.yaml` / `mappo_light.yaml` — 轻量快速验证
- `ippo.yaml` / `mappo.yaml` — 全规模正式训练（10M, 2048 rollout, [256,256]）
- `qmix.yaml` — QMIX

## 2026-06-19 20:36 — 全规模训练启动（最终方案）

**方案**: IPPO 独占 CPU，MAPPO/GPU 1，QMIX/GPU 0。避免三进程互抢 CPU。

### IPPO
- **设备**: CPU, PID 3625766
- **配置**: `src/configs/ippo.yaml`（10M, rollout=2048, ppo_epochs=10, [256,256]）
- **速率**: ~310 steps/s, 预估 ~9 小时

### MAPPO
- **设备**: GPU 1, PID 3625767
- **配置**: `src/configs/mappo.yaml`（10M, 同上）
- **速率**: ~267 steps/s, 预估 ~10 小时

### QMIX
- **设备**: GPU 0, PID 3626268
- **配置**: `src/configs/qmix.yaml`（10M, buffer=100K, 与 IPPO/MAPPO 等量环境步数）
- **速率**: ~100 steps/s, 预估 ~28 小时

### 状态
- 三个进程均正常运行
- Cron 由用户手动管理

## 2026-06-19 20:47 — 网络缩小至 [128,128] 重启训练

**原因**: 参考 HARL 默认配置，对于 128 维 RAM + 6 动作的简单任务，[256,256] 过度参数化（Actor ~100K vs ~34K）。缩小网络可提速并提高收敛稳定性。

**变更**: IPPO/MAPPO 的 actor_hidden 和 critic_hidden 从 `[256,256]` → `[128,128]`。QMIX 配置不变（已与 pyMARL 默认一致）。

### IPPO
- **设备**: CPU, PID 3626745
- **配置**: `src/configs/ippo.yaml`（10M, [128,128], rollout=2048）
- **速率**: ~238 steps/s（含首次 eval，待稳定）, 预估 ~12h

### MAPPO
- **设备**: GPU 1, PID 3626746
- **配置**: `src/configs/mappo.yaml`（10M, [128,128], rollout=2048）
- **速率**: ~194 steps/s, 预估 ~14h

### QMIX
- **设备**: GPU 0, PID 3626807
- **配置**: `src/configs/qmix.yaml`（10M, 不变）
- **速率**: ~24 steps/s（首次 eval 后偏低，待稳定）, 预估 ~116h

## 2026-06-19 23:44 — QMIX reward bug 修复后重启（已替换）

## 2026-06-20 00:30 — QMIX RNN state 修复 + 多项代码改进

**RNN state 丢失修复**: ReplayBuffer 现在存储每步的 GRU hidden state，训练时用存储的状态初始化 RNN，处理 obs 后链式传入 next_obs。不再每步从零开始。

**其他修复**:
- Config: IPPO/MAPPO 删除未使用的 `opponent` 字段（自博弈无对手概念）
- `set_seed`: 增加 `cudnn.deterministic` 确保 GPU 可复现
- Evaluator: `hasattr(actor, 'rnn')` 替代脆弱的 `mlp[0].in_features` 检测
- IPPO/MAPPO `get_actions`: deterministic 模式跳过 RNN 的 bug 修复
- MAPPO `_train_mappo_update`: 添加 PopArt 支持（`use_popart=True` 时生效）
- QMIX eval: 移除无效的预热循环

### QMIX
- **设备**: GPU 0, PID 3632092
- **配置**: `src/configs/qmix.yaml`（1M）
- **开始时间**: 00:03
- **速率**: 待观测

## 2026-06-20 00:31 — QMIX 稀疏奖励修复后重启

**根因**: 诊断 subagent 确认非代码 bug，而是训练动态问题。Quadrapong 进球率仅 0.2%，batch_size=32 时 94% 批次无奖励信号。TD 自举 `0 + γ·Q_target` → Q→0 不动点。

**修复**:
- `batch_size`: 32 → **256**（批次含信号概率 6.3% → 40.6%）
- `epsilon_decay`: 50K → **300K**（延长探索期，积累更多进球 transition）
- `target_update_tau`: 0.005 → **0.001**（拉长目标滞后，防止目标坍缩）

### QMIX
- **设备**: GPU 0, PID 3633937
- **配置**: `src/configs/qmix.yaml`（1M, batch=256, eps_decay=300K, tau=0.001）
- **开始时间**: 00:31
- **速率**: ~39/s
- **结论**: 失败。batch=256 仅延缓了 Q→0 崩溃，未根除。225K 时 q_loss 再次归零。

## 2026-06-20 02:33 — QMIX FF-Q + Reward Shaping 重构

**根因**: 稀疏奖励下 bootstrap collapse 是数学必然——99.8% transition reward=0，`target = γ·Q_target` 自举锁死在 Q=0。增大 batch 只延缓不解决。

**架构变更**:
1. DRQN → **FFQNetwork**（RAM 全观测，无需 GRU）：`networks.py` 新增类，`qmix.py` 全部重写
2. 移除所有 RNN state 逻辑：`get_actions` 只返回 actions，`update` 简化为纯 MLP forward
3. **Reward Shaping**: Team 1 agent 每步 +0.002（~2.0/episode vs 稀疏 ±1），提供密集梯度
4. gamma: 0.99 → **0.95**（缩短自举链）
5. Evaluator: 新增 `is_ppo` 检测，支持 FFQNetwork 的单返回值

### QMIX
- **设备**: GPU 0, PID 3636340
- **配置**: `src/configs/qmix.yaml`（1M, FF-Q, batch=256, γ=0.95, shaping=0.002/step）
- **开始时间**: 02:33
- **速率**: ~58/s
- **结论**: 失败。gamma=0.95 有效视界 ~30 步太短。365K 时 winrate 80%，470K 时跌至 10%。

## 2026-06-20 09:44 — QMIX γ=0.99 + PER + 硬更新 + shaping↑

**根因**: gamma=0.95 有效视界 ~30 步，进球信号几乎完全衰减。shaping 提供常数偏移而非动作梯度。epsilon→0 后缓冲区 goal transition 被稀释。

**修复**:
1. gamma: 0.95 → **0.99**（视界 30→400 步）
2. 软更新 → **硬更新**（每 200 步拷贝 target）
3. ReplayBuffer → **PER**（按 TD error 优先级采样）
4. shaping: 0.002 → **0.005/step**（γ=0.99 下下限从 0.08→1.0）

### QMIX
- **设备**: GPU 0, PID 3646201
- **配置**: `src/configs/qmix.yaml`（1M, FF-Q, γ=0.99, PER, hard update, shaping=0.005）
- **开始时间**: 09:44
- **速率**: ~57/s
- **结论**: 失败。梯度爆炸，q_loss→5.6e19。

## 2026-06-20 10:49 — QMIX γ=0.99 + 均匀采样 + hard update（第六次重启）

**根因**: PER 与 γ=0.99 组合产生反馈循环（高 TD error → 优先采样 → 更高 TD error → 爆炸）。前两次 PER 均爆炸（α=0.6 和 α=0.4）。

**修复**: PER α→0.0（等效均匀采样）。保持 γ=0.99 + shaping=0.002 + hard update。这组合在之前（γ=0.95）达到 80% winrate，γ=0.99 修复视界问题后应更稳定。

### QMIX
- **设备**: GPU 0, PID 3648098
- **配置**: γ=0.99, FF-Q, uniform, hard update, shaping=0.002
- **开始时间**: 10:49
- **速率**: 65/s
- **结论**: 爆炸。q_loss→3e6, q_tot→10000。γ 过高是三次爆炸共同因素。

## 2026-06-20 11:40 — QMIX γ=0.97 + RMSprop（第七次重启）

**修复**: gamma 0.99→0.97（shaping 下限 0.2→0.067），Adam→RMSprop（PyMARL 标配）。
- **设备**: GPU 0, PID 3649157
- **开始时间**: 11:40

## 2026-06-20 05:08 — IPPO 完成 ✅
- **最终步数**: 10M
- **最终熵**: 1.57
- **checkpoint**: `checkpoints/ippo/ippo_final.pt`

## 2026-06-20 10:05 — IPPO vs MAPPO 对抗测试结果（100 局/对阵）

| 对阵 | Team 1 (T1) | Team 2 (T2) | T1 胜率 | T2 胜率 | 平局 | T1 Reward | T2 Reward |
|------|------------|------------|---------|---------|------|-----------|-----------|
| IPPO vs Random | IPPO | Random | **65%** | 23% | 12% | +3.8 | -3.8 |
| Random vs IPPO | Random | IPPO | 46% | 36% | 18% | +0.4 | -0.4 |
| MAPPO vs Random | MAPPO | Random | 40% | 45% | 15% | +0.1 | -0.1 |
| IPPO vs MAPPO | IPPO | MAPPO | **100%** | 0% | 0% | +16.0 | -16.0 |
| MAPPO vs IPPO | MAPPO | IPPO | 0% | 0% | 100% | 0.0 | 0.0 |

**结论**:
1. **自博弈位置分化**: IPPO 在 Team 1（进攻角色）和 Team 2（防守角色）表现截然不同。攻击位 65% 胜随机、100% 胜 MAPPO；防守位以铁桶阵为主（大量平局）。这是共享网络根据位置编码涌现的角色分化。
2. **IPPO > MAPPO**: 攻击位 IPPO 完胜 MAPPO（+16 reward），防守位 IPPO 零封 MAPPO。MAPPO 的集中式 Critic 导致 Actor 依赖训练时对手模式，泛化差且无角色分化。
3. **对随机**: IPPO 攻击位 65%、防守位 36%（平均 ~50%），MAPPO ~40%。自博弈不优化"打败随机"。
4. **所有局打满 5000 步**: 策略偏防守（未收敛的 Pong 策略倾向于不丢分而非得分），需更长时间训练或不同奖励设计才能产生进攻性。

结果存入 `results/matchup_results.txt`。

## 待启动

- [ ] 三算法全面对比评估
- [ ] 创新点实验
- [ ] 对手多样性实验（random opponent vs 自博弈 vs 历史模型池）— 拓展方向
- [ ] 报告 & demo
