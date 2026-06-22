# Quadrapong — 2v2 多智能体乒乓对抗

## 项目概述

多智能体基础课程课题六。控制4个智能体（2v2）完成乒乓球对抗，实现 IPPO / MAPPO / QMIX 三种基线算法并做创新改进。

- 团队：3人
- 硬件：2× RTX 4090 GPU，容器可长时运行
- 截止：本学期不毕业 → 2026-07-01 18:00
- 满分：35分（算法设计21 + 创新14）
- 评分权重：报告60% + 代码&demo 40%

## 目录结构约定

```
quadrapong/
├── CLAUDE.md          # 项目规范（本文件）
├── docs/              # 文档（调研分析、实验记录、报告草稿）
├── src/
│   ├── algos/         # 算法实现（ippo, mappo, qmix, 创新算法）
│   ├── envs/          # 环境包装与注册
│   ├── utils/         # 通用工具（buffer, logger, trainer, evaluator）
│   └── configs/       # 配置文件（yaml/json），每种算法一份
├── scripts/           # 启动/评估/演示脚本
├── checkpoints/       # 训练检查点
├── logs/              # tensorboard / wandb 日志
├── papers/            # 参考论文 PDF
├── results/           # 最终结果
│   ├── models/        # 训练好的模型权重
│   ├── plots/         # 图表（学习曲线、行为对比）
│   └── videos/        # demo 录屏
└── report/            # LaTeX 报告源文件
```

## 环境

- Quadrapong (PettingZoo / ALE): https://ale.farama.org/multi-agent-environments/quadrapong/
- Python 3.10+, PyTorch 2.x, CUDA 12.x
- 依赖通过 pip/conda 安装，不修改系统级包

## 命名与代码规范

- 文件名: snake_case, 类名: PascalCase, 函数/变量: snake_case
- 算法缩写统一大写: IPPO, MAPPO, QMIX（文件名小写: ippo.py）
- 所有超参数通过 config 文件传入，不硬编码
- 每个算法独立一个文件，共享 base agent 基类
- 训练、评估、推理逻辑分离

## 文档纪律

- **所有工作必须入 docs/**：每次代码撰写、训练启动、实验结果、决策变更，必须在 `docs/` 下留下记录
- 代码任务 → `docs/code_log.md`：记录新增/修改了哪些模块、关键设计决策
- 训练任务 → `docs/train_log.md`：记录训练启动时间、配置、GPU、PID、预估完成时间
- 实验结论 → `docs/experiment_log.md`：记录每次实验的假设、结果、分析和下一步
- 每个条目带时间戳（`## 2026-06-19 15:30` 格式）
- 事后回顾时，`docs/` 里的三份日志应能完整还原整个项目过程
- CLAUDE.md 的「当前进度」节保持同步更新

## 实验纪律

- 每次实验前确认 config，记录到日志
- 固定随机种子（42），确保可复现
- 每 10K steps 做一次评估并记录指标
- 结果图统一用 matplotlib/seaborn，风格一致
- 论文引用用 BibTeX，放在 report/refs.bib

## 红线（必须先问）

- 删除文件/目录/git历史
- 修改 conda 环境或全局包
- git push（容器内不配远程）
- 下载超过 5GB 的数据集或模型

## 阶段划分

| Phase | 内容 | 工期 | 截止 | 状态 |
|-------|------|------|------|------|
| 1 | 基础框架搭建 | 3天 | 6/21 | ✅ |
| 2 | IPPO 实现与调优 | 3天 | 6/23 | ✅ RAM 10M 完成 |
| 3 | MAPPO 实现与调优 | 3天 | 6/25 | ✅ RAM 10M 完成 |
| 4 | QMIX 实现与适配 | 3天 | 6/27 | ✅ RAM 1M 完成 |
| 5 | 像素输入扩展 | — | — | ✅ 三算法 CNN 支持 |
| 6 | 创新点：对手池自博弈 | 3天 | 6/29 | ⏳ 方案已定，待实现 |
| 7 | 报告撰写 & Demo | 2天 | 7/1 | ⏳ |

## 创新点：对手池自博弈 (Opponent Pool Self-Play)

### 动机
标准自博弈导致策略固化——评估中发现 T2(agents 1,3) 必胜 100%，因 spawn 位置差异形成固定角色（进攻/防守），策略不再泛化。

### 方案
训练时 Team 2 不再总是当前策略，而是从**对手池**随机采样：

| 对手类型 | 来源 | 权重 | 作用 |
|----------|------|------|------|
| 历史自身 | 每 N 步保存的 past checkpoint | 60% | 防止遗忘，打破角色固化 |
| 随机策略 | ε=1.0 纯随机 | 20% | 基础探索，避免策略坍塌 |
| 当前自身 | 正在训练的 policy | 20% | 保持自博弈的对抗压力 |

### 实现要点
- 对手池容量 K=5-10 个 checkpoint
- 每 200K 步保存一份 checkpoint 入池（FIFO）
- 每个 episode 开始时随机采样对手（加权）
- 对手只做推理，不更新参数
- Team 1 始终使用当前训练策略
- 仅对 IPPO/MAPPO 实施（QMIX 为 off-policy，不适用）

### 预期效果
- 打破 T1/T2 角色固化，策略泛化到双方向
- 对抗历史版本防止策略震荡
- 保持自博弈压力，不降低训练效率

## 当前进度

- [x] Phase 1-4: 三算法 RAM 实现 + 全规模训练 ✅
- [x] Phase 5: 像素输入 (CNN) 三算法支持 ✅
- [x] 三算法 RAM 对抗评估完成 ✅
- [x] Phase 6: 像素训练 — IPPO ✅(随机) MAPPO ⏳(无用) QMIX ❌(爆炸)
- [x] Phase 7: 创新点 — 对手池自博弈 ✅ IPPO pool 完成+评估, MAPPO pool 训练中
- [ ] Phase 8: LaTeX 报告 + Demo 录屏

## 训练布局

| 算法 | PID | 设备 | obs | 网络 | 目标步数 | 速率 | 预估 |
|------|-----|------|-----|------|----------|------|------|
| IPPO | ✅ | CPU | RAM [128,128] | MLP | 10M | 333/s | 7.5h |
| MAPPO | ✅ | GPU 1 | RAM [128,128] | MLP | 10M | 211/s | 13.1h |
| QMIX | ✅ | GPU 0 | RAM [128,128] | FF-Q+Mixer | 1M | 59/s | ~3.5h |
| IPPO pixel | ✅ | GPU 1 | grayscale(3,84,84) | CNN | 10M | 155/s | 17.9h |
| MAPPO pixel | 395215 | GPU 0 | grayscale(3,84,84) | CNN batched | 10M | 91/s | ~21h |
| QMIX pixel | ❌ | GPU 0 | grayscale(3,84,84) | CNN+Mixer | 1M | 12/s | 已终止 |
| **IPPO pool** | 560840 | CPU | RAM [128,128] | MLP | 10M | 350/s | ~4.8h |
| **MAPPO pool** | 560904 | GPU 1 | RAM [128,128] | MLP | 10M | 240/s | ~8.5h |

## 对手策略说明

- **IPPO/MAPPO**: 使用自博弈（self-play），4 agent 共享同一策略互相对抗。标准做法。
- **QMIX**: 控制 Team 1，Team 2 使用随机策略。QMIX 原为纯合作设计。
- **拓展方向**: 对手多样性实验（自博弈 vs random vs 历史模型池）可作为创新点。
