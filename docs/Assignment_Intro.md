# 多智能体基础 — 组队课题

## 作业提交

最终提交需将以下内容压缩打包，命名为 `学号+姓名`：

**截止时间：**
- 本学期毕业的同学：**6月19日 18:00**
- 本学期不毕业的同学：**7月1日 18:00**

**提交内容：**

1. **研究性报告**（PDF，中英文皆可，建议不少于3页）
   - 内容包括：研究方案、训练过程与参数、核心代码、实验结果分析、有趣的现象和启发等，必要时配图说明，注意参考文献引用。
   - LaTeX 版：使用 ICML 2022 模板（https://media.icml.cc/Conferences/ICML2022/Styles/icml2022.zip）
   - Word 版：无给定模板，要求双栏、单倍行距。英文 Times New Roman，中文宋体。总标题三号，章节标题四号，正文五号。

2. **完整代码和模型**
   - 代码完整可复现，需注释说明思路。训练好的策略网络模型需保存一并提交。

3. **Demo 展示**
   - 完整录屏，展示如何启动算法、最终结果等。

## 评分说明

- 实验结果接近或超过给定基线、实验完整且分析合理、代码完整提交、符合学术诚信标准，即可获得大量基础分。
- 鼓励学术和创新性探索，**不注重模型绝对性能**。
- 实验报告占 **60%**，代码和 demo 展示占 **40%**。

---

## 课题一：啤酒厂供应链管理（Beer Game）

| 项目 | 信息 |
|------|------|
| 助教 | 王驰原 |
| 邮箱 | wang2021@stu.pku.edu.cn |
| 微信 | WCY_2610 |
| 所属 | 北京大学计算机学院 |

### 任务说明

**目标：** 通过调节订单数量，优化啤酒游戏中某个企业的利润。

啤酒游戏是一个串行供应链网络，包含多个角色：
- **零售商：** 接收客户订单，向批发商下订单
- **中间商：** 接收上游订单，向下游发货

供应链中的企业根据观察到的订单和需求等信息，决策向上游订购的订单量。研究对象可选链上任一企业（代码中定义 3 个企业），选择其中 1 个进行利润优化，其余视为环境的一部分。

### 实验环境

- 订购和发货存在延迟：t 时刻发出订单，t+1 时刻收到货物
- 参考代码：https://github.com/paperusername/coursebeergame.git
- **动作：** 设计合适的订购策略，替换对应企业动作
- **观察：** 包含 3 部分，选择研究的企业后加载对应的观察空间

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 基线算法：DQN
- 基于任意单智能体 RL 算法复现 baseline，优化自身利润
- 其余智能体可参考管理学供应链策略或使用随机策略

**二、提出创新（14 分）**
- 改进方案：调节环境变量、改变 demand 分布、将订单空间拓展至高维、增加供应商数量等
- 新算法：针对具体目标，提高算法在特定任务上的能力
- 改变 RL 决策智能体数量：研究多智能体订货行为的关联
- 分析改进原因，对比训练曲线，保存不同算法模型，比较最终策略效果（学习曲线、订购行为对比等）

### 参考文献

- [1] Mnih, V., et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533 (2015). (DQN)
- [2] Schulman, J., et al. "Proximal Policy Optimization Algorithms." (2017). (PPO)
- [3] Haarnoja, T., et al. "Soft Actor-Critic Algorithms and Applications." arXiv:1812.05905, 2018. (SAC)

---

## 课题二：2 人合作推箱子（Sokoban）

| 项目 | 信息 |
|------|------|
| 助教 | 王驰原 |
| 邮箱 | wang2021@stu.pku.edu.cn |
| 微信 | WCY_2610 |
| 所属 | 北京大学计算机学院 |

### 任务说明

**目标：** 控制 2 个智能体，合作将 2D 地图中的箱子推到指定位置。

**环境链接：** https://github.com/mpSchrader/gym-sokoban

### 实验环境

- **环境版本：** 共 6 种（v0–v5），通过 `env.make` 时指定 Room ID
- **Grid-Size：** 实际地图尺寸，用于逻辑判断
- **Pixels：** 观测地图像素大小，表示实际输入
- **#Boxes：** 需完成的箱子数量，数量越高越复杂

**注意：**
1. 每轮不强制要求某个 agent 行动
2. 可用单智能体算法求解
3. 奖励为双人共享
4. 多智能体算法需考虑信用分配（credit assignment）

### 基准算法

1. **HASAC**（Heterogeneous-Agent Soft Actor-Critic）— https://github.com/PKU-MARL/HARL
2. **HAPPO**（Heterogeneous-Agent Proximal Policy Optimisation）
   - 参考论文：https://arxiv.org/pdf/2109.11251 / https://arxiv.org/abs/2306.10715
3. 更多 MARL 算法：https://github.com/PKU-MARL/HARL

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 可使用 HARL 库将双人 Sokoban 环境部署到 HARL 框架内，完成多智能体算法训练与评估
- 也可使用其他 MARL 算法代码库或自行编写训练评估框架

**二、提出创新（14 分）**
- 改进方案：重构环境、增加箱子数量、修改游戏规则等
- 新算法：针对具体目标，提高算法在任务上的能力
- 分析改进原因，对比训练曲线，保存不同算法模型，比较最终策略效果（学习曲线、行为对比等）

---

## 课题三：最优税收设计（TaxAI）

| 项目 | 信息 |
|------|------|
| 助教 | 王驰原 |
| 邮箱 | wang2021@stu.pku.edu.cn |
| 微信 | WCY_2610 |
| 所属 | 北京大学计算机学院 |

### 任务说明

**目标：** 求解具有大量家户和政府的经济体均衡状态，寻找最优税制。

**链接：** GitHub: https://github.com/jidiai/TaxAI | ArXiv: https://arxiv.org/abs/2309.16307

### 实验环境

**家户智能体（同构）：**
- 观测空间：个人财富水平、收入水平 + 经济体平均资本、平均收入、平均劳动时间（低维）
- 动作空间：消费量、工作时长
- 奖励函数：CRRA 消费正效用 + Frisch 劳动负效用

**政府智能体（与家户异构）：**
- 观测空间：经济体平均资本、平均收入、平均劳动时间等公共信息（不含家户私有信息）
- 动作空间：税率
- 奖励函数：多目标可选（福利、平等、效率、产出）

**综合目标：** 同时求解家户和政府的最优策略。家户之间同构，政府与家户异构。

### 基准算法

- 原代码库提供 MAPPO、MADDPG 等算法
- HARL 库：https://github.com/PKU-MARL/HARL

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 可使用原代码库的 MAPPO、MADDPG 等算法直接实验
- 也可使用 HARL 库部署 TaxAI 环境，或使用其他 MARL 代码库 / 自行编写框架

**二、提出创新（14 分）**
- 改进方案：引入更丰富的冲击类型、增加新经济部门（财政、货币等）、增加政府观测维度（理论上政府需观测完整分布才能求最优税率）
- 新算法：重点关注 (1) 大规模 household 问题处理 (2) 异构智能体问题（政府-家户形成 Stackelberg 博弈）
- 分析改进原因，对比训练曲线，比较最终策略效果

---

## 课题四：胡闹厨房（Overcooked）

| 项目 | 信息 |
|------|------|
| 助教 | 王驰原 |
| 邮箱 | wang2021@stu.pku.edu.cn |
| 微信 | WCY_2610 |
| 所属 | 北京大学计算机学院 |

### 任务说明

**目标：** 控制两个智能体在限定时间内合作完成每个关卡的订单运送任务。

**环境链接：**
- 原始库：https://github.com/HumanCompatibleAI/overcooked_ai
- 新库（推荐）：https://github.com/Stanford-ILIAD/PantheonRL

### 实验环境

- **观测空间：** 游戏画面
- **动作空间：** 上下左右移动 + 停留 + 交互，共 6 种
- **奖励函数：** 完成订单有固定得分，完成其他任务有附加分

### 基准算法

1. **PPO** — https://arxiv.org/abs/1707.06347
2. **DQN** — https://arxiv.org/abs/1312.5602
3. **SAC** — https://arxiv.org/abs/1801.01290
4. 更多 MARL 算法：https://github.com/PKU-MARL/HARL

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 可使用原代码库提供的 PPO、SAC、A2C、DQN 等算法直接实验
- 也可使用 HARL 库部署 Overcooked 环境，或使用其他 MARL 代码库 / 自行编写框架

**二、提出创新（14 分）**
- 改进方案：不同 reward shaping 方式、对中间任务增加奖励、调整游戏参数增加难度等
- 新算法：尝试经典团队协作算法或多智能体算法，重点考察 (1) 多智能体信用分配问题 (2) 能否超越基于单智能体轮流优化的基线
- 分析改进原因，对比训练曲线，比较最终策略行为

---

## 课题五：多人仓储（RWARE）

| 项目 | 信息 |
|------|------|
| 助教 | 黄奕喆 |
| 邮箱 | szhyz@pku.edu.cn |
| 微信 | 15989316391 |
| 所属 | 北京大学智能学院 |

### 任务说明

**目标：** 控制多个智能体，合作完成仓储取物任务。

**环境链接：** https://github.com/semitable/robotic-warehouse

### 基准算法

1. **IAC**（Independent Actor-Critic）
2. **SNAC**（Shared Network Actor-Critic）
3. **SEAC**（Shared Experience Actor-Critic）
   - 论文：https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf
   - 代码：https://github.com/uoe-agents/seac/tree/master

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 使用已有仓库代码，实现 IAC、SNAC、SEAC 三个基线算法，完成多智能体算法训练与评估
- 也可自行编写训练评估框架
- 使用 `rware-tiny-2ag-v2` 和 `rware-small-4ag-v2` 进行测试

**二、提出创新（14 分）**
- 改进方案：修改训练方式、调整神经网络结构、增加新的损失函数等
- 新算法：针对具体目标，提高算法在任务上的能力
- 分析改进原因，对比训练曲线，保存不同算法模型，比较最终策略效果

---

## 课题六：2v2 乒乓赛（Quadrapong）

| 项目 | 信息 |
|------|------|
| 助教 | 黄奕喆 |
| 邮箱 | szhyz@pku.edu.cn |
| 微信 | 15989316391 |
| 所属 | 北京大学智能学院 |

### 任务说明

**目标：** 控制 4 个智能体（两两一队），进行桌面乒乓比赛。

**环境链接：** https://ale.farama.org/multi-agent-environments/quadrapong/

### 基准算法

1. **IPPO**（Independent PPO）— https://arxiv.org/abs/1707.06347
2. **MAPPO**（Multi-Agent PPO）— https://arxiv.org/abs/2103.01955
3. **QMIX**（Monotonic Value Function Factorisation）— https://arxiv.org/pdf/1803.11485
4. 更多 MARL 算法：https://github.com/PKU-MARL/HARL 及 https://cleanmarl-docs.readthedocs.io/en/latest/marl.html

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 实现 IPPO、MAPPO、QMIX 三个基线算法，完成多智能体算法训练与评估
- 也可自行编写训练评估框架
- 环境中有两个队伍，可将三种算法两两比较

**二、提出创新（14 分）**
- 改进方案：修改训练方式、调整神经网络结构、增加新的损失函数等
- 新算法：针对具体目标，提高算法在任务上的能力
- 分析改进原因，对比训练曲线，保存不同算法模型，比较最终策略效果

---

## 课题七：马尔可夫社会困境（Melting Pot）

| 项目 | 信息 |
|------|------|
| 助教 | 黄奕喆 |
| 邮箱 | szhyz@pku.edu.cn |
| 微信 | 15989316391 |
| 所属 | 北京大学智能学院 |

### 任务说明

**目标：** 控制三个智能体，提升群体合作水平，提高群体奖励。

**环境链接：** https://github.com/google-deepmind/meltingpot

**具体场景：** Allelopathic Harvest

> ⚠️ 本题观测空间与状态空间较大，训练时长可能较长，请谨慎选择。

### 基准算法

以下三种方法为对 reward 的不同处理方式，可套用在任意 RL 算法中（需保持三组实验所用 RL 算法一致）：

1. **Independent RL** — 独立 RL（PPO 或其他 RL 算法均可）
   - 参考：https://arxiv.org/abs/1707.06347
2. **Prosocial RL** — 在 Independent 基础上共享奖励
   - 参考：https://arxiv.org/pdf/1709.02865 / https://arxiv.org/pdf/2211.13746
3. **Inequity Aversion** — 对奖励的特殊处理方式
   - 参考：https://proceedings.neurips.cc/paper/2018/file/7fea637fd6d02b8f0adf6f7dc36aed93-Paper.pdf

### 具体内容（满分 35 分）

**一、算法设计（21 分）**
- 实现 Independent、Prosocial、Inequity Aversion 三种 reward 处理方法，完成训练与评估
- 也可自行编写训练评估框架
- 目标：提升群体合作水平、群体奖励水平

**二、提出创新（14 分）**
- 改进方案：修改训练方式、调整神经网络结构、增加新的损失函数、增加新的奖励处理方案等
- 新算法：针对具体目标，提高算法在任务上的能力
- 分析改进原因，对比训练曲线，保存不同算法模型，比较最终策略效果

---

## 课题快速对比

| # | 课题 | 助教 | 智能体数 | 类型 | 难度提示 |
|---|------|------|----------|------|----------|
| 1 | Beer Game 供应链 | 王驰原 | 1（可选多） | 单智能体 RL | 环境简单，适合入门 |
| 2 | 合作推箱子 | 王驰原 | 2 | 多智能体 RL | 信用分配是核心挑战 |
| 3 | TaxAI 最优税收 | 王驰原 | 多（异构） | 多智能体 RL | Stackelberg 博弈，经济背景 |
| 4 | Overcooked | 王驰原 | 2 | 多智能体 RL | 协作性强，环境成熟 |
| 5 | RWARE 仓储 | 黄奕喆 | 2–4 | 多智能体 RL | 三个基线，对比清晰 |
| 6 | Quadrapong 乒乓 | 黄奕喆 | 4（2v2） | 多智能体 RL | 对抗+合作混合 |
| 7 | Melting Pot 社会困境 | 黄奕喆 | 3 | 多智能体 RL | 训练时间长，社会困境建模 |
