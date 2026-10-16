UAV Path Planning for Dynamic Target Tracking and Sensing under Different Observability Conditions  


## 1. 项目背景与核心场景（Project Overview）

### 1.1 场景描述

在二维 GridWorld 格子地图中，一架无人机（UAV Agent）从指定起点 S 出发，最终需要到达终点 G。地图中有一辆持续运动的小车（Vehicle），其移动方向可能包括直行、左转、右转与掉头，转向由概率模型决定。可选地，小车自身也具有目标位置，从而形成有目的性、但存在随机性的运动轨迹（Goal-directed Stochastic Mobility）。

UAV 必须在行进过程中接近小车，在规定距离内保持一段时间，完成数据采集（Sensing），随后抵达终点。目标是在满足采集与终点约束的前提下，尽可能减少任务时间、路径长度或能耗。小车运动可用固定随机种子复现，以保证不同算法在同一测试任务上的可比性，但非 Oracle Agent 不得访问未来轨迹。

**图 1：GridWorld 动态目标追踪与采集场景示意图。**（Word 版在此处附图；MD 版仅作纯文字描述）

图示说明：S 代表 UAV 起点，G 代表 UAV 终点；蓝色虚线表示 UAV 的示例规划路径，橙色虚线表示随机移动车辆的一次示例运动轨迹；UAV 周围的圆圈表示可能的有效采集范围，灰色格子表示可选障碍物。示意图并非实际算法计算结果。

### 1.2 核心研究问题

How can a UAV efficiently intercept and track a moving target to complete sensing tasks under different levels of information availability?

通过由“已知未来轨迹”逐步过渡到“只知道当前状态”与“历史/有限视野信息”，比较信息可得性对最优决策、强化学习策略和任务完成效率的影响。

### 1.3 三种信息条件与负责人

| 场景 | 信息条件 | 研究方法 | 负责人 |
|---|---|---|---|
| A. Oracle Planning | 车辆完整未来轨迹已知 | Time-expanded A*、DP、其他规划基线 | Member A |
| B. Fully Observable RL | 当前车辆位置与方向可见，未来轨迹未知 | PPO / DQN | Member B |
| C. Partially Observable RL | 只能获得延迟历史位置，或在有限视野内看到车辆 | Frame Stacking、Recurrent PPO、Belief Estimation | Member C |

### 1.4 两种采集机制

| 模式 | 采集规则 | 策略特点 |
|---|---|---|
| Consecutive Sensing（连续采集） | 必须连续 K 步处于有效采集范围；中断清零 | 强调持续稳定跟踪 |
| Cumulative Sensing（累计采集） | 累积有效采集达到 K 步；中断后保留进度 | 允许多次拦截、间歇接触 |

两种采集机制均用于 A、B、C 三个场景，形成 **3 种信息条件 × 2 种采集机制 = 6 组核心实验条件**。建议两种机制共用环境代码，通过配置参数切换。

## 2. 技术框架与基本建模

### 2.1 统一模块

- Shared GridWorld Environment：地图、起点终点、静态障碍物（可选）、UAV 运动规则、车辆运动模型。
- Observation Interface：根据场景提供 Oracle、Fully Observable 或 Partially Observable 的信息；真实环境状态与 Agent Observation 分离。
- Sensing Logic：有效距离、连续/累计采集计数、完成条件。
- Agent / Planner：Member A 的 Oracle 与传统规划方法、Member B 的 MDP-RL、Member C 的 POMDP-RL。
- Evaluation & Visualization：固定测试轨迹、统一指标、图形回放、性能汇总。

### 2.2 基本任务定义

- UAV 状态：位置 (x_u, y_u)，可选朝向、剩余时间与能量；动作初版为上、下、左、右、原地停留。
- Vehicle 状态：位置 (x_v, y_v)、当前方向及可能的目标位置；通过具有惯性和概率转向的转移规则更新。
- 有效采集：在每个时间步判断 UAV 与车辆的距离是否小于等于阈值 R，再更新相应的采集计数。
- 成功条件：采集达到 K 步并且 UAV 抵达 G；设置最大步数上限，超时视为失败。
- 优化目标：在尽量提高任务成功率的基础上最小化成功任务的完成时间，并报告路径长度/能耗。
- 重要限制：如果车辆运动遵循已知马尔可夫转移概率，随机运动本身并不意味着一定需要 RL；应与传统动态规划或 MPC 进行对照。

## 3. 三人具体分工（Task Allocation）

### 3.1 Member A — Oracle Planning & Simulation / Testing

**角色定位：** 负责公共仿真环境、Oracle Planning 与传统基线，并主导统一测试、可视化和实验复现。

1. **环境构建：** 实现二维地图、UAV 动作、车辆概率转向/掉头/可选目标导向运动；实现随机种子与地图、轨迹配置。
2. **采集机制：** 实现 Consecutive 和 Cumulative Sensing；统一距离阈值、采集时间、超时和成功判定。
3. **Oracle Planning：** 在车辆完整未来轨迹已知条件下，基于时间扩展图或动态规划计算完成采集并到达终点的最短可行任务时间；分别适配两种采集规则。
4. **对照基线：** 实现 Greedy Pursuit、A* Replanning；视工作量增加短时预测 MPC。注意 Oracle 是含额外未来信息的理想参照，Greedy/MPC 才更适合与 RL 进行同信息公平比较。
5. **测试与评价：** 编写统一测试脚本、固定车辆测试轨迹、路径回放、结果存储、成功率与任务时间等指标统计；组织不同地图和转向概率实验。

**交付成果：** Gymnasium-compatible Environment、Oracle/Greedy/A*（可选 MPC）、仿真可视化、标准化测试集与评估脚本。

### 3.2 Member B — Fully Observable Reinforcement Learning

**角色定位：** 研究当前车辆状态完全可见、但未来轨迹未知时的无人机拦截与持续采集策略。

1. **MDP 建模：** 设计 State、Action、Reward、Episode Termination；状态包含 UAV 位置、车辆当前的位置/方向、终点及采集进度。
2. **算法训练：** 实现 PPO 或 DQN 的初版，完成稳定训练；基于初版表现选择是否比较第二种算法。
3. **奖励设计：** 平衡追车、保持有效采集、完成采集与抵达终点，防止长期跟车却不抵达终点、或只冲向终点而忽略采集。
4. **策略分析：** 比较直接追逐和提前拦截等行为，测试不同运动随机性、UAV/Car 速度比、采集时长下的表现。
5. **统一实验：** 分别在 Consecutive / Cumulative 两种采集模式下测试，与 Oracle、Greedy、A* Replanning、可选 MPC 对比；测试未见过的随机种子与运动参数。

**交付成果：** Fully Observable PPO/DQN 模型、训练代码与日志、消融和泛化实验、策略分析图表。

### 3.3 Member C — Partially Observable Reinforcement Learning

**角色定位：** 研究在缺失实时车辆位置、只能依赖历史观测或有限视野时的决策、状态估计与主动感知。

1. **POMDP 建模：** 仅向 Agent 暴露合法的观测信息；明确最近一次观察位置、观察时间戳、目标是否可见等信息，避免泄漏真实车辆位置。
2. **阶段 C1 — Delayed Observation：** 实现固定位置延迟（如 2/4/8 步）与可选随机消息丢失，研究观测变旧带来的影响。
3. **阶段 C2 — History-dependent Policy：** 比较单帧观察、Frame Stacking 与 Recurrent PPO（GRU/LSTM）；可选显式 Belief State，评估历史记忆效果。
4. **阶段 C3 — Limited Field of View：** 车辆只有进入视野时才能被精确观察；UAV 需要根据历史信息预测、搜索和重新发现车辆，并研究移动所带来的信息价值。
5. **统一实验：** 与 Fully Observable RL 对比；改变观测延迟、丢失率、视野大小，测试两种采集规则及泛化表现。

**交付成果：** 部分可观测环境接口、History-based / Recurrent RL 模型、Limited-FOV 扩展、实验结果。

## 4. 实施阶段（Development Roadmap）

| 阶段 | 工作内容 | 责任划分 | 主要交付物 |
|---|---|---|---|
| Phase 1 — Environment Design | 统一车辆运动、地图、行动空间、状态/观测接口、采集规则和终止条件 | A 主导，B/C 共同确认 | 可运行且可复现的统一环境 |
| Phase 2 — Baseline & Initial RL | Oracle / Greedy / A*；Fully Observable PPO/DQN；初版固定延迟 POMDP | A/B/C 并行 | 三条技术路线均能运行 |
| Phase 3 — Algorithm Development | 完善规划基线；优化 RL 与奖励；历史信息与 Recurrent RL | 各自负责 | 稳定模型及初步对比结果 |
| Phase 4 — Partial Observation Extension | 实现视野约束、目标丢失和重新发现；建立相应对照 | C 主导，A/B 支持 | Limited-FOV / Active Perception 扩展 |
| Phase 5 — Experiments | 六组核心组合、统一种子、参数敏感性、消融、泛化实验 | A 统筹，全员参与 | 指标数据、图表和分析 |
| Phase 6 — Integration & Report | 系统演示、代码整合、报告撰写、答辩准备 | 全员 | 最终 FYP 系统与报告 |

## 5. 统一实验设计（Experimental Design）

### 5.1 主要比较

| 实验 | 研究目的 |
|---|---|
| Oracle vs Fully Observable RL | 估计未知未来车辆轨迹产生的决策效率损失 |
| Fully Observable vs Partial Observable RL | 衡量观测延迟或视野限制带来的性能下降 |
| RL vs Greedy / A* / MPC | 验证 RL 相对传统规划方法是否有性能优势 |
| Consecutive vs Cumulative | 研究采集连续性对拦截与跟踪策略的影响 |
| Without History vs With History | 测试历史输入或循环记忆网络的作用 |
| Different Delay / Dropout / FOV | 验证部分可观测策略的鲁棒性 |
| Seen vs Unseen Vehicle Dynamics | 验证对不同车辆转向概率及轨迹的泛化能力 |

### 5.2 统一评价指标

| Metric | 含义及统计要点 |
|---|---|
| Mission Success Rate | 同时完成采集与抵达终点的 Episode 占比 |
| Mission Completion Time | 成功 Episode 的完成时间；应连同成功率报告，避免选择偏差 |
| UAV Travel Distance / Energy | UAV 总路程或估计能耗 |
| Sensing Completion Rate | 达到采集阈值的 Episode 占比 |
| Contact Duration / Ratio | 有效采集接触时间及占比 |
| Target Recovery Time | 部分可见条件下目标丢失到再次发现的时间 |
| Oracle Optimality Gap | (T_agent − T_oracle) / T_oracle；仅对可行且完成的任务按约定统计 |

### 5.3 公平性与复现要求

- 三组方案共享相同的地图、车辆生成规则、UAV 动作和采集判定。
- 使用相同的一批测试种子与预生成车辆轨迹；训练/验证/测试种子互不重叠。
- Oracle 允许读取完整未来车辆轨迹，其他 Agent 不能获取；对比时清楚标注信息差异。
- 如果车辆转向受到 UAV 位置或动作影响，不能再直接使用同一条固定预生成车轨迹进行严格配对比较；第一版建议车辆运动与 UAV 独立。
- 明确 FOV、传感器观测和采集有效性是不同概念；避免凭借采集进度或奖励意外泄露隐含车辆位置。
- 按采集模式分别训练或适当条件化策略；每项比较需固定评估预算和评估条件。

## 6. 项目范围、优先级及风险控制

| 优先级 | 内容 |
|---|---|
| P0 — 必须完成 | GridWorld、随机移动车辆、双采集规则、终点任务及统一接口 |
| P0 — 必须完成 | Oracle Planning、Full Observation RL、固定延迟 Partial Observation RL |
| P0 — 必须完成 | 统一测试、成功率与任务时间评价、基础方案对比 |
| P1 — 推荐扩展 | Recurrent PPO / History-based Policy、Limited FOV、目标重发现 |
| P1 — 推荐扩展 | MPC、车辆模型泛化、观察延迟与采集时长敏感性 |
| P2 — 选做研究 | 显式 Belief Map、uncertainty-aware planning、真实视觉目标检测 |

主要风险：

1. **研究贡献过于算法化：** 仅比较 PPO 与 DQN 不足以构成清晰研究问题，需要围绕信息条件和感知约束组织实验。
2. **工作量失衡：** Member A 除 Oracle 外还承担环境、基线、评估和仿真；需让 B/C 参与接口设计、集成和测试。
3. **RL 必要性不足：** 对小型、转移已知的 GridWorld，经典规划可能表现很好，应客观报告，而不是预设 RL 更优。
4. **部分可见信息泄漏：** 观测设计应避免无意提供车辆真实状态，特别是奖励、距离反馈或采集进度。
5. **任务不可行：** 速度比、K、R、时间限制和地图大小必须通过预实验校准。

## 7. 研究问题与预期成果（Research Questions & Deliverables）

**RQ1 — Perfect Foresight：** 如果车辆未来轨迹完全已知，最优无人机任务规划是什么？Oracle 可提供怎样的性能参考？

**RQ2 — Stochastic Motion：** 如果车辆当前状态可见、未来不确定，RL 能否学习有效的拦截、稳定采集和终点导航策略？

**RQ3 — Partial Observability：** 如果车辆位置延迟、丢失或受视野限制，历史信息与主动观测能否改善任务完成效率？

**Cross-cutting Question：** 连续采集与累计采集如何改变不同观测条件下的最优路径和行为策略？

**最终交付：** 统一可复现仿真环境、Oracle/传统规划基线、全可见与部分可见 RL 算法、两类 Sensing 任务的系统比较、可视化演示、FYP 报告与答辩材料。

## 8. 开始开发前需要统一确认的参数

- 地图大小、障碍物设置、UAV 起点/终点分布。
- 车辆是否拥有终点，以及车辆转向概率是否为所有 Agent 所知。
- UAV 与车辆的移动速度比及运动规则。
- 采集半径 R、采集时长 K、连续/累计规则、是否允许停留采集。
- 是否允许采集但未看见车辆，以及 Agent 是否能实时获知采集进度。
- 部分可见的延迟机制、FOV 形状及大小、车辆重新进入视野的规则。
- 最大 Episode 步数、失败条件、训练预算、测试种子与评价指标。

---

**建议的最低可交付版本（MVP）：** 先完成统一 GridWorld + 车辆随机运动 + 双采集机制 + Oracle + Fully Observable PPO + 固定延迟 Partial Observable PPO。运行并比较六组核心任务后，再加入 Recurrent PPO、Limited FOV、Active Perception 和 MPC 等扩展。
