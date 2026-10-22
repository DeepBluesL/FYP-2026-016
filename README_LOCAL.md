UAV Path Planning for Dynamic Target Tracking and Sensing under Different Observability Conditions

## 固定右下角起点：DQN 与贪心对比

车辆当前从 `(grid_size - 1, 0)` 出发，50×50 时为 `(49, 0)`。`uav_demo/vehicle.py` 中原随机起点代码已注释保留；车辆后续路线仍随机。旧 checkpoint 的 `car_spawn_region` 字段仅用于兼容，不再决定当前起点。由于起点抽样不再消耗随机数，相同种子生成的路线也会与旧版本不同。

在仓库根目录、包含 PyTorch 的 `uav` 环境中运行：

```powershell
python compare_pursuit.py --training-context random_start --episodes 100 --seed 4000000 --output-dir outputs/fixed_corner_comparison
```

脚本复用现有 `wait_train/best_model.pt`，不重新训练。两种策略采用 checkpoint 中相同的环境参数和逐条相同的车辆路线；DQN 保留学得的运动预测特征，贪心只追当前车位。输出逐回合 `episodes.csv`、`summary.json`，以及第一条测试路线两种策略的 JSON、PNG 和 GIF。

本次种子 4000000–4000099、连续感知 5 步、半径 2 的测试：两者成功率均为 100%；DQN 平均完成时间 46.95 步、平均路程 94.42 格，贪心分别为 43.56 步、89.89 格。上述均值取成功回合，本次 100 条路线均为双方成功。这是旧随机起点模型迁移到固定起点的测试，不代表在固定起点重新训练后的 RL 性能。

固定起点重新训练与配对测试：

```powershell
python train.py --episodes 1500 --eval-episodes 20 --log-every 100 --output-dir outputs/fixed_corner_train
python compare_pursuit.py --checkpoint outputs/fixed_corner_train/best_model.pt --training-context fixed_corner --episodes 100 --seed 4000000 --output-dir outputs/fixed_corner_retrained_comparison
```

本次从零重新拟合车辆预测器并训练 DQN，训练种子 42–1541、验证种子 1000000–1000019；根据验证集选择第 900 回合 checkpoint。测试仍使用 4000000–4000099，不用于模型选择。重训 DQN 与贪心成功率均为 100%；平均完成时间分别为 43.78 / 43.56 步，平均路程为 92.31 / 89.89 格。DQN 比旧模型更快，但在本次 100 条路线中仍未超过贪心的时间与路程表现。这是一次训练种子的结果，不代表多次训练的统计结论。实验起点与种子说明保存在 `outputs/fixed_corner_train/experiment.json`。

## 最小 Demo：50 × 50，1 UAV，1 Car，当前状态与学习预测

当前学习任务是：**UAV 从左下角出发，利用车辆当前状态和学习得到的运动预测，先尝试拦截并完成感知，再飞往右上角终点**。DQN 不知道本回合车辆的真实未来路线。场景暂不包含障碍物或视野限制；Oracle 作为单独的、拥有完整未来信息的启发式参照保留。

目前提供三个明确分开的控制器：`dqn` 用学习到的 Q 网络选择动作；`predictive` 学习车辆运动模型，再用规则规划器选择会合点、飞往会合点等待及跟踪；`oracle` 直接读取真实未来路线。**predictive 的运动预测来自学习，动作来自规划规则，不是 DQN 学会了等待。** DQN 的训练与结果始终单独保留。

### 快速运行与文件结构

需要 Python 3.10+。本机可先运行 `conda activate uav`。命令在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt

# 训练车辆预测器，再训练 DQN；1500 回合不保证收敛
python train.py --episodes 1500 --output-dir outputs/current_prediction_20261022/wait_train

# 用独立测试 seed 和验证集选择的模型运行学习策略
python demo.py --policy dqn --checkpoint outputs/current_prediction_20261022/wait_train/best_model.pt --seed 3000000 --output-dir outputs/current_prediction_20261022/wait_demo

# 同一学习运动模型 + 预测拦截规划；不使用 DQN 的 Q 值选择动作
python demo.py --policy predictive --checkpoint outputs/current_prediction_20261022/wait_train/best_model.pt --seed 3000000 --output-dir outputs/current_prediction_20261022/predictive_demo

# Oracle 拥有额外的真实未来信息，属于不同信息条件的参照
python demo.py --policy oracle --seed 3000000 --output-dir outputs/current_prediction_20261022/wait_oracle
```

`demo.py` 默认使用 `dqn`，输出 `episode.json`、`trajectory.png` 和 `episode.gif`；`--no-gif` 可跳过动画导出，`--show` 可打开交互动画。已有数据可通过 `python -m uav_demo.visualization --episode outputs/current_prediction_20261022/wait_demo/episode.json` 重绘。绘图使用独立进程，避免部分 Windows Conda 环境的 OpenMP 动态库冲突。

```text
uav_demo/
    config.py          # EnvConfig 场景参数、TrainConfig 训练参数
    vehicle.py         # 车辆出生、固定终点、惯性运动与模拟轨迹
    environment.py     # 当前状态观测、动作、感知、奖励与终止
    prediction.py      # 学习车辆移动方向概率，递归预测未来位置均值
    planner.py         # 读取真实未来的 Oracle 拦截启发式
    predictive_planner.py # 仅用当前状态与学习预测做会合等待/跟踪规划
    model.py           # 可配置 MLP Q 网络
    agent.py           # Double DQN、预测特征、回放、模型保存与加载
    visualization.py   # 策略来源标签、历史轨迹、预测线与 GIF
train.py               # 预测器拟合、主训练循环、验证与模型选择
demo.py                # 独立运行 DQN、predictive 或 Oracle，保存回放
evaluate.py            # 独立种子测试、感知优先与严格等车指标
tests/                 # 环境、预测、规划器和模型行为验证
requirements.txt       # 运行依赖
outputs/               # 模型、日志与图像，已被 Git 忽略
```

### 场景规则与速度设置

| 参数 | 默认值或行为 |
|---|---|
| 地图 | 50 × 50，无障碍；坐标 0–49，x 向右、y 向上 |
| UAV 起点 / 终点 | `(0, 0)` / `(49, 49)` |
| UAV 移动 | 八方向，每步选择网格步长 1 或 2；另有悬停 |
| Car 起点 | 固定右下角 `(49, 0)`（一般为 `(grid_size - 1, 0)`）；原随机起点代码已注释，`car_spawn_region` 仅保留兼容旧模型 |
| Car 终点 | 左上角 `(0, 49)`；按地图尺寸取 `(0, grid_size - 1)` |
| Car 速度 | `car_move_every=1`，每个环境步移动 1 格；到站后停留 |
| Car 方向惯性 | 基础权重：直行 0.65、左/右转各 0.15、掉头 0.05 |
| Car 终点偏置 | 权重再乘 `exp(car_goal_bias × 靠近终点格数)`，默认 bias=1.0 |
| 感知 | 欧氏距离 ≤ 2 格，连续 5 个环境步；可选累计模式 |
| 总步数 | `max_steps=300`，含感知与前往 UAV 终点两个阶段 |
| 完整任务成功 | 感知完成，并且 UAV 抵达自己的终点 |

车辆在 `reset(seed=...)` 时预生成模拟路线，与 UAV 动作无关；相同配置和 seed 可复现。该路线是环境内部真实状态，**不会拼接进 DQN 的观测或预测器推理输入**。方向概率经过目标偏置、边界和剩余时间约束后重新归一化，实际直行概率并不恒为 0.65。临近时限时只保留还能按时到站的移动。

Car 速度在 `EnvConfig.car_move_every` 调整：1 为每步一格，2 为两步一格。UAV 的可选步长在 `environment.py` 的 `ACTIONS` 定义；`initial_speed` 只设置初始值，不固定整回合速度。轴向实际路程是步长，斜向为步长 × √2。悬停位移为零并保留原步长。改变 UAV 的可选步长需同时处理规划器、归一化、测试与模型兼容性。

17 个动作依次为：悬停；上/右/下/左的步长 1；右上/右下/左上/左下的步长 1；同样八方向的步长 2。车辆保持四方向移动。越界请求取消整段位移，策略通过动作 mask 屏蔽越界动作；没有隐式坐标裁剪。每步同时更新双方位置后，只按步末位置判定一次感知，速度 2 不会在中间格额外采集。

### 感知优先与失败后返航

感知阶段的奖励目标是车辆；只有成功采集，或车辆到站导致采集失败，才转入前往 UAV 终点的阶段。到站当步仍有最后一次感知机会；刚好完成则不处罚。如果仍未完成，立即一次性扣 `sensing_failure_penalty=-100`，设置 `sensing_failed=True`，永久停止本回合感知。

保留“失败后继续飞往 UAV 终点”的规则，但失败后距离塑形只保留原来的 `failed_return_progress_scale=0.1`。失败后不再获得接触/采集完成奖励，也不获得 +100 任务成功奖励。`sensing_finished = sensing_complete or sensing_failed` 表示阶段结束，不等于感知成功。

采集阶段结束后抵达 UAV 终点，回合 `terminated=True`；完整任务成功仍须 `sensing_complete=True`。提前飞到 UAV 终点而感知阶段尚未结束，不算成功也不会提前结束；如果此后车辆到站，则按失败处理。300 步仍是整回合上限，不因失败重新计时。

这是任务阶段、奖励和成功判定的约束；环境不会替换 DQN 的动作，也没有把“必须朝车飞”硬编码成动作规则。随机探索或尚未学好的策略仍可能不去感知，需通过独立验证检查。Oracle 与 DQN 的决策路径独立，DQN 不会偷偷调用 Oracle 帮它飞行。

### 当前状态、车辆预测与网络

环境观测是 **20 维**：UAV 位置/步长/终点、车辆当前位置/终点、归一化时间、感知进度、成功/失败标志、车辆相对 UAV 的位置、UAV 终点相对位置、最近观察到的车辆非零位移，以及车辆移动时钟相位。车辆位移从已发生的连续观测计算；初始为零，非移动帧保持最近一次可见位移，不读取车辆内部随机朝向。

`prediction.py` 的车辆预测器为 **8 → 32 → 32 → 4** MLP，两个隐藏层使用 ReLU。8 维输入由当前位置、最近可见位移、车辆终点及其相对位置组成，输出北/东/南/西四方向 logits；合法方向的概率由 softmax 得到。训练标签是训练轨迹中实际发生的下一次移动方向，使用交叉熵监督训练。

默认 `prediction_routes=300`、`prediction_epochs=20`，路线数量为 `max(2, min(prediction_routes, episodes))`：即使只跑 1 回合 DQN，也生成至少 2 条训练用途路线，保证拟合和内部留出集合均非空。种子从训练 seed 开始，按整条路线划分前约 80% 拟合、后约 20% 内部验证，防止相邻帧跨集合泄漏；该完整种子范围同样纳入训练/测试隔离检查。DQN 验证和最终测试路线均不参与预测器训练。预测器随后固定，供 DQN 或 predictive 控制器使用。

推理时递归传播四方向概率的期望位移，输出未来 **+4、+8、+12 环境步**的位置均值，并转成相对 UAV 的 6 个坐标特征。它是近似均值预测，可能不是整格路径，也不是必然发生的真实路线或精确联合分布。加上当前状态后，Q 网络输入是 **26 维**；`--no-motion-prediction` 关闭预测器，输入恢复为 20 维，适合做消融对照。

Q 网络默认 **26 → 128 → 128 → 128 → 17**：3 个隐藏层，LeakyReLU，负半轴斜率 0.01；Q 输出层线性。`--hidden-sizes` 的数字个数决定层数，各数字决定宽度；激活函数可用 `relu`、`leaky_relu`、`tanh` 或 `silu`。使用 Double DQN：在线网络选下一动作，目标网络评价其 Q 值，辅以经验回放、Huber loss、梯度裁剪和 epsilon-greedy。算法参考 [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)。默认学习率 0.0005、gamma=0.99、batch size=64。

**有预测器不等于已经学会提前等车。** 是否形成“先到可能路径等候，再保持感知”的行为，需要查看未参与训练的 rollout：到达拦截区域的时间、车辆通过时间、等待/跟踪动作、感知连续性与最终成功率。不能只凭预测线或单张静态图宣称学会了该策略。

为了给悬停等车保留合理回报，默认 `progress_to_contact_boundary=True`：感知阶段的距离采用 `max(欧氏距离 - sensing_radius, 0)`，进入有效范围后不再因更贴近车中心而额外获得距离奖励。`movement_penalty=-0.1` 按实际飞行距离收费，悬停不产生飞行距离代价，但仍有每步时间代价。这样策略可权衡等候、省路程和更早完成任务；不会强制它每回合等待，也不保证一定学到等车。前往 UAV 终点阶段仍使用到终点中心的距离。

`PredictivePlanner` 是单独的模型辅助控制器：只接收当前观测，调用 checkpoint 内的已学习车辆运动模型形成短期预测，用规划规则选取可到达的会合位置，保持会合点等待车辆，在预测偏离时重新规划，接触后保持感知，阶段结束后飞往 UAV 终点。这是带滚动预测的拦截规划，其等待动作来自明确的控制规则。它不使用环境的真实未来路线，也不使用 Q 网络输出。`--policy predictive` 需要含运动预测器的模型，不能使用 `--no-motion-prediction` 训练出的 checkpoint。

### 训练、模型选择和兼容性

```powershell
# 冒烟训练，只检查流程，不代表收敛；预测器至少需要两条训练路线
python train.py --episodes 3 --prediction-epochs 2 --learning-starts 32 --batch-size 16 --eval-episodes 2 --output-dir outputs/current_prediction_20261022/smoke

# 禁用预测器的同信息对照
python train.py --episodes 1500 --no-motion-prediction --output-dir outputs/current_prediction_20261022/no_prediction

# 修改预测预算、探索衰减和 Q 网络结构
python train.py --prediction-routes 300 --prediction-epochs 20 --epsilon-decay-steps 60000 --hidden-sizes 256 128 64 --activation leaky_relu --output-dir outputs/current_prediction_20261022/custom

# 累计感知需单独训练
python train.py --sensing-mode cumulative --output-dir outputs/current_prediction_20261022/cumulative
```

训练输出 `training.csv`、`eval.csv`、`config.json`、`prediction_metrics.json`（启用预测时）、`model.pt`、`best_model.pt` 和 `best_validation.json`。`model.pt` 是最近一次验证时的模型；`best_model.pt` 按验证的“更高任务成功率 > 更高感知完成率 > 更短成功回合平均步数 > 更短平均路程”依次比较选择，不根据最终测试集挑选。平均奖励不用于模型选优，避免反复接触得到额外奖励的慢策略被误选。

默认 DQN 训练种子从 42 递增，验证固定从 1,000,000 开始。本轮新的独立测试建议从 **3,000,000** 开始；2,000,000 起的第一批路线已用于行为诊断和奖励调整，不应再作为调整后模型的最终独立测试集。日志分别记录任务成功、感知失败、UAV 到达，避免把失败后抵达终点算作成功。训练和验证使用相同的观测构造与预测器推理。

新模型采用 **checkpoint v4**，将预测器权重、Q 网络权重、环境与训练配置一同保存；加载后预测特征和网络结构可一致恢复。旧 v1–v3 模型的观测和任务设置不兼容，必须重新训练。较早的 v4 若缺少 `progress_to_contact_boundary` 和 `movement_penalty`，加载时分别恢复为 `False`、`0.0`，保留当时的奖励语义；新训练默认启用边界距离和飞行代价。模型用于推理，不保存完整经验池或随机状态，不提供断点续训语义。修改车辆速度/出生分布/地图/感知规则后应重新训练。

### 奖励参数的来源与限制

以下是针对 Demo 手工调整的初值，没有从论文直接复制，也没有经过参数搜索或收敛保证。提高感知事件奖励、加重失败惩罚，并弱化失败返航奖励，是为了降低“跳过感知等待失败”的吸引力；数值层次本身不能证明最优策略或确保训练一定成功。

| 配置项 | 当前值 | 作用 |
|---|---|---|
| `step_penalty` | -0.1 | 每步时间代价；300 步未折扣累计 -30 |
| `movement_penalty` | -0.1 | 每格实际飞行距离的代价；斜向距离含 √2，悬停为零 |
| `progress_to_contact_boundary` | `True` | 感知阶段使用到有效范围边界的距离，不要求贴近车中心 |
| `progress_reward` | 0.2 | 当前任务目标的前后距离差系数 |
| `contact_reward` | +2 | 未完成感知时每次有效接触奖励，连续 5 步合计 +10 |
| `sensing_complete_reward` | +50 | 一次性感知完成奖励 |
| `sensing_failure_penalty` | -100 | 车辆到站仍未完成感知时扣一次 |
| `failed_return_progress_scale` | 0.1 | 失败后距离项进一步缩放，实际距离系数为 0.02 |
| `success_reward` | +100 | 感知成功并抵达 UAV 终点 |
| `invalid_move_penalty` | -1 | 越界请求；当前策略通常会通过 mask 避开 |
| `timeout_penalty` | -10 | 回合达到总步数上限仍未结束 |

距离塑形为 `progress_reward × scale × (d_before - d_after)`。默认感知阶段采用到车辆感知范围的边界距离 `max(欧氏距离 - R, 0)`，阶段结束后采用到 UAV 终点的中心距离；切换当步仍按之前阶段计算，下一步才切换目标。另外累加 `movement_penalty × 本步实际飞行距离`。车辆自身运动也会影响距离差。连续感知中断后再次接触仍可获奖，累计接触奖励可能超过 +10，需留意反复接触却不完成任务的行为。

理论参考是 Ng、Harada、Russell（1999）的 [Policy invariance under reward transformations: Theory and application to reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。其潜势式塑形为 `F(s,s') = gamma * Phi(s') - Phi(s)`。当前距离项没有该折扣形式，并包含分阶段事件奖励，因此**不是该定理的严格实现，不宣称策略不变性保证**。应使用固定交互预算、多训练种子及消融实验比较成功率与耗时，不能只看奖励曲线上升。

### 可视化、Oracle 参照与验证

图上会显式区分学习动作的 DQN、学习运动模型加规则规划的 predictive、拥有真实未来的 Oracle，以及非学习诊断脚本；来源缺失时为 `Unknown policy`。DQN 和 predictive 只显示已发生的车辆轨迹，紫色虚线 `Learned forecast` 是当前学习预测，缺失预测时不绘制该线。只有 Oracle 显示浅橙色完整真实未来路线。JSON 为复现保存的环境真实路线不等于模型能够读取它。

Oracle 寻找最早可赶到的真实未来位置，再跟车采集并前往终点，是启发式而非全局最优求解器。其八向最短移动步数处理步长奇偶差异，但这不能证明整个感知任务最短。Oracle 比当前状态 DQN 多了未来信息，适合做理想信息条件参照，不能解释成同信息条件下的公平算法优劣比较。

```powershell
python -m unittest discover -s tests -v

# 新独立测试批次：只用模型贪心动作，不用 Oracle 接管
python evaluate.py --policy dqn --checkpoint outputs/current_prediction_20261022/wait_train/best_model.pt --episodes 100 --seed 3000000 --output-dir outputs/current_prediction_20261022/wait_test

# 配对比较：相同场景、种子与学习运动模型，动作改由预测拦截规划器选择
python evaluate.py --policy predictive --checkpoint outputs/current_prediction_20261022/wait_train/best_model.pt --episodes 100 --seed 3000000 --output-dir outputs/current_prediction_20261022/predictive_test
```

验证包括运动时钟、地图边界、感知及失败返航语义、当前观测不泄漏真实未来、预测器输入与保存恢复、Q 网络/模型兼容性，以及规划器步数与 BFS 对照。训练时还应结合独立测试的成功率、感知完成率和路径回放评估泛化；短训练或单一成功示例都不构成收敛证明。

`evaluate.py` 保存逐回合 `episodes.csv` 和汇总 `summary.json`，检查测试种子与训练、预测器拟合/内部验证、DQN 验证种子区间不重叠。它无法自动知道某批路线是否曾被人工用来调参，需像本轮这样另留新的测试区间。

DQN 与 predictive 采用同一 checkpoint 内的车辆模型、相同环境设置及相同测试 seed，可逐条路线配对比较成功率、耗时、路程与等待行为；二者的动作控制机制不同，应分别报告。predictive 的控制规则仅在验证路线调整，最终配对测试路线不得用于挑选规则或参数。Oracle 拥有额外信息，另作参照。

严格等车指标 `waiting_contact` 定义为：**首次进入感知范围的当步选择 hover，且此前至少连续两步 hover、尚未接触车辆**。因此需要先稳定停住，随后车辆主动进入范围，单步偶然悬停或接触后才悬停不计入该指标。`waiting_contact_rate` 是所有测试回合的占比，`successful_waiting_contact_count` 进一步只统计最终任务成功的回合；`early_goal_before_sensing_rate` 检查感知阶段结束前是否曾先到 UAV 终点。严格等车指标描述观察到的动作序列，仍不能单独证明策略准确预测了未来；需结合学习预测与实际后续轨迹、任务成功和无预测对照分析。测试决策始终只读取当前状态和模型预测，不读取真实未来路线。

---

以下保留原始项目整体方案。


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
