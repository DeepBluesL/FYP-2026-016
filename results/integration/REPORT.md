# 延迟车辆报告下的预测、追踪和提前会合

## 同一份 linear 预测，两种控制器

每个延迟使用相同的 32 条随机路线。任务成功要求连续采集 5 步并到达无人机终点。采集时刻只对实际采完的回合取平均，任务完成步数则对全部路线取平均，失败统一记 150 步。

| 延迟 | 控制器 | 任务成功 | 完成采集 | 采集时刻 | 任务完成步数 |
|---|---|---:|---:|---:|---:|
| 8 | 逐步追踪 | 20/32 | 21/32 | 71.90 | 109.31 |
| 8 | 提前会合 | 23/32 | 24/32 | 77.83 | 107.69 |
| 12 | 逐步追踪 | 7/32 | 10/32 | 100.80 | 138.91 |
| 12 | 提前会合 | 9/32 | 11/32 | 107.09 | 139.78 |

按路线计算“会合减追踪”，两个延迟的任务成功率差为 +9.38、+6.25 个百分点，平均完成步数差为 −1.63、+0.88。d=8 时会合有 6 条更快、16 条持平、10 条更慢。d=12 时分别是 5、23、4 条。对路线差值做 10,000 次有放回重采样（随机种子 42），平均步数差的 95% 百分位区间分别为 [−11.47, 7.59] 和 [−8.75, 11.25]。这批测试中会合多完成了几条任务，平均速度优势仍不确定。

每个决策时刻仅计算一次从当前到未来 34 步的 linear 预测。追踪器取下标 1，会合器接收同一数组并保留原有的会合点选择、等待、过期和接触维持逻辑。单路线冒烟测试先于批量运行。种子 3000000、d=8 的第 18–24 步，逐步追踪目标从 (14,19) 前移到 (14,25)，会合目标保持在 (14,38)。同一路线 d=12 第 83、84 步，会合器停在 (32,2)，距目标 (32,0) 两格。这次等待没有带来任务成功。见 [d=8 回放](controllers/replay_linear_8.json) 和 [d=12 回放](controllers/replay_linear_12.json)。

## 固定逐步追踪，改变预测器与选点

下表每个单元格为 d=8 / d=12。误差计算在所有路线、所有已收到报告且仍有下一帧的时刻，包含无人机已结束任务之后的车辆轨迹。模型只接收截至当前可见的报告，真值用于离线打分。命中表示预测点距下一步真实车位不超过 2 格。

| 方法 | 任务成功率 % | 任务完成步数 | 下一步平均位置误差（格） | 半径命中率 % |
|---|---:|---:|---:|---:|
| 直接追旧车位 | 18.8 / 0.0 | 137.56 / 150.00 | 6.36 / 8.61 | 2.8 / 1.3 |
| linear | 62.5 / 21.9 | 109.31 / 138.91 | 5.86 / 8.73 | 24.9 / 14.1 |
| VAR | 28.1 / 3.1 | 136.44 / 147.25 | 4.86 / 6.76 | 8.3 / 4.8 |
| TCN | 15.6 / 12.5 | 144.19 / 145.44 | 14.20 / 14.62 | 2.8 / 2.3 |
| direction_lstm，mode | 37.5 / 9.4 | 123.41 / 144.34 | 6.08 / 8.90 | 6.4 / 3.4 |
| direction_lstm，disk | 28.1 / 6.3 | 130.19 / 145.66 | 6.26 / 8.56 | 6.4 / 3.2 |
| goal_filter，mode | 75.0 / 56.3 | 101.88 / 117.00 | 5.10 / 6.82 | 27.9 / 19.0 |
| goal_filter，disk | 84.4 / 68.8 | 89.91 / 108.88 | 4.49 / 6.11 | 39.0 / 26.0 |

在这套随机路线生成器和训练预算下，逐步追踪首选 goal_filter + disk，相比 linear 平均少 19.41 / 30.03 步。该滤波器在拟合时使用生成器的训练路线目的地标签，测试时从延迟报告推断目的地。训练时可用的监督信息与其他预测器有差异。

同一个 direction_lstm 每次只抽一批 128 条样本，用同一份格点计数分别取概率最高的单格（mode）和半径 2 内总概率最高的中心（disk）。改用 disk 后平均多 6.78 / 1.31 步，任务成功率低 9.38 / 3.13 个百分点。goal_filter 的同类选点比较则有收益。选点效果与预测分布有关。VAR 的平均误差低于 linear，但半径命中率也低于 linear，仅凭平均误差选追踪预测器会得到不同结论。

## 强化学习与规则先验

同一无参数 linear 实例提供预测和下一步规则动作。基础特征按 main 默认顺序，末尾加 17 维 rule_action，合计 35 维。网络看 (1,) 时输入 37 维，看 (4,8,12) 时输入 41 维；规则始终追 t+1。三层宽度均为 64。PPO 使用 tanh，Double DQN 使用 leaky_relu。每组、每个延迟使用训练种子 42 和 10042，每次 4096 个环境交互步，按相同 8 条验证路线每 1024 步选模，再测试同一批 32 条路线。各训练种子均单独配对。下表为跨训练种子的任务成功率和平均完成步数。

| 策略 | 网络预测时距 | 先验 | d=8 成功率 / 步数 | d=12 成功率 / 步数 |
|---|---|---:|---:|---:|
| 纯规则 | 无网络 | 0 | 62.50% / 109.31 | 21.88% / 138.91 |
| PPO | 4、8、12 | 0 | 0.00% / 150.00 | 0.00% / 150.00 |
| PPO | 1 | 0 | 4.69% / 146.30 | 0.00% / 150.00 |
| Double DQN | 4、8、12 | 0 | 34.38% / 131.36 | 6.25% / 147.69 |
| Double DQN | 1 | 0 | 20.31% / 136.91 | 15.63% / 136.83 |
| Double DQN | 1 | 0.1 | 32.81% / 126.56 | 7.81% / 145.77 |

固定相同的 37 维 DQN 输入，先验 0.1 相对 0 在 d=8 少 10.34 步，在 d=12 多 8.94 步。32 条测试路线跨模型重复使用，64 个评估回合只有 32 条独立路线。两种算法的结果只描述这次短预算训练，长期收敛效果待运行。

## 诊断上限与历史参考

- **本次 Oracle 诊断**使用 t+1 的真实车位，同样等待首份报告。两个延迟均为 32/32 成功；平均采集时刻 20.72 / 26.06，任务完成步数 40.25 / 44.88。它是未来信息对照，未参与普通预测器排名。
- **历史记录**来自 spec 提交 5c3f957 的中文 README。原记录使用 10 个种子，d=8 / d=12 下纯 DQN 为 100.8 / 122.4 步，带先验的 linear、方向 LSTM、目的地推断依次为 56.8 / 80.9、53.7 / 72.0、53.8 / 69.1。本次使用自己的协议重新运行上表各组；历史数据仅供参考。
- **待运行**包括更长 RL 训练预算、更多预测器初始化种子和独立路线集的复验。现有 RNN/LSTM/GRU 坐标预测器保留在训练和纯规则入口，本轮性能矩阵按约定只运行表列预测器。

## 复现与验收

在整合工作区根目录使用安装了 requirements.txt 的 Python。本机使用 D:/anaconda3/envs/uav/python.exe，PowerShell 中先设 `$env:MKL_THREADING_LAYER='SEQUENTIAL'` 以避免本地 MKL/OpenMP 重复初始化。所有方法使用 50×50 网格、同一车辆随机路线生成器、每步移动、连续采集 5 步、半径 2、自动返航、势函数奖励、移动惩罚 0、上限 150 步。测试种子从 3000000 到 3000031。

```powershell
python -m unittest discover -s tests -q
python -m experiments.paired_rules --routes 1 --output outputs/integration_smoke
python -m experiments.paired_rules --predictors linear --controllers step meet --routes 32 --output results/integration/controllers
python -m experiments.paired_rules --predictors linear direct var tcn direction_lstm goal_filter --controllers step --points mode disk --routes 32 --output results/integration/predictors
python -m experiments.paired_rules --predictors oracle --controllers step --routes 32 --output results/integration/oracle
python -m experiments.paired_rl --steps 4096 --routes 32 --seeds 42 10042
python -m experiments.summarize_integration
foreach ($alg in 'ppo','dqn') {
  $model = "outputs/integration_rl/${alg}_d8_h1_p0_s42/best_model.pt"
  python evaluate.py --policy $alg --checkpoint $model --episodes 32 --output-dir "outputs/integration_accept/${alg}_eval"
  python demo.py --policy $alg --checkpoint $model --seed 3000000 --no-gif --output-dir "results/integration/replays/$alg"
}
```

纯规则没有策略训练种子。VAR 和 goal_filter 在固定 80 条训练路线上拟合一次；TCN 和方向 LSTM 各用同一组 80 条训练路线、10 条验证路线、1 个初始化种子（42）训练 8 轮。另有 10 条预测器内部留出路线。linear、直接追踪和 Oracle 无需拟合。主分支未跟踪脚本中有离线误差、按路线配对和固定预算训练的实现思路；它们原定的时距、终点指标和方法组合不同。本次调用仓库已有的预测、训练、评估函数，原脚本留在原工作区。

**接口验证**包含 147 个通过的单元测试。时间对齐用例明确检查当前 20、最后报告 12、规则目标 21，因此传给预测器的相对时距是 9；网络看未来 4/8/12 步时相对报告的时距是 12/16/20。训练、验证选模、独立评估、静态轨迹回放与部署加载均已运行。新训练模型和训练日志在 outputs/integration_rl 下。部署验收在本地网格观测接口执行。

**旧模型验证**在两个独立解释器中对同一组五个固定观测比较完整特征和模型输出。DQN 同时比较原始与含先验的 Q 值，PPO 比较 actor logits 和 critic 值。main PPO、main TCN-DQN、原版 spec 新训练的 linear-DQN，以及两边仓库均随附的 v7 d4/d6/d8，共六份权重，最大绝对差均为零。spec 工作区现存的是三个共享 v7 文件；v7 以当年发布代码 6480b83 为参照。spec 当前格式用原版代码训练两回合生成兼容性样本，该样本不计入效果对比。

```powershell
python -B ../FYP-2026-016-spec/train.py --episodes 2 --eval-every 1 --eval-episodes 2 --learning-starts 64 --prediction-model-type linear --observation-delay 8 --car-route random --rule-source predictor --rule-prior 0.1 --observation-features uav_pos uav_speed uav_goal car_pos time sensing_progress sensing_complete sensing_failed relative_car relative_uav_goal car_move_phase observation_age has_observation rule_action --output-dir outputs/spec_fixture
python -m experiments.check_compatibility --main ../FYP-2026-016 --spec ../FYP-2026-016-spec
```

[各组完整指标](summary.csv) · [配对差值汇总](paired_summary.csv) · [逐路线配对差值](paired_routes.csv) · [控制器原始结果](controllers/routes.csv) · [预测器与选点原始结果](predictors/routes.csv) · [RL 原始结果](rl/routes.csv) · [RL 输入、时距与选模记录](rl/protocol.json) · [Oracle 原始结果](oracle/routes.csv) · [权重兼容性](compatibility.json)
