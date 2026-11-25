# 动态目标跟踪与感知的无人机路径规划

[整合分支配对实验、结果和复现命令](results/integration/REPORT.md)

[English](README.md) | **简体中文**

## 当前实现与进展（2026-11-09）

本仓库已实现无障碍 GridWorld、Double DQN、车辆运动预测器、贪心追踪、外推追踪、预测拦截规划器与 Oracle 启发式对照。现已接入 RNN、LSTM、GRU 完整路线序列预测；原始规划及团队分工见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

### 环境与信息边界

- 默认地图 50×50，UAV 支持悬停及八方向、两档步长，共 17 个动作。支持连续和累计采集；完成采集并到达 UAV 终点才算成功。
- `reset()` / `step()` 返回原始单位的 dict。车辆终点和未来轨迹不进入普通策略观测；`info` 和回放文件中的真值仅供评估与可视化。
- 首次报告前，`has_observation=False`，`car_pos=None`，`car_history=()`。延迟为 3 时，t=3 才收到车辆 t=0 的位置。
- `--car-route fixed` 为默认固定起终点路线；`random` 使用随机起点、目的地，并在到达后更换目的地。隐藏字段不等于消除可学习的路线先验。

### 观测、预测与策略

| 位置 | 作用 |
|---|---|
| `uav_demo/environment.py` | 维护真实状态，生成延迟且不泄露终点的观测 |
| `uav_demo/observation.py` | 归一化、按字段顺序编码；支持可选 `rule_action` 特征 |
| `uav_demo/config.py`、`train.py` | 通过 `observation_features` / `--observation-features` 选择特征，按编码结果推导网络维数 |
| `uav_demo/prediction.py` | `forecast_vehicle()` 用 `age + h` 对齐预测时距，以报告时刻对齐车辆移动时钟 |
| `uav_demo/agent.py` | Double DQN、预测附加特征、经验回放及配置保存加载 |
| `uav_demo/predictive_planner.py` | 连续预测序列上的会合规划，记录当次实际预测 |
| `evaluate.py`、`demo.py` | 独立评估、首次报告指标、逐步观测与预测记录、PNG/GIF 回放 |
| `compare_delay.py` | 不同报告延迟下的策略比较 |

默认基础输入为 18 维（已移除车辆位移两维），三个预测点增加 6 维。`--prediction-steps 4 8 12` 表示相对当前时刻的预测偏移，而非相对历史报告时刻；没有报告时不调用预测器。经验回放仍保存数值数组，checkpoint 保存特征名称和顺序。

车辆观测提供 `car_history=((报告时间, (x, y)), …)`，只含已收到的报告，保留停顿帧；时间为测量时刻。MLP 和外推规则从历史报告推导最近运动，DQN 不直接接收位移。当前 checkpoint 版本为 9，旧 DQN 权重需要重新训练。

序列预测器支持 RNN/LSTM/GRU，默认采用单层 256 维循环网络和 `Linear(256,128) → ReLU → Linear(128,2)` 输出头。每批 32 条完整路线，默认学习率 0.001；`prediction_device=auto` 优先 CUDA，不可用时使用 CPU。DQN 的 `--device`、`--batch-size` 与预测器配置独立。

```powershell
python train.py --prediction-model-type lstm --prediction-learning-rate 0.001 --prediction-batch-size 32 --prediction-device auto --prediction-hidden-size 256 --prediction-head-size 128 --prediction-routes 1000 --prediction-seed 100000 --output-dir outputs/lstm
```

预测器总路线数不再受 DQN 回合数限制。默认种子 `100000..100999` 按 800/100/100 分成训练、验证、测试；`--prediction-split-seed` 控制分组。整个预测器种子区间与 DQN 训练/验证及最终系统测试分开。每个 epoch 按需生成路线，训练时打乱路线顺序、不切窗口、不丢尾批。`config.json` 保存配置，`prediction_metrics.json` 保存实际分组、训练设置和最佳验证一步位置 MSE；预测器测试组不用于选模型。

另外支持外推规则先验（`--rule-prior`，需选择 `rule_action` 特征）和采集结束后自动返航（`--auto-return`）。这些设置改变策略行为，比较实验时应显式记录；默认不开启。

### 安装与运行

在项目根目录执行（Python 环境需安装 PyTorch）：

```powershell
python -m pip install -r requirements.txt
python train.py --episodes 1500 --observation-delay 3 --output-dir outputs/train_delay3
python evaluate.py --checkpoint outputs/train_delay3/best_model.pt --policy dqn --episodes 100 --output-dir outputs/test_delay3
python demo.py --checkpoint outputs/train_delay3/best_model.pt --policy dqn --output-dir outputs/demo_delay3
```

评估和演示的 `--policy predictive` 可切换为预测拦截规划器，需要模型包含车辆预测器。`--no-motion-prediction` 关闭 DQN 的预测附加输入，`--no-gif` 可用于 demo 只生成静态图。模型、日志和动画保存在 `outputs/`，该目录不上传 GitHub。

### 已完成的实验

2026-11-04 的 15×15 固定路线实验：报告延迟 3 步，完整默认特征，DQN 训练 1,500 回合，预测器使用 300 条路线训练 20 个 epoch。验证集选出第 1,200 回合模型，在 100 条独立测试路线上，DQN 成功率 **95%**，预测规划器 **59%**。

这些结果对应当时的固定路线、无规则先验实验，不代表后续随机路线或其他延迟配置的表现。最后一轮验证成功率降至 43.3%，训练仍有波动，尚需多种子与消融验证。具体配置与复现命令见 [README_LOCAL.md](README_LOCAL.md)；原始结果位于本地 `outputs/dict_train_1500_20261104/`。


## 预测器独立配置

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--prediction-model-type` | lstm | rnn / lstm / gru |
| `--prediction-routes` | 1000 | 三组路线总数 |
| `--prediction-seed` | 100000 | 路线种子起点 |
| `--prediction-split-seed` | 42 | 分组随机种子 |
| `--prediction-learning-rate` | 0.001 | Adam 学习率 |
| `--prediction-batch-size` | 32 | 每批完整路线数 |
| `--prediction-device` | auto | 优先 CUDA，无 CUDA 则 CPU；支持显式 cpu/cuda |
| `--prediction-hidden-size` | 256 | 循环层宽度 |
| `--prediction-head-size` | 128 | 输出头隐藏宽度 |
| `--prediction-num-layers` | 1 | 循环层数 |
| `--prediction-epochs` | 20 | 预测器训练轮数 |
| `--prediction-steps` | 4 8 12 | 相对决策时刻的预测偏移 |

DQN 的 `--device` 默认仍为 CPU。每批保留完整路线，尾批损失按路线数加权。单向循环网络在每个时刻预测下一步位置，递归推理期间使用预测位置推进。训练入口按验证一步 MSE 选模型，不自动评估预测器测试组；延迟多步误差及下游任务表现需另行评估。

代码入口：`prediction_data.py` 负责完整路线生成，`sequence_prediction.py` 负责网络、batch、训练和递归预测，`prediction.py` 提供统一预测适配，`agent.py` 按模型类型加载权重，`evaluate.py` 检查最终系统测试种子边界。

## 验证与兼容性

当前 checkpoint 为 v9；旧权重需要重新训练，部署目录中的历史模型及参考结果不代表当前序列模型性能。尚未进行新版长训练或三种网络的性能比较。本机为 CPU 版 PyTorch，CUDA 实际运行尚未验证。

需要时可执行 `python -m unittest discover -s tests -q`。本次文档修改未运行测试。上述历史实验结果早于当前序列模型，不能作为新版结果。另见 [部署说明](deploy/README.md)。
