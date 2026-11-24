# 动态目标跟踪与感知的无人机路径规划

[English](README.md) | **简体中文**

无人机收到的车辆位置延迟 d 步，要连续 5 步待在车 2 格以内完成采集，再飞到自己的终点。车辆走随机路线，目的地不告诉无人机。
现在先在 50×50 网格上把方法做出来，最终场景是路网。

## 方法

每一步，预测器根据延迟报告预测车的下一步位置，规则朝这个点走一步。Double DQN（van Hasselt et al., AAAI 2016）的输入里有规则动作和预测点；
不照规则走的动作扣 `b`（`--rule-prior`）。这是残差 Q 学习（Li et al., NeurIPS 2023）的自制简化：用硬最大值代替软最大值。

| `--prediction-model-type` | 做法 | 出处 |
|---|---|---|
| `linear`（默认） | 线性预测：匀速外推 | Schöller et al., RA-L 2020 |
| `direction_lstm` | 方向概率 LSTM：对下一步移动做分类，多步时逐步抽样 | Wu et al., IJCAI 2017；Graves 2013 |
| `goal_filter` | 目的地推断：对车的目的地做贝叶斯推断 | Baker et al. 2009；Ziebart et al. 2009 |
| `oracle` | 真实未来位置，只作诊断上限 | — |

`direction_lstm` 和 `goal_filter` 用 `--prediction-point disk` 取半径内概率最大点，即车落在采集半径内概率最大的格子；做法同 HOME（Gilles et al., ITSC 2021）的 MR 取点只取一个点时的形式。

## 运行

```powershell
python -m pip install -r requirements.txt
python train.py --car-route random --auto-return --reward-v2 --movement-penalty 0 --observation-delay 8 --prediction-model-type direction_lstm --rule-source predictor --rule-prior 0.1 --observation-features uav_pos uav_speed uav_goal car_pos time sensing_progress sensing_complete sensing_failed relative_car relative_uav_goal car_move_phase observation_age has_observation rule_action --output-dir outputs/example
python evaluate.py --checkpoint outputs/example/best_model.pt --episodes 200 --output-dir outputs/example_test
python demo.py --checkpoint outputs/example/best_model.pt --output-dir outputs/example_demo
python -m pytest -q
```

`experiments/`：`run_queue.sh` 按任务清单排队训练（结果写到 `outputs/diag_v9/`）；`analyze_groups.py` 做组间比较；
`offline_predictors.py` 离线检查预测器的命中率和校准；`rule_only.py` 只用规则、不学习。

## 已完成的实验

| 时间 | 完成人 | 设置 | 结果 |
|---|---|---|---|
| 10-22 | lhy | 50×50，车从固定角落出发，无延迟，1 个训练种子 | 100 条测试路线：DQN 与贪心追踪成功率都是 100%，平均 43.8 / 43.6 步 |
| 11-04 | lhy | 15×15，固定路线，延迟 3 步 | 100 条测试路线：DQN 成功率 95%，预测拦截规划器 59% |
| 11-09 至 11-11 | lhy | 实现 RNN / LSTM / GRU、TCN、VAR 预测器（均方误差训练） | 之后在随机路线上比较，都不如线性预测；方向 LSTM 由其 LSTM 和 MLP 方向分类器改来 |
| 11-05 至 11-24 | karnes | 随机路线，延迟 8 / 12 步，各 10 个种子，失败记 150 步 | 纯 DQN 100.8 / 122.4 步；规则 + 先验：线性预测 56.8 / 80.9，方向 LSTM 53.7 / 72.0，目的地推断 53.8 / 69.1 |

前三行的配置和复现命令见 [README_LOCAL.md](README_LOCAL.md)。

## 代码结构

| 文件 | 作用 |
|---|---|
| `uav_demo/environment.py`、`vehicle.py`、`observation.py` | 网格环境、车辆路线、延迟报告、特征 |
| `uav_demo/prediction.py` 和 `*_prediction.py`、`goal_inference.py` | 预测器接口和各预测器 |
| `uav_demo/point_selection.py` | 从分布取点 |
| `uav_demo/extrapolate_pursuit.py` | 规则：朝目标点贪心走一步 |
| `uav_demo/agent.py`、`model.py`、`config.py`、`train.py` | 带规则先验的 Double DQN、配置、训练 |
| `evaluate.py`、`demo.py`、`uav_demo/visualization.py`、`planner.py` | 评估和回放 |
| `deploy/` | 仿真平台接口 |

2026-11 清理时删掉的代码（MSE 预测器、预测拦截规划器、DRQN、`uav_delay/` 等）在提交 `5043846` 里。另见[部署说明](deploy/README.md)、[历史记录](README_LOCAL.md)、[原始规划](PROJECT_PLAN.md)。
