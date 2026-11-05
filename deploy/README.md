# 仿真平台接入说明

训练好的策略：DQN + 匀速预测规则先验（b = 0.1）+ 简化奖励，随机车辆路线，采完自动返程。
每个延迟一个模型：`models/d4.pt`、`models/d6.pt`、`models/d8.pt`（各约 330 KB）。

## 需要的文件和环境

- 整个 `deploy/` 和 `uav_demo/` 文件夹（`policy.py` 会调用 `uav_demo` 里的特征计算、匀速预测规则和网络）。
- Python 3.10+，`numpy`、`torch`（CPU 即可，单步推理不到 1 ms）。

## 每一步怎么调用

```python
from deploy.policy import Policy, ObservationBuilder, ACTION_NAMES

policy = Policy(delay=6)                      # 选和仿真延迟一致的模型
builder = ObservationBuilder(policy.config)   # 每个回合新建一个

for t in range(150):
    # 1. 有车辆报告到达时登记：report_time 是“这个位置对应的时刻”，不是到达时刻
    builder.report(report_time, car_cell)
    # 2. 组装观测并决策
    obs = builder.build(t, uav_cell, uav_speed, sensing_progress, sensing_complete)
    action = policy.act(obs)                  # 0..16
    next_cell, next_speed = policy.next_state(obs, action)
    # 3. 仿真端让无人机在 1 s 内飞到 next_cell，再更新采集进度
```

`check.py` 是完整的可运行示例：把我们的训练环境当成仿真平台，只通过上面的接口喂数据，
并逐步核对动作和原评估代码完全一致（`python deploy/check.py --delay 6`，输出不一致步数为 0）。

## 坐标、时间和动作

| 项 | 约定 |
| --- | --- |
| 地图 | 50 × 50 格，坐标 `(x, y)` 为整数 0–49；1 格 = 5 m，+x 向东，+y 向北 |
| 时间 | 1 步 = 1 s；一回合最多 150 步 |
| 无人机 | 起点 `(0, 0)`，终点 `(49, 49)`（固定） |
| 动作 | 17 个：0 悬停；1–8 八个方向各走 1 格（5 m/s）；9–16 八个方向各走 2 格（10 m/s）。顺序见 `ACTION_NAMES` |
| 速度档 `uav_speed` | 上一步动作的档位（1 或 2），开始为 1；悬停时保持原档位不变 |
| 出界 | 掩码已屏蔽；`next_state` 对出界动作返回原位置 |
| 采集 | 无人机与车的欧氏距离 ≤ 2 格（10 m）记一次接触；连续 5 步完成，中途离开清零。由仿真端计算 `sensing_progress`（0–5）和 `sensing_complete` |
| 返程 | 采完后掩码只保留最短路径返程动作，`act` 会自动返回它 |

## 车辆报告

- 第 t 步收到的是车在第 t − d 步的位置（整数格坐标），d 必须等于模型的延迟（`policy.delay`）。
  第 d 步之前没有报告（`has_observation` 为 False），策略照常给出动作：匀速预测规则在这时悬停，
  学出来的策略通常会先往地图中部飞。
- 车速不用单独给：`ObservationBuilder` 用相邻两条报告之差算“最近一次非零位移”，和训练时一致。
- 只能给延迟报告，**不能把车的真实位置传给策略**，否则就不是延迟场景了。

## 使用范围（超出要重新训练）

模型只在训练时的设定下有效：地图 50 × 50、1 格 = 5 m、1 步 = 1 s、车每步走 1 格（5 m/s）、
无人机 5 / 10 m/s、采集半径 10 m 连续 5 步、固定延迟 4 / 6 / 8 步。
仿真是连续坐标时，把位置四舍五入到格子再传入，把 `next_cell` 当成 1 s 后要到达的航点。
延迟不是 4 / 6 / 8 时，选最接近的模型会有一定偏差；需要准确结果请按新延迟重新训练
（命令见 `README_LOCAL.md`）。

## 测试路线上的表现（`check.py`，200 条路线）

| 模型 | 成功率 | 平均完成时间（失败记 150 步） |
| --- | --- | --- |
| d4.pt | 100% | 39.7 步 |
| d6.pt | 99.0% | 46.5 步 |
| d8.pt | 96.5% | 53.4 步 |

每个延迟的三个训练种子中，取选模型路线（2000000–2000199）上表现最好的种子 0。
