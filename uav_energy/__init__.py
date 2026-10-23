"""单 UAV 车辆采集实验，包含能量预算和主动定位查询。

建议阅读顺序：
1. env.py：物理参数、动作/观测定义和环境每步执行顺序。
2. policies.py：手写追踪规则和采样前瞻基线。
3. learn.py：当前完整任务的训练奖励、PPO 创建和训练循环。
4. mixed.py / phase_policy.py：混合动作分布和双阶段网络。

只提交以上核心模块和本文件。早期实验、批量评估、报告和作图脚本保留本地。
训练需要的固定场景环境在 env.py，回合评估和结果保存函数在 learn.py。
模型、JSON 记录和图片由运行时生成，不随 Python 源码提交。

依赖（与本地 requirements-energy.txt 相同）：numpy>=2.0、gymnasium>=1.0、
stable-baselines3>=2.4、torch>=2.0、matplotlib>=3.8、pillow>=10。

从仓库根目录运行（完整训练配置见 learn.py 的命令行参数）：
    python -m uav_energy.learn --mode B --output outputs/new_run

PPO 更新使用 Stable-Baselines3；源码分工参考 OpenAI Spinning Up：
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
"""
