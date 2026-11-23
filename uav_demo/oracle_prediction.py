"""真实位置对照：直接返回真实未来车位，只用于诊断实验，衡量“预测信息本身”最多能带来多少提升。

它读取观测里仅供诊断的 true_car_trajectory（环境须以 expose_truth=True 创建），不是可部署的预测器，结果只能作为上限。
"""

import numpy as np


class OraclePredictor:
    model_type = "oracle"

    def forecast(self, state, max_steps: int, steps) -> np.ndarray | None:
        """steps 相对当前时刻，与学习预测器的输出口径一致；未收到报告时同样返回 None。"""
        if not state["has_observation"]:
            return None
        if "true_car_trajectory" not in state:
            raise ValueError("真实位置对照需要以 expose_truth=True 创建环境")
        trajectory = state["true_car_trajectory"]
        return np.asarray([trajectory[min(state["time"] + h, max_steps)] for h in steps], dtype=np.float32)

    def export_state(self) -> dict:
        return {"model_type": self.model_type}
