"""线性预测器：沿最近一次观测到的位移方向匀速外推。

作为与 LSTM 等同接口的预测器，供规则（rule_source=predictor）和 DQN 预测特征共用同一个估计。
无参数、无需训练；输出按格取整，与 extrapolate_pursuit.predicted_car 保持一致。
"""

import numpy as np

from .prediction import observed_motion


class LinearPredictor:
    model_type = "linear"

    def predict_positions(self, history, grid_size: int, steps, move_every: int = 1) -> np.ndarray:
        """steps 相对最后报告时刻；每步按车辆移动时钟计入可移动次数。"""
        report_time, position = history[-1]
        velocity = observed_motion(history, move_every)
        last = grid_size - 1
        forecasts = []
        for step in steps:
            moves = sum((report_time + k) % move_every == 0 for k in range(1, step + 1))
            point = np.clip(np.asarray(position, dtype=np.float64) + moves * velocity, 0, last)
            forecasts.append(np.trunc(point))
        return np.asarray(forecasts, dtype=np.float32)

    def export_state(self) -> dict:
        return {"model_type": self.model_type}
