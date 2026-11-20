import numpy as np

from .prediction_data import iter_prediction_routes


def route_regression(positions, lags):
    # 输入[1, x_t, y_t, x_(t-1), y_(t-1), ...]
    # 标签[x_(t+1), y_(t+1)]
    inputs = np.asarray([
        np.concatenate((
            [1.0],
            positions[t - lags:t][::-1].reshape(-1),))
        for t in range(lags, len(positions))], dtype=np.float64)

    targets = positions[lags:].astype(np.float64)
    return inputs, targets


class VARPredictor:
    def __init__(self, lags=4):
        if lags < 1:
            raise ValueError("lags must be pisitive integer")

        self.lags = lags
        self.coefficients = None

    @classmethod
    def fit(cls, config, groups, lags=4):
        if not 1 <= lags <= config.max_steps:
            raise ValueError("lags must be within the range 1..max_steps")

        model = cls(lags)
        inputs_parts = []
        targets_parts = []

        for sample in iter_prediction_routes(config, groups["train"]):
            position = sample["features"][:, :2]

            inputs, targets = route_regression(position, lags)
            inputs_parts.append(inputs)
            targets_parts.append(targets)

        inputs = np.concatenate(inputs_parts, axis=0)
        targets = np.concatenate(targets_parts, axis=0)

        model.coefficients = np.linalg.lstsq(inputs, targets, rcond=None)[0]
        validation_mse = model.evaluate(config, groups["validation"])

        metrics = {
            "model_type": "var",
            "lags": lags,
            "validation_mse": validation_mse,
            "seed_groups": groups,
        }
        return model, metrics

    def evaluate(self, config, seeds):
        # 使用真实历史评价一步位置预测，跳过历史不足的时刻
        squared_error = 0.0
        value_count = 0

        for sample in iter_prediction_routes(config, seeds):
            positions = sample["features"][:, :2]
            inputs, targets = route_regression(positions, self.lags)

            predicted = inputs @ self.coefficients

            squared_error += np.square(predicted - targets).sum()
            value_count += targets.size

        return float(squared_error / value_count)

    def predict_positions(
            self,
            history,
            grid_size,
            steps,
            move_every=1,
    ):
        # steps 相对最后报告时刻；返回原始网格位置
        steps = tuple(steps)
        times = np.asarray([t for t, _ in history])

        # 当前训练路线每个环境步都有一帧，推理沿用同一采样间隔。
        if np.any(np.diff(times) != 1):
            raise ValueError("VAR currently only supports location reports with an interval of 1.")

        scale = grid_size - 1
        positions = np.asarray([position for _, position in history],
                               dtype=np.float64, ) / scale

        last_position = positions[-1].copy()

        # 首报及历史不足期间保持最后位置。
        if len(positions) < self.lags:
            return np.repeat((last_position * scale)[None, :], len(steps), axis=0,
                             ).astype(np.float32)

        recent = positions[-self.lags:].copy()
        forecasts = [last_position]

        for _ in range(max(steps)):
            inputs = np.concatenate(([1.0], recent[::-1].reshape(-1),))
            next_position = inputs @ self.coefficients

            # 与现有序列预测器相同：递归时约束在地图内。
            next_position = np.clip(next_position, 0.0, 1.0)

            forecasts.append(next_position.copy())
            recent = np.vstack((recent[1:], next_position,))

        return (np.asarray([forecasts[h] for h in steps]) * scale).astype(np.float32)

    def export_state(self):
        return {
            "model_type": "var",
            "lags": self.lags,
            "coefficients": self.coefficients.tolist(),
        }
