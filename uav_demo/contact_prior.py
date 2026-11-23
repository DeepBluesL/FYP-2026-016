"""接触概率：用网格推演的车位概率图，算每个动作执行后无人机离车不超过采集半径的概率。

两种用法（都只读延迟报告；网格推演取 goal 来源时会读真实目的地，只能作上限）：
- contact_probability 特征：17 个动作各自的接触概率，作为 DQN 输入；
- 接触先验 ω：Q 值上加 ω × [log(p_a + ε) − max_a log(p_a + ε)]，最好的动作为 0、其余为负，取代固定的规则先验 b。
只看下一步（报告后第 d+1 步）的车位分布；离车太远时各动作的概率都是 0，先验不起作用。
未收到报告或采集已结束时，概率全为 0。
"""

from __future__ import annotations

import numpy as np

from .environment import ACTIONS, preview_move
from .grid_prediction import GridPredictor, disk_sum  # noqa: F401（disk_sum 也供外部从这里导入）

CONTACT_EPSILON = 0.01  # log(p + ε)：p 为 0 时先验有下限，不会出现 −∞


class ContactModel:
    """包装网格推演预测器；按报告历史缓存“半径内概率和”图，同一报告下不同无人机位置复用。"""

    def __init__(self, predictor: GridPredictor):
        self.predictor = predictor
        self._cache = {}

    @property
    def needs_truth(self) -> bool:
        return self.predictor.needs_truth

    def probabilities(self, observation, config) -> np.ndarray:
        result = np.zeros(len(ACTIONS), dtype=np.float32)
        if (not observation["has_observation"] or observation["sensing_complete"]
                or observation["sensing_failed"]):
            return result
        history, age = observation["car_history"], observation["observation_age"]
        key = (tuple((t, tuple(p)) for t, p in history), age)
        if self.needs_truth or key not in self._cache:
            if len(self._cache) >= 20_000:
                self._cache.clear()
            goals = observation.get("true_car_goals") if self.needs_truth else None
            if self.needs_truth and goals is None:
                raise ValueError("goal 来源需要真实目的地序列，环境须以 expose_truth=True 创建")
            probability = self.predictor.distributions(history, config.grid_size, (age + 1,),
                                                       config.car_move_every, goals)[0]
            self._cache[key] = disk_sum(probability, config.sensing_radius)
        summed = self._cache[key]
        for action in range(len(ACTIONS)):
            position, _, _ = preview_move(observation["uav_pos"], observation["uav_speed"], action, config.grid_size)
            result[action] = summed[position]
        return result

    def export_state(self) -> dict:
        return self.predictor.export_state()

    @classmethod
    def from_state(cls, saved: dict) -> "ContactModel":
        return cls(GridPredictor.from_state(saved))
