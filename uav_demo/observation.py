"""命名观测到网络特征的唯一编码入口。"""
import numpy as np

DEFAULT_OBSERVATION_FEATURES = (
    "uav_pos", "uav_speed", "uav_goal", "car_pos", "time",
    "sensing_progress", "sensing_complete", "sensing_failed",
    "relative_car", "relative_uav_goal", "car_velocity", "car_move_phase",
    "observation_age", "has_observation",
)


def observation_features(observation, config):
    scale = config.grid_size - 1
    uav = np.asarray(observation["uav_pos"], dtype=np.float64)
    goal = np.asarray(observation["uav_goal"], dtype=np.float64)
    if observation["has_observation"]:
        car = np.asarray(observation["car_pos"], dtype=np.float64)
        car_position, relative_car = car / scale, (car - uav) / scale
        velocity = observation["car_velocity"]
    else:
        car_position, relative_car, velocity = (0, 0), (0, 0), (0, 0)
    return {
        "uav_pos": uav / scale,
        "uav_speed": observation["uav_speed"] / 2,
        "uav_goal": goal / scale,
        "car_pos": car_position,
        "time": observation["time"] / config.max_steps,
        "sensing_progress": observation["sensing_progress"] / config.sensing_steps,
        "sensing_complete": observation["sensing_complete"],
        "sensing_failed": observation["sensing_failed"],
        "relative_car": relative_car,
        "relative_uav_goal": (goal - uav) / scale,
        "car_velocity": velocity,
        "car_move_phase": (observation["time"] % config.car_move_every) / config.car_move_every,
        "observation_age": observation["observation_age"] / config.max_steps,
        "has_observation": observation["has_observation"],
    }


# 各特征的数值维度；rule_action 为外推规则所选动作的 17 维 one-hot，不在默认特征中。
FEATURE_SIZES = {
    "uav_pos": 2, "uav_speed": 1, "uav_goal": 2, "car_pos": 2, "time": 1,
    "sensing_progress": 1, "sensing_complete": 1, "sensing_failed": 1,
    "relative_car": 2, "relative_uav_goal": 2, "car_velocity": 2, "car_move_phase": 1,
    "observation_age": 1, "has_observation": 1, "rule_action": 17, "rule_distance": 17,
}


def rule_action_onehot(observation, config):
    """外推贪心在当前观测下选择的动作；只用延迟报告，不读真实车辆位置。"""
    from .environment import ACTIONS  # 延迟导入：environment 不依赖本模块，但避免循环
    from .extrapolate_pursuit import ExtrapolatePursuit
    onehot = np.zeros(len(ACTIONS), dtype=np.float32)
    action = ExtrapolatePursuit(config).choose_action(observation, np.ones(len(ACTIONS), dtype=bool))
    onehot[action] = 1.0
    return onehot


def rule_distance_gaps(observation, config):
    """各动作执行后到外推目标的距离比最近的动作多出多少格；规则所选动作为 0。"""
    from .extrapolate_pursuit import ExtrapolatePursuit
    distances = ExtrapolatePursuit(config).action_distances(observation)
    return (distances - distances.min()).astype(np.float32)


def feature_slice(feature_names, name):
    """特征 name 在拼接向量中的位置。"""
    start = sum(FEATURE_SIZES[n] for n in feature_names[:list(feature_names).index(name)])
    return slice(start, start + FEATURE_SIZES[name])


def encode_observation(observation, config, feature_names=DEFAULT_OBSERVATION_FEATURES):
    features = observation_features(observation, config)
    if "rule_action" in feature_names:
        features["rule_action"] = rule_action_onehot(observation, config)
    if "rule_distance" in feature_names:
        features["rule_distance"] = rule_distance_gaps(observation, config)
    return np.concatenate([
        np.asarray(features[name], dtype=np.float32).reshape(-1)
        for name in feature_names
    ])
