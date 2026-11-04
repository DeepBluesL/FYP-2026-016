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


def encode_observation(observation, config, feature_names=DEFAULT_OBSERVATION_FEATURES):
    features = observation_features(observation, config)
    return np.concatenate([
        np.asarray(features[name], dtype=np.float32).reshape(-1)
        for name in feature_names
    ])
