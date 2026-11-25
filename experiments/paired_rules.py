"""Route-paired controller, predictor and point-selection experiments on GridWorld."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from uav_demo.config import EnvConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.extrapolate_pursuit import ExtrapolatePursuit
from uav_demo.linear_prediction import LinearPredictor
from uav_demo.point_selection import point_positions
from uav_demo.prediction import forecast_vehicle
from uav_demo.predictive_planner import PredictivePlanner
from uav_demo.prediction_data import split_route_seeds


def environment(delay):
    return EnvConfig(car_route="random", observation_delay=delay, auto_return=True,
                     reward_v2=True, movement_penalty=0.0)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def models_for(names, cfg):
    from uav_demo.direction_prediction import DirectionPredictor
    from uav_demo.goal_inference import GoalFilterPredictor
    from uav_demo.sequence_prediction import fit_sequence_predictor
    from uav_demo.var_prediction import VARPredictor
    from uav_demo.oracle_prediction import OraclePredictor
    groups = split_route_seeds(100, start_seed=100_000, split_seed=42)
    models = {"linear": LinearPredictor(), "direct": None, "oracle": OraclePredictor()}
    if "var" in names:
        models["var"], _ = VARPredictor.fit(cfg, groups, lags=4)
    for name in ("rnn", "lstm", "gru", "tcn"):
        if name not in names:
            continue
        models[name], _ = fit_sequence_predictor(cfg, groups, model_type=name, epochs=8,
                            seed=42, device="cpu", hidden_size=32, head_size=32)
    if "direction_lstm" in names:
        models["direction_lstm"], _ = DirectionPredictor.fit(cfg, groups, epochs=8, seed=42,
                             device="cpu", hidden_size=32, head_size=32, samples=128)
    if "goal_filter" in names:
        models["goal_filter"], _ = GoalFilterPredictor.fit(cfg, groups, samples=128)
    return models


def forecast_points(model, state, cfg, steps):
    if not state["has_observation"]:
        return {}
    if model is None:
        return {"native": np.array([state["car_pos"]] * len(steps))}
    if hasattr(model, "sample_positions"):
        offsets = tuple(state["observation_age"] + h for h in steps)
        samples = model.sample_positions(state["car_history"], cfg.grid_size, offsets)
        counts = [np.bincount(row[:, 0] * cfg.grid_size + row[:, 1], minlength=cfg.grid_size**2)
                  .reshape(cfg.grid_size, cfg.grid_size) for row in samples]
        return {point: point_positions(counts, point, cfg.sensing_radius) for point in ("mode", "disk")}
    return {"native": forecast_vehicle(model, state, cfg, steps)}


def route(cfg, model, name, seed, methods, replay):
    # Advance an independent observation clock even after a controller finishes.
    # This also gives every predictor the same offline scoring frames.
    clock = GridWorldEnv(cfg, expose_truth=name == "oracle")
    clock.reset(seed=seed)
    steps = tuple(range(35)) if any(c == "meet" for c, _ in methods) else (0, 1)
    envs, controllers, complete, rows, errors = {}, {}, {}, [], {}
    for key in methods:
        envs[key] = GridWorldEnv(cfg)
        envs[key].reset(seed=seed)
        controllers[key] = PredictivePlanner(model, cfg) if key[0] == "meet" else ExtrapolatePursuit(cfg)
        errors[key] = []
    for t in range(cfg.max_steps):
        if t:
            clock.car_observation_history.append((t, tuple(clock.car_trajectory[t])))
        clock.time = t
        state = clock._observation()
        points = forecast_points(model, state, cfg, steps)
        for key in methods:
            kind, point = key
            forecast = points.get(point)
            target = forecast[1] if forecast is not None else None
            if target is not None:
                errors[key].append(float(np.linalg.norm(target - clock.car_trajectory[t+1])))
            env = envs.get(key)
            if env is None:
                continue
            obs, mask = env._observation(), env.valid_action_mask()
            assert obs["car_history"] == state["car_history"]
            controller = controllers[key]
            if kind == "meet":
                action = controller.choose_action(obs, mask, predictions=forecast)
            else:
                action = controller.choose_action(obs, mask, car_target=target)
            _, _, term, trunc, info = env.step(action)
            if replay is not None:
                replay.append(dict(time=t, method=kind, point=point, uav=obs["uav_pos"],
                                   next_target=target.tolist() if target is not None else None,
                                   meeting=getattr(controller, "target", None), action=action,
                                   after=info["uav_pos"], sensing=info["sensing_progress"]))
            if info["sensing_complete"]:
                complete.setdefault(key, t+1)
            if term or trunc:
                rows.append(dict(delay=cfg.observation_delay, seed=seed, predictor=name, controller=kind,
                                 point=point, oracle=name=="oracle", success=int(info["success"]),
                                 collected=int(info["sensing_complete"]), acquisition=complete.get(key),
                                 completion=t+1 if info["success"] else cfg.max_steps))
                envs[key] = None
    for row in rows:
        err = np.array(errors[row["controller"],row["point"]])
        row.update(mean_error=err.mean(), radius_hit=(err <= cfg.sensing_radius).mean())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=20)
    parser.add_argument("--predictors", nargs="+", default=["linear"],
                        choices=("direct","linear","var","tcn","rnn","lstm","gru","direction_lstm","goal_filter","oracle"))
    parser.add_argument("--controllers", nargs="+", choices=["step","meet"], default=["step","meet"])
    parser.add_argument("--points", nargs="+", choices=["mode","disk"], default=["disk"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    models = models_for(args.predictors, environment(8))
    rows = []
    for delay in (8,12):
        for name in args.predictors:
            model = models[name]
            points = args.points if hasattr(model, "sample_positions") else ["native"]
            methods = [(controller, point) for controller in args.controllers for point in points]
            if name=="direct" and "meet" in args.controllers:
                raise ValueError("direct baseline uses step controller")
            for i in range(args.routes):
                replay = [] if i == 0 else None
                rows.extend(route(environment(delay), model, name, 3_000_000+i, methods, replay))
                if replay is not None:
                    (args.output/f"replay_{name}_{delay}.json").write_text(json.dumps(replay, default=int))
            write_csv(args.output/"routes.csv", rows)
    manifest = dict(
        environments=[asdict(environment(d)) for d in (8,12)], routes_per_delay=args.routes,
        predictors=args.predictors, controllers=args.controllers, points=args.points,
        base_features=[], base_dimension=0, network_input_dimension=0, hidden_sizes=[],
        policy_training_seeds=0, rule_prior=0, rule_source="selected predictor or latest report",
        network_horizons=[], rule_target_horizon=1, forecast_horizons=list(range(35)) if "meet" in args.controllers else [0,1],
        uses_true_future_by_predictor={name: name=="oracle" for name in args.predictors},
        predictor_seed_groups=split_route_seeds(100,start_seed=100_000,split_seed=42),
        neural_seed=42, neural_epochs=8, samples=128)
    (args.output/"protocol.json").write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
