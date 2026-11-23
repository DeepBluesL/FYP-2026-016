# UAV Path Planning for Dynamic Target Tracking and Sensing

**English** | [简体中文](README.zh-CN.md)

A UAV receives the car's position with a delay of d steps. It must stay within 2 cells of the car for 5 consecutive steps, then fly to its goal. The car follows a random route with hidden destinations. The method is developed on a 50×50 grid first; the final scenario will be a road network.

## Method

At each step a predictor forecasts the car's next position from the delayed reports. The rule moves one step toward that point. Double DQN (van Hasselt et al., AAAI 2016) sees the rule's action and the predicted point. Actions that differ from the rule lose `b` (`--rule-prior`). This is our own simplified form of residual Q-learning (Li et al., NeurIPS 2023): it uses a hard max instead of a soft max.

| `--prediction-model-type` | Method | Reference |
|---|---|---|
| `linear` (default) | Constant-velocity extrapolation | Schöller et al., RA-L 2020 |
| `direction_lstm` | LSTM classifier of the next move; multi-step forecast by sampling step by step | Wu et al., IJCAI 2017; Graves 2013 |
| `goal_filter` | Bayesian inference of the car's destination | Baker et al. 2009; Ziebart et al. 2009 |
| `oracle` | True future position, diagnostic upper bound only | — |

For `direction_lstm` and `goal_filter`, `--prediction-point disk` takes the cell where the car is most likely to be within the sensing radius. This matches the MR point of HOME (Gilles et al., ITSC 2021) with one point.

## Run

```powershell
python -m pip install -r requirements.txt
python train.py --car-route random --auto-return --reward-v2 --movement-penalty 0 --observation-delay 8 --prediction-model-type direction_lstm --rule-source predictor --rule-prior 0.1 --observation-features uav_pos uav_speed uav_goal car_pos time sensing_progress sensing_complete sensing_failed relative_car relative_uav_goal car_move_phase observation_age has_observation rule_action --output-dir outputs/example
python evaluate.py --checkpoint outputs/example/best_model.pt --episodes 200 --output-dir outputs/example_test
python demo.py --checkpoint outputs/example/best_model.pt --output-dir outputs/example_demo
python -m pytest -q
```

`experiments/`: `run_queue.sh` trains a job list (results go to `outputs/diag_v9/`). `analyze_groups.py` compares groups of seeds. `offline_predictors.py` checks predictor hit rates and calibration. `rule_only.py` runs the rule without learning.

## Code map

| File | Responsibility |
|---|---|
| `uav_demo/environment.py`, `vehicle.py`, `observation.py` | Grid world, car routes, delayed reports, features |
| `uav_demo/prediction.py` and `*_prediction.py`, `goal_inference.py` | Predictor interface and predictors |
| `uav_demo/point_selection.py` | Point from a distribution |
| `uav_demo/extrapolate_pursuit.py` | Rule: one greedy step toward the target |
| `uav_demo/agent.py`, `model.py`, `config.py`, `train.py` | Double DQN with rule prior, configuration, training |
| `evaluate.py`, `demo.py`, `uav_demo/visualization.py`, `planner.py` | Evaluation and replays |
| `deploy/` | Simulator-facing policy |

Code removed in the 2026-11 cleanup (MSE predictors, predictive planner, DRQN, `uav_delay/` and others) is at commit `5043846`. See also the [deployment guide](deploy/README.md), the [historical notes](README_LOCAL.md) and the [original plan](PROJECT_PLAN.md).
