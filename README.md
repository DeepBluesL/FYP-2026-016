# UAV Path Planning for Dynamic Target Tracking and Sensing

**English** | [简体中文](README.zh-CN.md)

A UAV receives the car's position with a delay of d steps. It must stay within 2 cells of the car for 5 consecutive steps, then fly to its goal. The car follows a random route with hidden destinations. The method is developed on a 50×50 grid first; the final scenario will be a road network.

## Method

At each step a predictor forecasts the car's next position from the delayed reports. The rule moves one step toward that point. Double DQN (van Hasselt et al., AAAI 2016) sees the predicted point. The rule's action is used only by the prior (`--no-rule-action-input`): actions that differ from the rule lose `b` (`--rule-prior`). This is our own simplified form of residual Q-learning (Li et al., NeurIPS 2023): it uses a hard max instead of a soft max.

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
python train.py --episodes 4000 --car-route random --auto-return --reward-v2 --movement-penalty 0 --observation-delay 8 --prediction-model-type direction_lstm --prediction-point disk --rule-source predictor --rule-prior 0.1 --no-rule-action-input --observation-features uav_pos uav_speed uav_goal car_pos time sensing_progress sensing_complete sensing_failed relative_car relative_uav_goal car_move_phase observation_age has_observation rule_action --output-dir outputs/example
python evaluate.py --checkpoint outputs/example/best_model.pt --episodes 200 --output-dir outputs/example_test
python demo.py --checkpoint outputs/example/best_model.pt --output-dir outputs/example_demo
python -m pytest -q
```

`experiments/`: `run_queue.sh` trains a job list (results go to `outputs/diag_v9/`). `analyze_groups.py` compares groups of seeds. `offline_predictors.py` checks predictor hit rates and calibration. `rule_only.py` runs the rule without learning.

## Completed experiments

| Date | By | Setting | Result |
|---|---|---|---|
| 10-22 | lhy | 50×50, car starts from a fixed corner, no delay, 1 training seed | 100 test routes: DQN and greedy pursuit both 100% success, 43.8 / 43.6 steps |
| 11-04 | lhy | 15×15, fixed route, delay 3 | 100 test routes: DQN 95% success, predictive planner 59% |
| 11-09 to 11-11 | lhy | RNN / LSTM / GRU, TCN and VAR predictors trained with MSE | Later compared on random routes: all worse than linear prediction; the direction LSTM was adapted from their LSTM and MLP direction classifier |
| 11-05 to 11-24 | karnes | Random routes, delay 8 / 12, 10 seeds each, failure counted as 150 steps | Pure DQN 100.8 / 122.4 steps; rule + prior: linear 56.8 / 80.9, direction LSTM 53.7 / 72.0, goal filter 53.8 / 69.1 |
| 11-25 | karnes | Rule action only in the prior, not in the network input; delay 6 / 8 / 12, 3 seeds | 1500 episodes (`jobs_table3_4k_*.txt` without `--episodes 4000`): direction LSTM 15.4 / 9.0 / 2.7 steps worse than with the rule action as input. 4000 episodes: linear within 4 steps at all delays; direction LSTM d=6 seed 0 equal (43.9). Table 3 is being rerun this way with 4000 episodes (`jobs_table3_4k_*.txt`) |

Settings and commands for the first three rows: [README_LOCAL.md](README_LOCAL.md).

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
