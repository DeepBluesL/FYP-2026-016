# UAV Path Planning for Dynamic Target Tracking and Sensing

[Paired integration experiments, results and reproduction commands](results/integration/REPORT.md)

**English** | [简体中文](README.zh-CN.md)

A GridWorld research project combining vehicle motion prediction with Double DQN for UAV tracking, sensing, and navigation. Current implementation: November 9, 2026.

## Current behavior

- A 50×50 obstacle-free grid by default, with 17 UAV actions: hover and eight directions at two step sizes.
- Fixed or randomly generated vehicle routes, delayed position reports, and consecutive or cumulative sensing.
- Vehicle observations contain received history as `car_history=((measurement_time, (x, y)), …)`. No explicit vehicle displacement is passed to the policy. Before the first report, history is empty and `has_observation=False`.
- Default DQN observations have 18 dimensions; three predicted positions add six features. UAV state and sensing progress remain available.
- RNN, LSTM, and GRU predictors share a single implementation. Default architecture: input `(x,y,time)` → one 256-unit recurrent layer → Linear(256,128) → ReLU → Linear(128,2).
- The original MLP, greedy pursuit, motion extrapolation, predictive planning, and Oracle heuristic remain in the codebase. MLP and extrapolation infer motion from received history.
- Rule prior (`--rule-prior b`, needs `rule_action` in the features): Q-values of actions that differ from the rule lose `b`. With `--no-rule-action-input` the rule action is used only by the prior, not as a network input (20 inputs with one predicted point instead of 37); this is the spec branch's table 3 setting.

## Route data and training

The predictor uses 1,000 route seeds by default (`100000..100999`), shuffled into 800 training, 100 validation, and 100 test seeds before route generation. Each sample is a complete route, including stationary frames; there are no sliding windows. Routes are regenerated on demand, without storing all trajectories.

Training uses batches of 32 complete routes, retains the final partial batch, and shuffles training routes. At each timestep a unidirectional network predicts the next position. Validation selects the best checkpoint using normalized one-step position MSE. The predictor test group is reserved and is not automatically evaluated by the training entrypoint. Multi-step rollout accuracy and downstream task performance require separate evaluation.

The total predictor route count is independent of DQN episode count. Its seed range must not overlap DQN training/validation or final system evaluation routes. `config.json` records configuration; `prediction_metrics.json` records actual seed groups, training settings, and validation MSE.

## Install and run

Install a PyTorch build appropriate for the target machine, then install project dependencies:

```powershell
python -m pip install -r requirements.txt
python train.py --prediction-model-type lstm --prediction-learning-rate 0.001 --prediction-batch-size 32 --prediction-device auto --observation-delay 3 --output-dir outputs/lstm
python evaluate.py --checkpoint outputs/lstm/best_model.pt --policy dqn --episodes 100 --output-dir outputs/test_lstm
python demo.py --checkpoint outputs/lstm/best_model.pt --policy dqn --output-dir outputs/demo_lstm
python train.py --episodes 4000 --car-route random --auto-return --reward-v2 --movement-penalty 0 --observation-delay 8 --prediction-model-type direction_lstm --prediction-point disk --prediction-hidden-size 128 --prediction-head-size 64 --prediction-epochs 40 --rule-source predictor --prediction-steps 1 --rule-prior 0.1 --no-rule-action-input --hidden-sizes 128 128 128 --observation-features uav_pos uav_speed uav_goal car_pos time sensing_progress sensing_complete sensing_failed relative_car relative_uav_goal car_move_phase observation_age has_observation rule_action --output-dir outputs/example
```

Change `--prediction-model-type` to `rnn` or `gru` for comparisons. All predictor options are independent of DQN settings:

| Option | Default | Meaning |
|---|---|---|
| `--prediction-routes` | 1000 | Total routes across the three groups |
| `--prediction-seed` | 100000 | First route seed |
| `--prediction-split-seed` | 42 | Seed controlling group assignment |
| `--prediction-learning-rate` | 0.001 | Adam learning rate |
| `--prediction-batch-size` | 32 | Complete routes per batch |
| `--prediction-device` | auto | Prefer CUDA, otherwise CPU; explicit cpu/cuda also supported |
| `--prediction-hidden-size` | 256 | Recurrent width |
| `--prediction-head-size` | 128 | Output-head hidden width |
| `--prediction-num-layers` | 1 | Recurrent depth |
| `--prediction-epochs` | 20 | Predictor training epochs |
| `--prediction-steps` | 4 8 12 | Forecast offsets from the current decision time |

DQN device selection remains separate (`--device`, default CPU). `--no-motion-prediction` disables forecast features; `--policy predictive` selects predictive planning in evaluation/demo. Forecasting adds report age to requested horizons exactly once. Outputs are stored under `outputs/`.

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
| `uav_demo/prediction_data.py` | Seed splits and complete-route generation |
| `uav_demo/sequence_prediction.py` | Networks, route batching, training, rollout, serialization |
| `uav_demo/prediction.py` | Original MLP and shared forecast adapter |
| `uav_demo/environment.py`, `observation.py` | Delayed observations and feature encoding |
| `uav_demo/config.py`, `train.py` | Configuration, seed checks, training entrypoint |
| `uav_demo/agent.py` | Double DQN and predictor checkpoint loading |
| `evaluate.py`, `demo.py` | Evaluation and visualization |
| `deploy/policy.py` | Simulator-facing policy and report builder |

## Compatibility and validation

Checkpoint format is now **v9**. Earlier checkpoints require retraining; bundled legacy deployment weights and reference results are not current-model benchmarks. No new long training run or RNN/LSTM/GRU performance comparison has been completed. The local environment has CPU-only PyTorch, so actual CUDA execution remains unverified.

Tests can be run explicitly with `python -m unittest discover -s tests -q`. No tests were run as part of this documentation update.

Historical experiment details: [local experiment notes](README_LOCAL.md). The November 4 fixed-route results predate the current sequence models and should not be attributed to them. [Deployment guide](deploy/README.md) · [Original project plan and team assignments (Chinese)](PROJECT_PLAN.md).
