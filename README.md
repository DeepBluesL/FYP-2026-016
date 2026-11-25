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
