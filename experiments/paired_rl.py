"""Validation-selected PPO/DQN paired comparison with a shared linear predictor."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from deploy.policy import Policy
from evaluate import evaluate_episode, check_test_seeds
from experiments.paired_rules import environment, write_csv
from train import train
from uav_demo.config import TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.linear_prediction import LinearPredictor
from uav_demo.observation import DEFAULT_OBSERVATION_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,10042])
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--routes", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/integration/rl"))
    parser.add_argument("--models", type=Path, default=Path("outputs/integration_rl"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, protocols = [], []
    predictor = LinearPredictor()
    for delay in (8,12):
        env_config = environment(delay)
        for seed in args.seeds:
            for algorithm, horizons, prior in [
                ("ppo",(4,8,12),0), ("ppo",(1,),0), ("dqn",(4,8,12),0),
                ("dqn",(1,),0), ("dqn",(1,),0.1)
            ]:
                run = f"{algorithm}_d{delay}_h{len(horizons)}_p{prior}_s{seed}"
                cfg = TrainConfig(algorithm=algorithm, seed=seed, total_steps=args.steps,
                    rollout_steps=256, eval_every_steps=1024, eval_episodes=8, test_episodes=args.routes,
                    prediction_model_type="linear", rule_source="predictor", rule_prior=prior,
                    observation_features=DEFAULT_OBSERVATION_FEATURES + ("rule_action",),
                    prediction_steps=horizons, device="cpu", prediction_device="cpu",
                    hidden_sizes=(64,64,64), train_pi_iters=10, train_v_iters=10,
                    epsilon_decay_steps=args.steps, learning_starts=256)
                root = args.models/run
                train(env_config, cfg, root, motion_predictor=predictor)
                policy = Policy(root/"best_model.pt")
                agent = policy.agent
                check_test_seeds(agent, cfg.test_seed, args.routes)
                env = GridWorldEnv(env_config)
                for i in range(args.routes):
                    result = evaluate_episode(agent, env, cfg.test_seed+i)
                    rows.append(dict(delay=delay, train_seed=seed, seed=result["seed"], algorithm=algorithm,
                        horizons="/".join(map(str,horizons)), prior=prior, success=int(result["success"]),
                        collected=int(result["sensing_complete"]), acquisition=result["sensing_complete_step"],
                        completion=result["steps"] if result["success"] else env_config.max_steps))
                protocols.append(dict(run=run, config=asdict(agent.config), env=asdict(env_config),
                                      metadata=agent.run_metadata))
                write_csv(args.output/"routes.csv",rows)
                (args.output/"protocol.json").write_text(json.dumps(protocols,indent=2))


if __name__ == "__main__":
    main()
