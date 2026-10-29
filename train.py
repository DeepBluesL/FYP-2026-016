"""训练入口：python train.py --episodes 1500 --output-dir outputs/train。"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from uav_demo.agent import DQNAgent
from uav_demo.config import ACTIVATION_NAMES, EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.prediction import MotionPredictor


def epsilon_at(step: int, config: TrainConfig) -> float:
    fraction = min(1.0, step / config.epsilon_decay_steps)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def evaluate(agent: DQNAgent, env_config: EnvConfig, config: TrainConfig) -> dict:
    """使用固定且独立的验证种子，只执行 greedy policy，不更新模型。"""
    env = GridWorldEnv(env_config)
    rewards, successes, sensing, successful_times, distances = [], [], [], [], []
    sensing_failures, goal_arrivals = [], []
    for index in range(config.eval_episodes):
        observation, info = env.reset(seed=config.eval_seed + index)
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(observation, epsilon=0.0, mask=env.valid_action_mask())
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
        successes.append(info["success"])
        sensing.append(info["sensing_complete"])
        sensing_failures.append(info["sensing_failed"])
        goal_arrivals.append(info["uav_reached_goal"])
        distances.append(info["travel_distance"])
        if info["success"]:
            successful_times.append(info["time"])
    return {
        "success_rate": float(np.mean(successes)),
        "sensing_completion_rate": float(np.mean(sensing)),
        "sensing_failure_rate": float(np.mean(sensing_failures)),
        "uav_goal_rate": float(np.mean(goal_arrivals)),
        "mean_reward": float(np.mean(rewards)),
        "mean_success_steps": float(np.mean(successful_times)) if successful_times else "",
        "mean_travel_distance": float(np.mean(distances)),
    }


def train(env_config: EnvConfig, config: TrainConfig, output_dir: str | Path) -> DQNAgent:
    # 验证集合固定复用；另留 2,000,000+ 作为最终测试种子。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({"environment": asdict(env_config), "training": asdict(config)}, indent=2),
        encoding="utf-8",
    )
    env = GridWorldEnv(env_config)
    predictor = None
    if config.use_motion_prediction:
        # 只使用训练种子生成监督样本；验证和最终测试路线都不能进入预测器拟合。
        prediction_count = config.prediction_route_count
        print(f"Fitting vehicle motion model on {prediction_count} training routes...", flush=True)
        predictor, prediction_metrics = MotionPredictor.fit(
            env_config, range(config.seed, config.seed + prediction_count),
            epochs=config.prediction_epochs, seed=config.seed,
        )
        (output_dir / "prediction_metrics.json").write_text(
            json.dumps(prediction_metrics, indent=2), encoding="utf-8",
        )
        print(f"Motion model: direction accuracy={prediction_metrics['holdout_accuracy']:.1%}, "
              f"straight baseline={prediction_metrics['straight_baseline_accuracy']:.1%}", flush=True)
    agent = DQNAgent(env.observation_size, env.action_count, config,
                     motion_predictor=predictor, env_config=env_config)
    best_score = (-1.0, -1.0, -float("inf"), -float("inf"))
    training_columns = [
        "episode", "seed", "steps", "reward", "success", "sensing_complete",
        "sensing_failed", "uav_reached_goal",
        "travel_distance", "epsilon", "mean_loss", "gradient_steps",
    ]
    eval_columns = [
        "episode", "success_rate", "sensing_completion_rate", "mean_reward",
        "sensing_failure_rate", "uav_goal_rate",
        "mean_success_steps", "mean_travel_distance",
    ]
    with (
        (output_dir / "training.csv").open("w", newline="", encoding="utf-8") as train_file,
        (output_dir / "eval.csv").open("w", newline="", encoding="utf-8") as eval_file,
    ):
        writer = csv.DictWriter(train_file, fieldnames=training_columns)
        eval_writer = csv.DictWriter(eval_file, fieldnames=eval_columns)
        writer.writeheader()
        eval_writer.writeheader()
        for episode in range(1, config.episodes + 1):
            seed = config.seed + episode - 1
            observation, info = env.reset(seed=seed)
            mask = env.valid_action_mask()
            episode_reward, losses = 0.0, []
            done = False
            while not done:
                epsilon = epsilon_at(agent.environment_steps, config)
                action = agent.choose_action(observation, epsilon, mask)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_mask = env.valid_action_mask()
                done = terminated or truncated
                agent.remember(
                    observation, action, reward, next_observation, done, next_mask
                )
                if (
                    agent.environment_steps >= config.learning_starts
                    and agent.environment_steps % config.train_every == 0
                ):
                    loss = agent.update()
                    if loss is not None:
                        losses.append(loss)
                observation, mask = next_observation, next_mask
                episode_reward += reward
            writer.writerow(
                {
                    "episode": episode, "seed": seed, "steps": info["time"],
                    "reward": episode_reward, "success": int(info["success"]),
                    "sensing_complete": int(info["sensing_complete"]),
                    "sensing_failed": int(info["sensing_failed"]),
                    "uav_reached_goal": int(info["uav_reached_goal"]),
                    "travel_distance": info["travel_distance"], "epsilon": epsilon,
                    "mean_loss": float(np.mean(losses)) if losses else "",
                    "gradient_steps": agent.gradient_steps,
                }
            )
            if episode == 1 or episode % config.log_every == 0 or episode == config.episodes:
                print(
                    f"episode={episode}/{config.episodes} steps={info['time']} "
                    f"reward={episode_reward:.1f} success={info['success']} "
                    f"epsilon={epsilon:.3f} updates={agent.gradient_steps}",
                    flush=True,
                )
                train_file.flush()
            if episode % config.eval_every == 0 or episode == config.episodes:
                metrics = evaluate(agent, env_config, config)
                eval_writer.writerow({"episode": episode, **metrics})
                eval_file.flush()
                agent.save(output_dir / "model.pt", env_config)
                # 同样成功率下偏好更快完成，避免反复接触刷奖励被选为最佳。
                success_steps = metrics["mean_success_steps"]
                score = (metrics["success_rate"], metrics["sensing_completion_rate"],
                         -float(success_steps) if success_steps != "" else -float("inf"),
                         -metrics["mean_travel_distance"])
                if score > best_score:
                    best_score = score
                    agent.save(output_dir / "best_model.pt", env_config)
                    (output_dir / "best_validation.json").write_text(
                        json.dumps({"episode": episode, **metrics}, indent=2), encoding="utf-8",
                    )
                print(
                    f"validation: success={metrics['success_rate']:.1%}, "
                    f"sensing={metrics['sensing_completion_rate']:.1%}, "
                    f"reward={metrics['mean_reward']:.1f}",
                    flush=True,
                )
    print(f"Saved model and logs to {output_dir.resolve()}", flush=True)
    return agent


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="当前状态 + 学习预测 DQN：先采集，再到 UAV 终点。")
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=defaults.device)
    parser.add_argument("--sensing-mode", choices=("consecutive", "cumulative"), default=EnvConfig().sensing_mode)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=defaults.hidden_sizes,
        help="各隐藏层宽度，例如 256 128 64；数字个数就是隐藏层数",
    )
    parser.add_argument("--activation", choices=ACTIVATION_NAMES, default=defaults.activation)
    parser.add_argument("--leaky-relu-slope", type=float, default=defaults.leaky_relu_slope)
    parser.add_argument("--no-motion-prediction", action="store_true", help="关闭学习预测特征，作当前状态消融对照")
    parser.add_argument("--prediction-routes", type=int, default=defaults.prediction_routes)
    parser.add_argument("--prediction-epochs", type=int, default=defaults.prediction_epochs)
    parser.add_argument("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)
    args = parser.parse_args()
    config = TrainConfig(
        episodes=args.episodes, seed=args.seed, device=args.device,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        log_every=args.log_every, hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation, leaky_relu_slope=args.leaky_relu_slope,
        use_motion_prediction=not args.no_motion_prediction,
        prediction_routes=args.prediction_routes, prediction_epochs=args.prediction_epochs,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    train(EnvConfig(sensing_mode=args.sensing_mode), config, args.output_dir)


if __name__ == "__main__":
    main()
