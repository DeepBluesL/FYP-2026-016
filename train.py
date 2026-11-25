"""训练入口：python train.py --episodes 1500 --output-dir outputs/train。"""

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from contextlib import ExitStack
from itertools import count

import numpy as np
import torch

from evaluate import evaluate_episode
from uav_demo.agent import DQNAgent
from uav_demo.ppo import PPOAgent
from uav_demo.config import ACTIVATION_NAMES, EnvConfig, TrainConfig
from uav_demo.environment import GridWorldEnv
from uav_demo.observation import encode_observation
from uav_demo.prediction_data import split_route_seeds
from uav_demo.linear_prediction import LinearPredictor
from uav_demo.oracle_prediction import OraclePredictor
from uav_demo.sequence_prediction import fit_sequence_predictor


def epsilon_at(step: int, config: TrainConfig) -> float:
    fraction = min(1.0, step / config.epsilon_decay_steps)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def evaluate(agent: DQNAgent | PPOAgent, env_config: EnvConfig, config: TrainConfig) -> dict:
    """使用固定且独立的验证种子，只执行 greedy policy，不更新模型；逐局逻辑复用 evaluate_episode。"""
    env = GridWorldEnv(env_config, expose_truth=config.needs_true_trajectory)
    rows = [evaluate_episode(agent, env, config.eval_seed + index) for index in range(config.eval_episodes)]
    successful_steps = [row["steps"] for row in rows if row["success"]]

    def mean(field):
        return float(np.mean([row[field] for row in rows]))

    return {
        "success_rate": mean("success"),
        "sensing_completion_rate": mean("sensing_complete"),
        "sensing_failure_rate": mean("sensing_failed"),
        "uav_goal_rate": mean("uav_reached_goal"),
        "mean_reward": mean("total_reward"),
        "mean_success_steps": float(np.mean(successful_steps)) if successful_steps else "",
        "mean_travel_distance": mean("travel_distance"),
    }


def fit_predictor(env_config: EnvConfig, config: TrainConfig, output_dir: Path):
    """按 prediction_model_type 建预测器；需要拟合的只用预测器训练路线，验证和最终测试路线都不进入拟合。"""
    if config.prediction_model_type == "oracle":
        return OraclePredictor()
    if config.prediction_model_type == "linear":
        return LinearPredictor()
    groups = split_route_seeds(
        route_count=config.prediction_routes,
        start_seed=config.prediction_seed,
        split_seed=config.prediction_split_seed,
    )
    if config.prediction_model_type == "goal_filter":
        from uav_demo.goal_inference import GoalFilterPredictor
        predictor, metrics = GoalFilterPredictor.fit(env_config, groups)
        print(f"Goal filter: turn probabilities={metrics['turn_probabilities']}, "
              f"goal bias={metrics['goal_bias']:.3f}, "
              f"validation log-likelihood per move={metrics['validation_log_likelihood']:.4f}", flush=True)
    elif config.prediction_model_type == "direction_lstm":
        from uav_demo.direction_prediction import DirectionPredictor
        print(f"Fitting {config.prediction_model_type} model on {len(groups['train'])} training routes...", flush=True)
        predictor, metrics = DirectionPredictor.fit(
            env_config, groups, epochs=config.prediction_epochs, seed=config.seed,
            learning_rate=config.prediction_learning_rate, batch_size=config.prediction_batch_size,
            device=config.prediction_device, hidden_size=config.prediction_hidden_size,
            head_size=config.prediction_head_size,
        )
        print(f"Motion model: validation log-likelihood per move="
              f"{metrics['best_validation_log_likelihood']:.4f}", flush=True)
    elif config.prediction_model_type == "var":
        from uav_demo.var_prediction import VARPredictor
        predictor, metrics = VARPredictor.fit(
            env_config, groups, lags=config.prediction_var_lags,
        )
    else:
        predictor, metrics = fit_sequence_predictor(
            env_config,
            groups,
            model_type=config.prediction_model_type,
            learning_rate=config.prediction_learning_rate,
            batch_size=config.prediction_batch_size,
            device=config.prediction_device,
            hidden_size=config.prediction_hidden_size,
            head_size=config.prediction_head_size,
            num_layers=config.prediction_num_layers,
            tcn_blocks=config.prediction_tcn_blocks,
            tcn_kernel_size=config.prediction_tcn_kernel_size,
            epochs=config.prediction_epochs,
            seed=config.seed,
        )

    (output_dir / "prediction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # 输出分布的预测器按 prediction_point 从分布里取点。
    if hasattr(predictor, "point"):
        predictor.point, predictor.radius = config.prediction_point, env_config.sensing_radius
    return predictor


def train(env_config: EnvConfig, config: TrainConfig, output_dir: str | Path, *, motion_predictor=None) -> DQNAgent | PPOAgent:
    # 验证集合固定复用；另留 2,000,000+ 作为最终测试种子。
    step_budget = config.total_steps
    config = replace(config, eval_every_steps=(
        config.eval_every_steps if config.eval_every_steps is not None else 10 * config.rollout_steps
    ))
    if step_budget is not None and min(step_budget, config.eval_every_steps) < 1:
        raise ValueError("total_steps 和 eval_every_steps 必须为正整数")
    if step_budget is not None and config.eval_every_steps % config.rollout_steps:
        raise ValueError("eval_every_steps 必须为 rollout_steps 的整数倍")
    train_seeds = range(config.seed, config.seed + (config.episodes if step_budget is None else 1))
    eval_seeds = range(config.eval_seed, config.eval_seed + config.eval_episodes)
    test_seeds = range(config.test_seed, config.test_seed + config.test_episodes)
    if config.test_seed < 0 or config.test_episodes < 1:
        raise ValueError("固定测试集需要非负种子和正数回合数")
    reserved_ranges = [train_seeds, eval_seeds]
    if config.use_motion_prediction:
        reserved_ranges.append(range(config.prediction_seed, config.prediction_seed + config.prediction_routes))
    if any(seeds.start < test_seeds.stop and test_seeds.start < seeds.stop for seeds in reserved_ranges):
        raise ValueError("固定测试种子与训练、验证或预测器种子重叠")
    if train_seeds.start < eval_seeds.stop and eval_seeds.start < train_seeds.stop:
        raise ValueError("训练与验证种子重叠")
    if config.use_motion_prediction:
        start = config.prediction_seed
        end = start + config.prediction_routes
        if any(start < seeds.stop and seeds.start < end for seeds in (train_seeds, eval_seeds)):
            raise ValueError("预测器种子与训练或验证种子重叠")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = GridWorldEnv(env_config, expose_truth=config.needs_true_trajectory)
    predictor = motion_predictor
    if predictor is not None and not config.use_motion_prediction:
        raise ValueError("A supplied predictor requires use_motion_prediction=True")
    if config.use_motion_prediction and predictor is None:
        predictor = fit_predictor(env_config, config, output_dir)
    sample, _ = env.reset(seed=config.seed)
    observation_size = encode_observation(sample, env_config, config.observation_features).size
    use_ppo = config.algorithm == "ppo"
    agent_type = {"ppo": PPOAgent, "dqn": DQNAgent}[config.algorithm]
    agent = agent_type(observation_size, env.action_count, config,
                      motion_predictor=predictor, env_config=env_config)
    config = agent.config
    agent.run_metadata = {
        "observation_features": list(config.observation_features), "observation_size": observation_size,
        "prediction_steps": list(config.prediction_steps), "network_input_size": agent.network_input_size,
        "rule_source": config.rule_source, "rule_prior": config.rule_prior, "hidden_sizes": list(config.hidden_sizes),
        "uses_true_future": config.network_uses_true_future,
        "run_complete": False,
        "run_train_seed_start": config.seed,
        "run_train_seed_end_exclusive": config.seed,
        "test_seed_start": test_seeds.start,
        "test_seed_end_exclusive": test_seeds.stop,
    }
    (output_dir / "config.json").write_text(
        json.dumps({"environment": asdict(env_config), "training": asdict(config)}, indent=2),
        encoding="utf-8",
    )
    counts = agent.parameter_counts()
    (output_dir / "parameter_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8",
    )
    print(f"Parameter counts: {counts}", flush=True)
    best_score = (-1.0, -1.0, -float("inf"), -float("inf"))
    training_columns = [
        "episode", "seed", "steps", "reward", "success", "sensing_complete",
        "sensing_failed", "uav_reached_goal",
        "travel_distance", "environment_steps", "episode_complete",
    ]
    training_columns += ([] if use_ppo else ["epsilon", "mean_loss"])
    training_columns.append("gradient_steps")
    eval_columns = [
        "episode", "environment_steps", "success_rate", "sensing_completion_rate", "mean_reward",
        "sensing_failure_rate", "uav_goal_rate",
        "mean_success_steps", "mean_travel_distance",
    ]
    with (
        ExitStack() as stack,
        (output_dir / "training.csv").open("w", newline="", encoding="utf-8") as train_file,
        (output_dir / "eval.csv").open("w", newline="", encoding="utf-8") as eval_file,
    ):
        writer = csv.DictWriter(train_file, fieldnames=training_columns)
        eval_writer = csv.DictWriter(eval_file, fieldnames=eval_columns)
        writer.writeheader()
        eval_writer.writeheader()
        def validate_and_save(episode):
            nonlocal best_score
            agent.run_metadata.update(
                selected_at_environment_steps=agent.environment_steps,
                selected_at_episode=episode,
            )
            metrics = evaluate(agent, env_config, config)
            eval_writer.writerow({"episode": episode, "environment_steps": agent.environment_steps, **metrics})
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
                    json.dumps({"episode": episode, "environment_steps": agent.environment_steps, **metrics}, indent=2), encoding="utf-8",
                )
            print(
                f"validation: success={metrics['success_rate']:.1%}, "
                f"sensing={metrics['sensing_completion_rate']:.1%}, "
                f"reward={metrics['mean_reward']:.1f}",
                flush=True,
            )

        update_index = 0
        if use_ppo:
            update_file = stack.enter_context((output_dir / "updates.csv").open("w", newline="", encoding="utf-8"))
            update_writer = csv.DictWriter(update_file, fieldnames=[
                "update", "episode", "environment_steps", "batch_steps", "policy_loss",
                "value_loss", "value_loss_before", "value_loss_after", "explained_variance",
                "approx_kl", "entropy", "clip_fraction", "policy_updates", "value_updates",
            ])
            update_writer.writeheader()
        episodes = count(1) if step_budget is not None else range(1, config.episodes + 1)
        for episode in episodes:
            seed = config.seed + episode - 1
            if step_budget is not None:
                if seed in eval_seeds or (config.use_motion_prediction and
                        config.prediction_seed <= seed < config.prediction_seed + config.prediction_routes):
                    raise ValueError("训练种子与验证或预测器种子重叠")
                if seed in test_seeds:
                    raise ValueError("训练种子与固定测试集重叠")
                config.episodes = episode  # checkpoint 记录实际已使用的训练种子范围
            agent.run_metadata["run_train_seed_end_exclusive"] = seed + 1
            observation, info = env.reset(seed=seed)
            mask = env.valid_action_mask()
            budget_reached = False
            episode_reward, losses = 0.0, []
            done = False
            while not done and not budget_reached:
                if use_ppo:
                    observation, reward, done, info = agent.collect_step(env, observation, mask)
                    mask = env.valid_action_mask()
                else:
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
                budget_reached = step_budget is not None and agent.environment_steps >= step_budget
                training_finished = budget_reached or (
                    step_budget is None and done and episode == config.episodes
                )
                if use_ppo and (len(agent.rollout) == config.rollout_steps or training_finished):
                    update_metrics = agent.update(observation)
                    update_index += 1
                    update_writer.writerow({"update": update_index, "episode": episode,
                                            "environment_steps": agent.environment_steps, **update_metrics})
                    update_file.flush()
                if step_budget is not None and (
                        agent.environment_steps % config.eval_every_steps == 0 or budget_reached):
                    validate_and_save(episode)
            if use_ppo:
                details = {}
            else:
                details = {"epsilon": epsilon, "mean_loss": float(np.mean(losses)) if losses else ""}
            writer.writerow(
                {
                    "episode": episode, "seed": seed, "steps": info["time"],
                    "environment_steps": agent.environment_steps, "episode_complete": int(done),
                    "reward": episode_reward, "success": int(info["success"]),
                    "sensing_complete": int(info["sensing_complete"]),
                    "sensing_failed": int(info["sensing_failed"]),
                    "uav_reached_goal": int(info["uav_reached_goal"]),
                    "travel_distance": info["travel_distance"], **details,
                    "gradient_steps": agent.gradient_steps,
                }
            )
            if episode == 1 or episode % config.log_every == 0 or training_finished:
                progress = (f"env_steps={agent.environment_steps}/{step_budget}" if step_budget is not None
                            else f"episode={episode}/{config.episodes}")
                print(
                    f"{progress} steps={info['time']} "
                    f"reward={episode_reward:.1f} success={info['success']} "
                    f"updates={agent.gradient_steps}",
                    flush=True,
                )
                train_file.flush()
            if step_budget is None and (episode % config.eval_every == 0 or episode == config.episodes):
                validate_and_save(episode)
            if budget_reached:
                break
    # 补全两个 checkpoint 的运行范围，保留各自选中时刻、权重和训练计数。
    agent.run_metadata["run_complete"] = True
    checkpoint_metadata = {}
    for filename in ("model.pt", "best_model.pt"):
        path = output_dir / filename
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["run_metadata"].update(
            run_complete=True,
            run_train_seed_end_exclusive=agent.run_metadata["run_train_seed_end_exclusive"],
        )
        temporary_path = path.with_suffix(".tmp")
        torch.save(saved, temporary_path)
        temporary_path.replace(path)
        checkpoint_metadata[filename] = saved["run_metadata"]
    (output_dir / "run_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps({"environment": asdict(env_config), "training": asdict(config)}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model and logs to {output_dir.resolve()}", flush=True)
    return agent


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="使用 DQN 或 PPO 训练 GridWorld 策略")
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default=defaults.algorithm)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--train-pi-iters", type=int, default=defaults.train_pi_iters,
                        help="Actor 最大更新次数，默认 80，受 KL 提前停止控制")
    parser.add_argument("--train-v-iters", type=int, default=defaults.train_v_iters,
                        help="Critic 更新次数，默认 80")
    parser.add_argument("--update-iters", type=int, default=defaults.update_iters,
                        help="兼容旧用法，设置两个更新次数；独立参数优先")
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip-ratio", type=float, default=defaults.clip_ratio)
    parser.add_argument("--target-kl", type=float, default=defaults.target_kl)
    parser.add_argument("--policy-lr", type=float, default=defaults.policy_lr)
    parser.add_argument("--value-lr", type=float, default=defaults.value_lr)
    parser.add_argument("--total-steps", type=int, default=None, help="按环境交互步数训练，设置后覆盖 episodes")
    parser.add_argument("--eval-every-steps", type=int, default=defaults.eval_every_steps)
    parser.add_argument("--test-seed", type=int, default=defaults.test_seed)
    parser.add_argument("--test-episodes", type=int, default=defaults.test_episodes)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default=defaults.device)
    parser.add_argument("--sensing-mode", choices=("consecutive", "cumulative"), default=EnvConfig().sensing_mode)
    parser.add_argument(
        "--observation-delay",
        type=int,
        default=EnvConfig().observation_delay,
        help="车辆位置观测延迟的环境步数，0 表示实时观测",
    )
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=defaults.hidden_sizes,
        help="DQN 与 PPO 共用的隐藏层宽度，例如 64 64 64",
    )
    parser.add_argument("--activation", choices=ACTIVATION_NAMES, default=defaults.activation,
                        help="默认 PPO 使用 tanh，DQN 使用 leaky_relu")
    parser.add_argument("--leaky-relu-slope", type=float, default=defaults.leaky_relu_slope)
    parser.add_argument("--no-motion-prediction", action="store_true", help="关闭学习预测特征，作当前状态消融对照")
    parser.add_argument("--prediction-routes", type=int, default=defaults.prediction_routes)
    parser.add_argument("--prediction-epochs", type=int, default=defaults.prediction_epochs)
    parser.add_argument("--prediction-seed", type=int, default=defaults.prediction_seed)
    parser.add_argument("--prediction-split-seed", type=int, default=defaults.prediction_split_seed)
    parser.add_argument("--prediction-learning-rate", type=float, default=defaults.prediction_learning_rate)
    parser.add_argument("--prediction-batch-size", type=int, default=defaults.prediction_batch_size)
    parser.add_argument("--prediction-hidden-size", type=int, default=defaults.prediction_hidden_size)
    parser.add_argument("--prediction-head-size", type=int, default=defaults.prediction_head_size)
    parser.add_argument("--prediction-num-layers", type=int, default=defaults.prediction_num_layers)
    parser.add_argument("--prediction-var-lags", type=int, default=defaults.prediction_var_lags,
                        help="VAR 使用的历史位置数量，默认 4")
    parser.add_argument("--prediction-tcn-blocks", type=int, default=defaults.prediction_tcn_blocks)
    parser.add_argument("--prediction-tcn-kernel-size", type=int, default=defaults.prediction_tcn_kernel_size)
    parser.add_argument("--rule-source", choices=("linear","predictor","oracle"), default="linear")
    parser.add_argument("--prediction-point", choices=("mode","disk"), default="disk")
    parser.add_argument("--prediction-model-type", default=defaults.prediction_model_type)
    parser.add_argument("--prediction-device", choices=("auto", "cuda", "cpu"), default=defaults.prediction_device)
    parser.add_argument("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)
    parser.add_argument(
        "--prediction-steps",
        type=int,
        nargs="+",
        default=defaults.prediction_steps,
        help="相对当前时刻的预测步数，例如 0 1 4",
    )
    parser.add_argument("--rule-prior", type=float, default=defaults.rule_prior,
                        help=">0 时启用外推规则先验：偏离规则的动作在 Q 值上扣该分数；需在 --observation-features 中加入 rule_action")
    parser.add_argument("--auto-return", action="store_true", help="采集结束后自动沿最短路径返回终点")
    parser.add_argument("--car-route", choices=("fixed", "random"), default=EnvConfig().car_route,
                        help="fixed：固定起终点；random：每局随机起点和目的地，到达后换新目的地")
    parser.add_argument("--movement-penalty", type=float, default=EnvConfig().movement_penalty,
                        help="每格飞行距离的代价（非正数）；设为 0 时奖励只看时间，与完成步数指标一致")
    parser.add_argument("--reward-v2", action="store_true",
                        help="进度与接触奖励改为势函数塑形、去掉非法移动扣分、随机路线超时未采完按失败扣分")
    parser.add_argument("--observation-features", nargs="+",
                        default=defaults.observation_features,
                        help="按顺序选择策略的命名观测特征")
    args = parser.parse_args()
    config = TrainConfig(
        total_steps=args.total_steps, eval_every_steps=args.eval_every_steps,
        test_seed=args.test_seed, test_episodes=args.test_episodes,
        algorithm=args.algorithm, rollout_steps=args.rollout_steps, update_iters=args.update_iters,
        train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters,
        gae_lambda=args.gae_lambda, clip_ratio=args.clip_ratio, target_kl=args.target_kl,
        policy_lr=args.policy_lr, value_lr=args.value_lr,
        episodes=args.episodes, seed=args.seed, device=args.device,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        log_every=args.log_every, hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation, leaky_relu_slope=args.leaky_relu_slope,
        use_motion_prediction=not args.no_motion_prediction,
        prediction_routes=args.prediction_routes, prediction_epochs=args.prediction_epochs,
        prediction_seed=args.prediction_seed,
        prediction_split_seed=args.prediction_split_seed,
        prediction_learning_rate=args.prediction_learning_rate,
        prediction_batch_size=args.prediction_batch_size,
        prediction_hidden_size=args.prediction_hidden_size,
        prediction_head_size=args.prediction_head_size,
        prediction_num_layers=args.prediction_num_layers,
        prediction_var_lags=args.prediction_var_lags,
        prediction_tcn_blocks=args.prediction_tcn_blocks,
        prediction_tcn_kernel_size=args.prediction_tcn_kernel_size,
        prediction_model_type=args.prediction_model_type, rule_source=args.rule_source, prediction_point=args.prediction_point,
        prediction_device=args.prediction_device,
        epsilon_decay_steps=args.epsilon_decay_steps,
        prediction_steps=tuple(args.prediction_steps),
        observation_features=tuple(args.observation_features),
        rule_prior=args.rule_prior,
    )
    env_config = EnvConfig(
        sensing_mode=args.sensing_mode,
        observation_delay=args.observation_delay,
        auto_return=args.auto_return,
        car_route=args.car_route,
        movement_penalty=args.movement_penalty,
        reward_v2=args.reward_v2,
        shaping_gamma=config.gamma,
    )
    train(env_config, config, args.output_dir or Path("outputs/ppo" if args.algorithm == "ppo" else "outputs/train"))


if __name__ == "__main__":
    main()
