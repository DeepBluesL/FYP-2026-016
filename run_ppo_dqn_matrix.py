"""Three-seed GPU integration matrix with paired predictor checkpoints and held-out tests."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

SEEDS = (42, 10042, 20042)


def jobs():
    result = []
    for algorithm, route, sensing, delay, auto, predictor, seed in itertools.product(
        ("dqn", "ppo"), ("fixed", "random"), ("consecutive", "cumulative"),
        (0, 6), (False, True), ("none", "gru"), SEEDS,
    ):
        result.append(dict(algorithm=algorithm, route=route, sensing=sensing,
                           delay=delay, auto=auto, predictor=predictor, seed=seed))
    for algorithm, predictor, seed in itertools.product(
        ("dqn", "ppo"), ("var", "rnn", "lstm", "tcn"), SEEDS,
    ):
        result.append(dict(algorithm=algorithm, route="random", sensing="cumulative",
                           delay=6, auto=True, predictor=predictor, seed=seed))
    return result


def job_name(job):
    return "_".join(str(job[key]) for key in
                    ("algorithm", "route", "sensing", "delay", "auto", "predictor", "seed"))


def prepare(root):
    import torch
    from uav_demo.config import EnvConfig
    from uav_demo.prediction_data import split_route_seeds
    from uav_demo.sequence_prediction import fit_sequence_predictor
    from uav_demo.var_prediction import VARPredictor

    torch.set_num_threads(1)
    groups = split_route_seeds(route_count=100, start_seed=100_000, split_seed=42)
    directory = root / "predictors"
    directory.mkdir(parents=True, exist_ok=True)
    keys = sorted({(j["route"], j["predictor"], j["seed"]) for j in jobs()
                   if j["predictor"] != "none"})
    for route, kind, seed in keys:
        path = directory / f"{route}_{kind}_{seed}.pt"
        if path.exists():
            continue
        env = EnvConfig(car_route=route)
        if kind == "var":
            predictor, metrics = VARPredictor.fit(env, groups)
        else:
            predictor, metrics = fit_sequence_predictor(
                env, groups, model_type=kind, epochs=2, seed=seed, device="cuda",
                hidden_size=64, head_size=32, batch_size=32,
            )
        torch.save(predictor.export_state(), path)
        path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Prepared {path.name}", flush=True)


def worker(root, index, steps):
    import numpy as np
    import torch
    from train import train
    from evaluate import check_test_seeds, evaluate_episode
    from uav_demo.agent import DQNAgent, restore_predictor
    from uav_demo.ppo import PPOAgent
    from uav_demo.config import EnvConfig, TrainConfig
    from uav_demo.environment import GridWorldEnv

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this matrix")
    job = jobs()[index]
    folder = root / job_name(job)
    folder.mkdir(parents=True, exist_ok=True)
    env = EnvConfig(car_route=job["route"], sensing_mode=job["sensing"],
                    observation_delay=job["delay"], auto_return=job["auto"],
                    reward_v2=True, movement_penalty=0)
    cfg = TrainConfig(
        algorithm=job["algorithm"], seed=job["seed"], total_steps=steps, device="cuda",
        rollout_steps=1024, eval_every_steps=steps, eval_episodes=10,
        test_seed=6_000_000, test_episodes=30, use_motion_prediction=job["predictor"] != "none",
        prediction_model_type=job["predictor"], prediction_routes=100, prediction_epochs=2,
        prediction_hidden_size=64, prediction_head_size=32, prediction_device="cpu",
    )
    predictor = None
    predictor_hash = None
    if cfg.use_motion_prediction:
        source = root / "predictors" / f"{job['route']}_{job['predictor']}_{job['seed']}.pt"
        predictor = restore_predictor(torch.load(source, map_location="cpu", weights_only=True), "cpu")
        predictor_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    start = time.perf_counter()
    agent = train(env, cfg, folder, motion_predictor=predictor)
    network = agent.model if job["algorithm"] == "ppo" else agent.online
    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
        raise RuntimeError("Non-finite model parameters")
    if agent.environment_steps != steps:
        raise RuntimeError("Environment step budget mismatch")
    agent_type = PPOAgent if job["algorithm"] == "ppo" else DQNAgent
    # CPU evaluation releases inference from CUDA synchronization overhead for these small MLPs.
    loaded, config = agent_type.load(folder / "best_model.pt", device="cpu")
    if loaded.motion_predictor is not None and hasattr(loaded.motion_predictor, "to"):
        loaded.motion_predictor.to("cpu")
    check_test_seeds(loaded, cfg.test_seed, cfg.test_episodes)
    test_env = GridWorldEnv(config)
    rows = [evaluate_episode(loaded, test_env, seed)
            for seed in range(cfg.test_seed, cfg.test_seed + cfg.test_episodes)]
    (folder / "test_episodes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    successful = [r["steps"] for r in rows if r["success"]]
    result = dict(
        **job, environment_steps=steps, seconds=time.perf_counter() - start,
        success_rate=float(np.mean([r["success"] for r in rows])),
        sensing_completion_rate=float(np.mean([r["sensing_complete"] for r in rows])),
        mean_reward=float(np.mean([r["total_reward"] for r in rows])),
        mean_success_steps=float(np.mean(successful)) if successful else None,
        predictor_sha256=predictor_hash, parameter_counts=agent.parameter_counts(),
        training=asdict(agent.config), environment=asdict(env),
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
    )
    (folder / "done.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Finished {job_name(job)} in {result['seconds']:.1f}s", flush=True)


def summarize(root, report_dir):
    records = [json.loads((root / job_name(job) / "done.json").read_text()) for job in jobs()]
    if len({r["environment_steps"] for r in records}) != 1:
        raise ValueError("Training step budgets differ")
    groups = {}
    test_rows = []
    run_rows = []
    update_rows = []
    ppo_update_count = 0
    undefined_explained_variance_count = 0
    keys = ("algorithm", "route", "sensing", "delay", "auto", "predictor")
    for record in records:
        if record["training"]["device"] != "cuda":
            raise ValueError("Every policy must have trained on CUDA")
        key = tuple(record[k] for k in keys)
        groups.setdefault(key, []).append(record)
        run_rows.append({k: record[k] for k in (*keys, "seed", "environment_steps", "seconds",
                                               "success_rate", "sensing_completion_rate", "mean_reward",
                                               "mean_success_steps", "predictor_sha256")})
        episode_file = root / job_name(record) / "test_episodes.json"
        episodes = json.loads(episode_file.read_text())
        if [episode["seed"] for episode in episodes] != list(range(6_000_000, 6_000_030)):
            raise ValueError("Held-out route range differs across runs")
        efficiency = {
            "mean_episode_steps": statistics.mean(r["steps"] for r in episodes),
            "mean_travel_distance": statistics.mean(r["travel_distance"] for r in episodes),
            "mean_capped_completion_steps": statistics.mean(
                r["steps"] if r["success"] else record["environment"]["max_steps"] for r in episodes
            ),
        }
        record.update(efficiency)
        run_rows[-1].update(efficiency)
        metadata = json.loads((episode_file.parent / "run_metadata.json").read_text())["best_model.pt"]
        if not metadata["run_complete"] or metadata["selected_at_environment_steps"] != record["environment_steps"]:
            raise ValueError("Incomplete run or unexpected checkpoint selection time")
        run_rows[-1].update(metadata)
        run_rows[-1]["optimized_parameters"] = record["parameter_counts"]["optimized_total"]
        run_rows[-1]["stored_parameters"] = record["parameter_counts"]["stored_total"]
        if record["algorithm"] == "ppo":
            with (episode_file.parent / "updates.csv").open() as file:
                updates = list(csv.DictReader(file))
            if sum(int(r["batch_steps"]) for r in updates) != record["environment_steps"]:
                raise ValueError("PPO rollout coverage does not match training budget")
            for update in updates:
                for field in ("policy_loss", "value_loss_before", "value_loss_after", "approx_kl", "entropy", "clip_fraction"):
                    if not math.isfinite(float(update[field])):
                        raise ValueError(f"Non-finite PPO metric {field}")
                undefined_explained_variance_count += math.isnan(float(update["explained_variance"]))
                update_rows.append({**{k: record[k] for k in keys}, "training_seed": record["seed"], **update})
            ppo_update_count += len(updates)
        for episode in episodes:
            test_rows.append({**{k: record[k] for k in keys}, "training_seed": record["seed"], **episode})
    summaries = []
    for key, values in groups.items():
        if sorted(r["seed"] for r in values) != sorted(SEEDS):
            raise ValueError("Every configuration requires all three seeds")
        row = dict(zip(keys, key))
        for field in ("success_rate", "sensing_completion_rate", "mean_reward",
                      "mean_episode_steps", "mean_travel_distance", "mean_capped_completion_steps"):
            samples = [r[field] for r in values]
            row[field + "_mean"] = statistics.mean(samples)
            row[field + "_std"] = statistics.stdev(samples)
        row["seeds"] = len(values)
        summaries.append(row)
    # Pairing check includes the same predictor bytes for each algorithm pair.
    for job in jobs():
        if job["algorithm"] == "dqn":
            first = next(r for r in records if all(r[k] == job[k] for k in job))
            paired = {**job, "algorithm": "ppo"}
            second = next(r for r in records if all(r[k] == paired[k] for k in paired))
            if first["predictor_sha256"] != second["predictor_sha256"]:
                raise ValueError("Algorithm pair used different predictor weights")
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("runs.csv", run_rows), ("summary.csv", summaries),
                           ("test_episodes.csv", test_rows), ("ppo_updates.csv", update_rows)):
        with (report_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    protocol = {
        "training_steps_per_run": records[0]["environment_steps"],
        "training_seeds": SEEDS, "configuration_count": len(groups), "run_count": len(records),
        "test_seed_range": [6_000_000, 6_000_030], "validation_seed_range": [1_000_000, 1_000_010],
        "predictor_seed_range": [100_000, 100_100], "predictor_epochs": 2,
        "predictor_hidden_size": 64, "predictor_head_size": 32,
        "aggregation": "Mean and sample standard deviation across three training-seed results",
        "capped_completion_steps": "Successful episode length; unsuccessful episodes assigned max_steps (150)",
        "gpu": records[0]["gpu"], "torch_version": records[0]["torch_version"],
        "audited_ppo_updates": ppo_update_count,
        "undefined_explained_variance_count": undefined_explained_variance_count,
        "critic_mse_improved_updates": sum(
            float(r["value_loss_after"]) < float(r["value_loss_before"]) for r in update_rows
        ),
        "scope": "Short-budget integration and learning comparison; convergence has not been established",
        "evaluation_device": "cpu", "training_device": "cuda",
        "predictor_fit_device": "cuda (VAR uses CPU least squares)", "online_forecast_device": "cpu",
        "base_configuration": records[0]["training"],
        "algorithm_configurations": {
            algorithm: next(r["training"] for r in records if r["algorithm"] == algorithm)
            for algorithm in ("dqn", "ppo")
        },
        "environment_defaults": records[0]["environment"],
        "parameter_counts": {
            f"{algorithm}_{predictor}": next(
                r["parameter_counts"] for r in records
                if r["algorithm"] == algorithm and r["predictor"] == predictor
            )
            for algorithm in ("dqn", "ppo") for predictor in ("none", "gru")
        },
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("train.py", "evaluate.py", "uav_demo/agent.py", "uav_demo/ppo.py",
                         "uav_demo/ppo_model.py", "uav_demo/model.py", "uav_demo/config.py",
                         "uav_demo/environment.py", "run_ppo_dqn_matrix.py")
        },
        "predictor_metrics": {p.stem: json.loads(p.read_text()) for p in (root / "predictors").glob("*.json")},
    }
    (report_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = list(itertools.product(
        ("fixed", "random"), ("consecutive", "cumulative"), (0, 6), (False, True), ("none", "gru")
    ))
    means = np.zeros((len(conditions), 2))
    deviations = np.zeros_like(means)
    for i, condition in enumerate(conditions):
        for j, algorithm in enumerate(("dqn", "ppo")):
            values = groups[(algorithm, *condition)]
            means[i, j] = 100 * statistics.mean(r["success_rate"] for r in values)
            deviations[i, j] = 100 * statistics.stdev(r["success_rate"] for r in values)
    fig, ax = plt.subplots(figsize=(11, 14))
    plot = ax.imshow(means, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks([0, 1], ["DQN", "PPO"])
    labels = [f"{route} | {sensing} | delay {delay} | return {'on' if auto else 'off'} | {predictor}"
              for route, sensing, delay, auto, predictor in conditions]
    ax.set_yticks(range(len(conditions)), labels, fontsize=9)
    for i in range(len(conditions)):
        for j in range(2):
            ax.text(j, i, f"{means[i, j]:.1f} ± {deviations[i, j]:.1f}",
                    ha="center", va="center", fontsize=9, color="white" if means[i, j] > 55 else "black")
    ax.set_title(f"Test success (%) — mean ± sample SD over 3 training seeds\n"
                 f"{records[0]['environment_steps']} training steps; 30 paired test routes per run", fontsize=12)
    fig.colorbar(plot, ax=ax, label="Success rate (%)", fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(report_dir / "success_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Summarized {len(records)} runs / {len(groups)} configurations / {len(test_rows)} episodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ppo_dqn_matrix_20261125"))
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--job", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize", type=Path, metavar="REPORT_DIR")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.root, args.summarize)
        return
    if args.prepare:
        prepare(args.root)
        return
    if args.job is not None:
        worker(args.root, args.job, args.steps)
        return
    (args.root / "manifest.json").write_text(json.dumps(
        {"jobs": jobs(), "steps": args.steps, "workers": args.workers}, indent=2), encoding="utf-8")
    def launch(index):
        folder = args.root / job_name(jobs()[index])
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / "done.json"
        if done.exists():
            if json.loads(done.read_text())["environment_steps"] != args.steps:
                raise ValueError("Use a new output directory when changing step budget")
            return
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, "-u", __file__, "--root", str(args.root),
                 "--steps", str(args.steps), "--job", str(index)],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "MKL_THREADING_LAYER": "SEQUENTIAL", "OMP_NUM_THREADS": "1"},
            )
        print(f"{index + 1}/{len(jobs())} {job_name(jobs()[index])} exit={completed.returncode}", flush=True)
        if completed.returncode:
            raise RuntimeError(f"Worker failed; see {folder / 'run.log'}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(launch, range(args.limit or len(jobs()))))


if __name__ == "__main__":
    main()
