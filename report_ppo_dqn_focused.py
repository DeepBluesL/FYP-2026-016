"""Export validation curves, representative replays and an HTML experiment report."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_ppo_dqn_focused import CONDITIONS, SEEDS, job_name, summarize


def read_csv(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def condition_label(condition):
    route, sensing, delay, auto, predictor = condition
    return f"{route} / {sensing} / delay={delay} / return={auto} / {predictor}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ppo_dqn_focused64_20261125"))
    parser.add_argument("--report", type=Path, default=Path("results/ppo_dqn_focused64_20261125"))
    args = parser.parse_args()
    root, out = args.root, args.report
    summarize(root, out)
    import torch
    from uav_demo.agent import DQNAgent
    from uav_demo.ppo import PPOAgent
    from compare_pursuit import run_episode
    from uav_demo.visualization import render_episode
    torch.set_num_threads(1)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    curves, stability, selections, critic_diagnostics = [], [], [], []
    summaries = read_csv(out / "summary.csv")
    run_records = read_csv(out / "runs.csv")
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(13, 3.1 * len(CONDITIONS)), squeeze=False,
                             constrained_layout=True)
    representative = {}
    for index, condition in enumerate(CONDITIONS):
        for algorithm, color in (("dqn", "#2563eb"), ("ppo", "#d97706")):
            route, sensing, delay, auto, predictor = condition
            runs = []
            for seed in SEEDS:
                job = dict(algorithm=algorithm, route=route, sensing=sensing,
                           delay=delay, auto=auto, predictor=predictor, seed=seed)
                folder = root / job_name(job)
                if predictor != "none":
                    source = torch.load(root / "predictors" / f"{route}_{predictor}_{seed}.pt",
                                        map_location="cpu", weights_only=True)
                    checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
                    for name, weights in source["state_dict"].items():
                        torch.testing.assert_close(checkpoint["motion_predictor"]["state_dict"][name],
                                                   weights, rtol=0, atol=0)
                validation = read_csv(folder / "eval.csv")
                best = json.loads((folder / "best_validation.json").read_text())
                runs.append((seed, folder, best, validation))
                for row in validation:
                    curves.append(dict(condition=index, algorithm=algorithm, training_seed=seed, **row))
                last_five = [float(r["success_rate"]) for r in validation[-5:]]
                stability.append(dict(condition=index, algorithm=algorithm, training_seed=seed,
                    selected_at_environment_steps=best["environment_steps"],
                    best_validation_success=best["success_rate"],
                    final_validation_success=float(validation[-1]["success_rate"]),
                    last_five_success_min=min(last_five), last_five_success_max=max(last_five)))
                if algorithm == "ppo":
                    updates = read_csv(folder / "updates.csv")[-20:]
                    ev = [float(r["explained_variance"]) for r in updates
                          if np.isfinite(float(r["explained_variance"]))]
                    critic_diagnostics.append(dict(condition=index, training_seed=seed,
                        final_validation_success=float(validation[-1]["success_rate"]),
                        median_last20_preupdate_explained_variance=float(np.median(ev)) if ev else float("nan"),
                        mean_last20_entropy=float(np.mean([float(r["entropy"]) for r in updates]))))
            x = np.array([int(r["environment_steps"]) for r in runs[0][3]])
            for col, field in enumerate(("success_rate", "sensing_completion_rate")):
                values = np.array([[float(r[field]) for r in run[3]] for run in runs]) * 100
                ax = axes[index, col]
                ax.plot(x, values.mean(axis=0), color=color, label=algorithm.upper())
                ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=color, alpha=.15)
                ax.set(title=f"{index+1}. {condition_label(condition)}\n{field}",
                       xlabel="Environment steps", ylabel="Percent", ylim=(-2, 102))
                ax.grid(alpha=.2)
                ax.legend()
            ordered = sorted(runs, key=lambda r: (r[2]["success_rate"], r[2]["sensing_completion_rate"],
                -float(r[2]["mean_success_steps"] or 150), -r[2]["mean_travel_distance"], r[0]))
            representative[index, algorithm] = ordered[1]
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)
    for filename, rows in (("validation_curves.csv", curves), ("checkpoint_stability.csv", stability),
                           ("critic_diagnostics.csv", critic_diagnostics)):
        with (out / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    gallery = []
    for index, condition in enumerate(CONDITIONS):
        agents, tests = {}, {}
        for algorithm, agent_type in (("dqn", DQNAgent), ("ppo", PPOAgent)):
            seed, folder, _, _ = representative[index, algorithm]
            agent, config = agent_type.load(folder / "best_model.pt", device="cpu")
            agents[algorithm] = (agent, config, seed)
            tests[algorithm] = json.loads((folder / "test_episodes.json").read_text())
        dqn, ppo = tests["dqn"], tests["ppo"]
        assert [r["seed"] for r in dqn] == [r["seed"] for r in ppo]
        completion = lambda r: r["steps"] if r["success"] else 150
        difference = [completion(d)-completion(p) for d, p in zip(dqn, ppo)]
        common_success = [i for i, (d, p) in enumerate(zip(dqn, ppo)) if d["success"] and p["success"]]
        common_failure = [i for i, (d, p) in enumerate(zip(dqn, ppo)) if not d["success"] and not p["success"]]
        choices = []
        missing = []
        if common_success:
            ordered = sorted(common_success, key=lambda i: (completion(dqn[i])+completion(ppo[i]), i))
            choices.append(("median_common_success", ordered[len(ordered)//2]))
        else:
            missing.append("无共同成功路线")
        choices.append(("largest_absolute_gap", max(range(len(dqn)), key=lambda i: abs(difference[i]))))
        if common_failure:
            choices.append(("common_failure", common_failure[len(common_failure)//2]))
        else:
            missing.append("无共同失败路线")
        fig, axes = plt.subplots(len(choices), 2, figsize=(10, 4.4 * len(choices)), squeeze=False,
                                 constrained_layout=True)
        for row_index, (kind, route_index) in enumerate(choices):
            route_seed = dqn[route_index]["seed"]
            realized_routes = []
            for col, algorithm in enumerate(("dqn", "ppo")):
                agent, config, seed = agents[algorithm]
                result, episode = run_episode(agent, config, route_seed, algorithm)
                recorded = tests[algorithm][route_index]
                assert result["success"] == recorded["success"] and result["steps"] == recorded["steps"]
                realized_routes.append(episode["car_trajectory"])
                folder = out / "replays" / f"case{index+1}_{kind}_{algorithm}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "episode.json").write_text(json.dumps(episode, indent=2), encoding="utf-8")
                records = episode["records"]
                uav, car = (np.array([r[key] for r in records]) for key in ("uav_pos", "car_pos"))
                ax = axes[row_index, col]
                ax.plot(car[:, 0], car[:, 1], "--", color="#d97706", label="Vehicle (realized)")
                ax.plot(uav[:, 0], uav[:, 1], color="#2563eb", label="UAV")
                ax.scatter(*uav[0], color="#2563eb", marker="o")
                ax.scatter(*config.uav_goal, color="green", marker="*", s=120, label="UAV goal")
                sensed = next((r for r in records if r["sensing_complete"]), None)
                if sensed:
                    ax.scatter(*sensed["uav_pos"], color="purple", marker="s", label="Sensing complete")
                ax.set(xlim=(-1, 50), ylim=(-1, 50), aspect="equal",
                       title=f"{algorithm.upper()} train={seed} test={route_seed}\n{kind}: success={result['success']}, steps={result['steps']}")
                ax.grid(alpha=.2)
                if row_index == 0 and col == 0:
                    ax.legend(fontsize=8)
                selections.append(dict(condition=index, kind=kind, algorithm=algorithm,
                    training_seed=seed, test_seed=route_seed, success=result["success"],
                    steps=result["steps"], replay=str(folder.relative_to(out))))
                # Two anchor animations; all other cases retain complete JSON replays and static plots.
                if index == 0 and kind == "largest_absolute_gap":
                    render_episode(config, np.asarray(episode["car_trajectory"]), tuple(episode["car_goal"]),
                                   records, folder, policy=algorithm, save_gif=True)
            assert realized_routes[0] == realized_routes[1]
        filename = f"trajectories_{index+1}.png"
        fig.suptitle(condition_label(condition), fontsize=12)
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
        gallery.append(f"<h3>场景 {index+1}：{html.escape(condition_label(condition))}</h3>"
                       f"<p>{'；'.join(missing)}</p><img src='{filename}'>")
        if index == 0:
            gallery.append("<p>基准差异案例动画："
                           "<a href='replays/case1_largest_absolute_gap_dqn/episode.gif'>DQN</a> · "
                           "<a href='replays/case1_largest_absolute_gap_ppo/episode.gif'>PPO</a></p>")
    (out / "case_selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")

    table = "<table><tr><th>场景</th><th>算法</th><th>成功率（均值±SD）</th><th>逐种子成功率</th><th>采集完成率</th><th>完成步数（失败=150）</th></tr>"
    for index, condition in enumerate(CONDITIONS):
        for algorithm in ("dqn", "ppo"):
            r = next(r for r in summaries if r["algorithm"] == algorithm and
                     (r["route"], r["sensing"], int(r["delay"]), r["auto"] == "True", r["predictor"]) == condition)
            seed_values = [x for x in run_records if all(x[k] == r[k] for k in
                           ("algorithm", "route", "sensing", "delay", "auto", "predictor"))]
            seed_text = " / ".join(f"{float(x['success_rate']):.1%}" for x in sorted(seed_values, key=lambda x: int(x["seed"])))
            table += (f"<tr><td>{index+1}</td><td>{algorithm.upper()}</td>"
                      f"<td>{100*float(r['success_rate_mean']):.1f}% ± {100*float(r['success_rate_std']):.1f}pp</td>"
                      f"<td>{seed_text}</td>"
                      f"<td>{100*float(r['sensing_completion_rate_mean']):.1f}%</td>"
                      f"<td>{float(r['mean_capped_completion_steps_mean']):.2f}</td></tr>")
    table += "</table>"
    stability_table = "<table><tr><th>场景/算法/种子</th><th>选中步数</th><th>最佳验证</th><th>末轮验证</th><th>末5轮范围</th></tr>"
    for r in stability:
        stability_table += (f"<tr><td>{r['condition']+1}/{r['algorithm']}/{r['training_seed']}</td>"
            f"<td>{r['selected_at_environment_steps']}</td><td>{r['best_validation_success']:.1%}</td>"
            f"<td>{r['final_validation_success']:.1%}</td>"
            f"<td>{r['last_five_success_min']:.1%}–{r['last_five_success_max']:.1%}</td></tr>")
    stability_table += "</table>"
    protocol = json.loads((out / "protocol.json").read_text())
    large_drop = sum(round(100 * (r["best_validation_success"] - r["final_validation_success"])) >= 10 for r in stability)
    large_swing = sum(round(100 * (r["last_five_success_max"] - r["last_five_success_min"])) >= 20 for r in stability)
    misleading_ev = [r for r in critic_diagnostics if r["final_validation_success"] <= .10
                     and r["median_last20_preupdate_explained_variance"] >= .95]
    ev_note = ""
    if misleading_ev:
        example = misleading_ev[0]
        ev_note = (f"<p>解释方差不能替代任务指标。例如场景 {example['condition']+1} 的 PPO 种子 "
                   f"{example['training_seed']}：末 20 批更新前解释方差中位数为 "
                   f"{example['median_last20_preupdate_explained_variance']:.3f}，末轮验证成功率却只有 "
                   f"{example['final_validation_success']:.1%}。预测当前策略的回报可以很准确，同时策略仍经常失败。"
                   "这里比较的是训练末期诊断与末轮验证，不是选中 checkpoint 的测试成绩。"
                   "详见 <a href='critic_diagnostics.csv'>Critic 末期诊断</a>。</p>")
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PPO/DQN 代表性场景长训练报告</title>
<style>body{{max-width:1150px;margin:36px auto;padding:0 20px;font:16px/1.7 system-ui;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%;height:auto}}code{{background:#eee}}a{{color:#2563eb}}</style>
<h1>PPO / DQN 代表性场景长训练报告</h1>
<p>共 {protocol['configuration_count']} 组配置 × 3 种子，{protocol['run_count']} 次训练；每次 {protocol['training_steps_per_run']} 环境步。
每 10240 步用固定 50 个验证种子选模；最终使用新的 200 个共享测试种子 [7000000,7000200)。同一场景内两种算法面对相同车辆路线，fixed 与 random 的路线分布不同。验证与测试均采用合法动作中的贪心选择，PPO 取概率最大的动作。三种子结果报告均值与样本标准差，逐种子列按 42 / 10042 / 20042 排列，不把复用测试路线当作独立训练重复。</p>
<p>隐藏层 64×3；PPO 使用 Tanh，DQN 使用 LeakyReLU。GRU 复用旧实验的冻结权重（1000 路线，30 epoch），预测偏移 (0,1,4)。奖励和环境其余参数保持一致。GPU 并行训练、CPU 在线预测和最终评估，运行耗时不是严格硬件性能基准。完整设置见 <a href="protocol.json">protocol.json</a> 和 <a href="PLAN.md">预先计划</a>。</p>
<p>不含冻结预测器，无预测 DQN 优化 {protocol['parameter_counts']['dqn_none']['optimized_total']} 个参数，含目标网络保存 {protocol['parameter_counts']['dqn_none']['stored_total']} 个；PPO 的 Actor＋Critic 优化 {protocol['parameter_counts']['ppo_none']['optimized_total']} 个。加入预测特征后，优化参数分别为 {protocol['parameter_counts']['dqn_gru']['optimized_total']} 和 {protocol['parameter_counts']['ppo_gru']['optimized_total']}。相同环境步数对齐的是采样预算，不代表梯度更新次数或计算量相同。</p>
<h2>独立测试结果</h2><p>自动返程开启时，完成采集后的合法动作受到最短返程路线约束；这不等于策略独立学会返程。关闭自动返程的场景单独测试完整任务学习。只在同一场景内比较两种算法，不把不同难度场景混成一个整体排名。</p>{table}<img src="success_matrix.png">
<h2>学习曲线与训练充分性</h2><p>实线为三个种子平均，阴影为最小–最大。{large_drop}/{len(stability)} 次运行的最佳与末轮验证成功率相差至少 10 个百分点；{large_swing}/{len(stability)} 次运行的末 5 次验证成功率跨度至少 20 个百分点。这些阈值仅作描述，不是收敛检验。更大预算不保证收敛：末期仍上升、波动大或最佳与末轮差距大时，应保留未稳定的判断；不得把低成功率自动归因于算法本身。</p><img src="learning_curves.png">{stability_table}
<h2>典型轨迹及反例</h2><p>每个场景分别按验证成绩选中位数训练种子；在其配对测试路线上选共同成功中位案例、绝对完成时间差最大案例和共同失败案例。失败完成时间记 150；类别不存在时明确标注。案例可能重复，属于事后说明性展示，不是独立验证或整体表现的替代。蓝线 UAV、橙线车辆实际路线、紫点采集完成；真实路线只用于回放；每幅图仅显示该策略回合结束前的车辆路径，结束时刻不同会导致橙线长度不同，同场景配对的完整车辆轨迹已核对一致。各图的种子及选择依据见 <a href="case_selection.json">case_selection.json</a>。</p>
{''.join(gallery)}
<h2>完整数据与边界</h2><p><a href="summary.csv">汇总</a> · <a href="runs.csv">逐种子结果及参数量</a> · <a href="test_episodes.csv">逐测试回合</a> · <a href="ppo_updates.csv">PPO诊断</a> · <a href="validation_curves.csv">验证曲线数据</a> · <a href="checkpoint_stability.csv">选模与末轮稳定性</a></p>
<p>共审计 {protocol['audited_ppo_updates']} 次 PPO 更新，受检损失、KL、熵及裁剪比例均为有限数。其中 {protocol['critic_mse_improved_updates']} 次 Critic 批内 MSE 在更新后下降，{protocol['undefined_explained_variance_count']} 次解释方差未定义。批内拟合改善不等于任务成功或独立泛化。预算、测试种子、运行完成元信息、配对预测器哈希和冻结权重检查均通过。</p>
{ev_note}
<p>这是代表性单因素场景对照，不是完整因子组合，不覆盖所有预测器、奖励或超参数。零延迟对照在最终测试前因耗时预算被删除，本轮延迟统一为 6。本轮没有对两种算法做等量超参数搜索，比较的是当前实现与配置在相同步数下的表现，不是算法性能上限。三个种子不足以精确估计训练随机性的总体分布。没有按测试成绩更换种子或调参，模型选取仅使用验证集。源 checkpoint 和训练日志留在本地 outputs/ppo_dqn_focused64_20261125。</p>
<p>复现依赖本地旧预测器 <code>outputs/prediction_comparison/gru_0.pt</code>、<code>gru_1.pt</code>、<code>gru_2.pt</code> 及同目录的 <code>training.json</code>；它们未包含在报告内。执行 <code>python run_ppo_dqn_focused.py --prepare</code>；<code>python run_ppo_dqn_focused.py --workers 12</code>；<code>python report_ppo_dqn_focused.py</code>。Windows 环境设置 MKL_THREADING_LAYER=SEQUENTIAL。</p></html>'''
    (out / "report.html").write_text(report, encoding="utf-8")
    print(f"Report saved: {out / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
