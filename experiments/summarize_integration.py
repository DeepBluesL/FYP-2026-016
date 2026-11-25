"""Summarize matched routes. Each learned policy seed has its own paired row."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.paired_rules import write_csv

ROOT = Path("results/integration")


def read(group):
    with (ROOT/group/"routes.csv").open() as stream:
        return list(csv.DictReader(stream))


def main():
    groups = defaultdict(dict)
    methods = defaultdict(list)
    for group in ("controllers","predictors","rl","oracle"):
        for row in read(group):
            method = (f"{row['algorithm']}_{row['horizons']}_{row['prior']}" if group=="rl" else
                      f"{row['predictor']}_{row['controller']}_{row['point']}")
            if method not in methods[group]:
                methods[group].append(method)
            key = (row.get("train_seed",""), int(row["seed"]))
            arm = groups[group,int(row["delay"]),method]
            assert key not in arm
            arm[key] = row

    summaries, differences, pairs = [], [], []
    for (group,delay,method), arm in groups.items():
        rows = list(arm.values())
        sensed = [int(r["acquisition"]) for r in rows if r["acquisition"]]
        summaries.append(dict(group=group,delay=delay,method=method,
            policy_train_seeds=len({k[0] for k in arm if k[0]}),routes=len({k[1] for k in arm}),
            success=mean(int(r["success"]) for r in rows),collected=mean(int(r["collected"]) for r in rows),
            acquisition_if_collected=mean(sensed) if sensed else None,
            completion=mean(int(r["completion"]) for r in rows),
            mean_error=mean(float(r["mean_error"]) for r in rows) if group!="rl" else None,
            radius_hit=mean(float(r["radius_hit"]) for r in rows) if group!="rl" else None))
    def compare(group, delay, a, b, base_group=None):
        arm, base = groups[group,delay,a], groups[base_group or group,delay,b]
        matched = []
        for (train_seed,seed), row in arm.items():
            ref = base[train_seed if base_group is None else "",seed]
            diff = {f"delta_{f}":int(row[f])-int(ref[f]) for f in ("success","collected","completion")}
            matched.append(diff)
            pairs.append(dict(group=group,delay=delay,method=a,reference=b,train_seed=train_seed,seed=seed,**diff))
        differences.append(dict(group=group,delay=delay,method=a,reference=b,n=len(matched),
                                **{f:mean(r[f] for r in matched) for f in matched[0]}))
    for delay in (8,12):
        compare("controllers",delay,"linear_meet_native","linear_step_native")
        for method in methods["predictors"]:
            if method != "linear_step_native":
                compare("predictors",delay,method,"linear_step_native")
        compare("predictors",delay,"direction_lstm_step_disk","direction_lstm_step_mode")
        for alg in ("dqn","ppo"):
            compare("rl",delay,alg+"_1_0",alg+"_4/8/12_0")
        for h in ("1","4/8/12"):
            compare("rl",delay,"dqn_"+h+"_0","ppo_"+h+"_0")
        compare("rl",delay,"dqn_1_0.1","dqn_1_0")
        for method in ("dqn_1_0","dqn_1_0.1"):
            compare("rl",delay,method,"linear_step_native",base_group="controllers")
    write_csv(ROOT/"summary.csv",summaries)
    write_csv(ROOT/"paired_summary.csv",differences)
    write_csv(ROOT/"paired_routes.csv",pairs)


if __name__=="__main__":
    main()
