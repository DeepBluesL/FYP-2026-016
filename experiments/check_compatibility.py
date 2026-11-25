"""Compare inference in isolated interpreters against both original source trees."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np


def probe(source, checkpoint, algorithm, version7):
    sys.path.insert(0, str(source))
    import torch
    from uav_demo.agent import DQNAgent
    from uav_demo.environment import GridWorldEnv
    agent_type = DQNAgent
    if algorithm == "ppo":
        from uav_demo.ppo import PPOAgent
        agent_type = PPOAgent
    agent, cfg = agent_type.load(checkpoint)
    env = GridWorldEnv(cfg)
    obs, _ = env.reset(seed=3_000_001)
    results = []
    for t in range(32):
        if t in (0, 7, 12, 20, 31):
            # Construct the same delayed reports and platform state in both interpreters.
            history = tuple((i, (10 + i//2, 12 + (i+1)//2))
                            for i in range(t - cfg.observation_delay + 1))
            obs = dict(obs, time=t, has_observation=bool(history), car_history=history,
                       car_pos=history[-1][1] if history else None,
                       observation_age=cfg.observation_delay if history else 0,
                       uav_pos=(8,9), sensing_progress=0, sensing_complete=False, sensing_failed=False)
            if version7:
                obs["car_velocity"] = tuple(np.subtract(history[-1][1], history[-2][1])) if len(history)>1 else (0,0)
            x = torch.tensor(agent._features(obs))
            with torch.no_grad():
                y = (torch.cat((agent.model.actor(x), agent.model.value(x).reshape(1))) if algorithm=="ppo"
                     else torch.cat((agent.online(x), agent._q(agent.online,x))))
            results.append([x.tolist(),y.tolist()])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main",type=Path)
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--probe",nargs=4)
    parser.add_argument("--output",type=Path,default=Path("results/integration/compatibility.json"))
    args = parser.parse_args()
    if args.probe:
        source, checkpoint, alg, version = args.probe
        print(json.dumps(probe(source,checkpoint,alg,version=="7")))
        return
    root = Path(__file__).resolve().parents[1]
    report = []
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        archive = subprocess.check_output(["git","archive","--format=zip","6480b83","uav_demo"])
        zipfile.ZipFile(io.BytesIO(archive)).extractall(temp)
        spec_model = root/"outputs/spec_fixture/best_model.pt"
        cases = [
            ("main PPO",args.main, args.main/"outputs/ppo_dqn_focused64_20261125/ppo_random_consecutive_6_False_none_42/best_model.pt","ppo",9),
            ("main DQN TCN",args.main,args.main/"outputs/dqn_prediction_comparison/tcn_seed1/best_model.pt","dqn",9),
            ("spec DQN linear freshly trained",args.spec,spec_model,"dqn",9),
        ] + [(f"bundled v7 d{d}",temp,args.spec/f"deploy/models/d{d}.pt","dqn",7) for d in (4,6,8)]
        for label, source, checkpoint, alg, version in cases:
            def run(folder):
                args = [sys.executable,"-B",__file__,"--probe",str(folder),str(checkpoint),alg,str(version)]
                return json.loads(subprocess.check_output(args))
            original, merged = run(source), run(root)
            delta = np.max([np.max(np.abs(np.array(a[0])-b[0])) for a,b in zip(original,merged)])
            output = np.max([np.max(np.abs(np.array(a[1])-b[1])) for a,b in zip(original,merged)])
            report.append(dict(case=label,checkpoint=checkpoint.name,frames=len(original),
                               feature_max_difference=float(delta),output_max_difference=float(output)))
            assert delta==0 and output==0, report[-1]
    args.output.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
