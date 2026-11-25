#!/usr/bin/env bash
# 按任务清单训练，最多同时跑 P 个（限制是内存：每个约 0.55 GB）。已经训完 1500 局的任务直接跳过。
# 每行：<运行名> <train.py 的额外参数>。共用设置：1500 局、随机车辆路线、采完自动返程、势函数奖励、不计飞行距离代价。
# 结果写到 outputs/diag_v9/<运行名>/，与之前的实验放在一起，analyze_groups.py 从那里读。
# 用法：bash experiments/run_queue.sh experiments/jobs_rulep.txt [P]；用别的 Python 时设 PYTHON=/路径/python。
cd "$(dirname "$0")/.."
P=${2:-8}
export PYTHON=${PYTHON:-python}
run() {
  name=$1; shift
  out=outputs/diag_v9/$name
  [ -f "$out/best_model.pt" ] && [ -f "$out/eval.csv" ] && [ "$(tail -1 $out/eval.csv | cut -d, -f1)" = 1500 ] && return
  mkdir -p $out
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PYTHON train.py --episodes 1500 \
    --car-route random --auto-return --reward-v2 --movement-penalty 0 "$@" --output-dir $out > $out.log 2>&1
  echo "done $name: $(tail -2 $out.log | head -1)"
}
export -f run
grep -v '^#' "$1" | grep -v '^$' | xargs -P $P -L 1 bash -c 'run "$@"' _
echo ALL DONE
