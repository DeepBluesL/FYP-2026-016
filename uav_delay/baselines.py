"""规则基线入口：只追收到的位置 / 线性外推，在开发、验证、测试场景上评估。结果在 outputs/delay_check/。"""

from dataclasses import asdict
import json
from pathlib import Path

from .config import Config
from .evaluation import DEVELOPMENT, TEST, VALIDATION, evaluate
from .rules import chase


def main():
    cfg, rows = Config(), []
    for delay in (0, 2, 4, 6, 8):
        for name, predict in (("greedy", False), ("extrapolate", True)):
            for split, seeds in (("development", DEVELOPMENT), ("validation", VALIDATION),
                                 ("test", TEST)):
                result = evaluate(chase(predict), delay, cfg, seeds)
                summary = dict(split=split, delay=delay, policy=name,
                               **{k: v for k, v in result.items() if k != "episodes_detail"})
                print(json.dumps(summary))
                rows.append(summary)
    output = Path("outputs/delay_check")
    output.mkdir(parents=True, exist_ok=True)
    (output / "baselines.json").write_text(json.dumps(dict(config=asdict(cfg), results=rows),
                                                      indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
