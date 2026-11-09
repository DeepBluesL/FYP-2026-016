"""先划分路线种子，再逐条生成完整路线；一条路线就是一个样本。

复现时记录 split_route_seeds() 返回的种子列表和 EnvConfig。
迭代器按需生成路线，不切窗口、不写入轨迹文件。
"""

from collections.abc import Iterable, Iterator

import numpy as np

from .config import EnvConfig
from .vehicle import generate_vehicle_trajectory


def split_route_seeds(
        route_count: int = 1000, *, start_seed: int = 0, split_seed: int = 42,
) -> dict[str, list[int]]:
    """打乱连续路线种子后按 80% / 10% / 其余划分 train/validation/test。

    split_seed 只控制分组；每条路线仍由自己的路线种子生成。
    默认 1000 条路线得到互不重叠的 800 / 100 / 100 个种子。
    """
    if route_count < 10 or start_seed < 0:
        raise ValueError("`route_count` must be at least 10, and `start_seed` cannot be negative.")

    seeds = np.random.default_rng(split_seed).permutation(
        np.arange(start_seed, start_seed + route_count)
    ).tolist()
    train_end = route_count * 8 // 10
    validation_end = train_end + route_count // 10
    return {
        "train": seeds[:train_end],
        "validation": seeds[train_end:validation_end],
        "test": seeds[validation_end:],
    }


def iter_prediction_routes(
        config: EnvConfig,
        seeds: Iterable[int],
) -> Iterator[dict]:
    """对一个种子组逐条返回完整序列，保留 t=0..max_steps 的所有帧。

    每条样本包含：
      features: float32 [max_steps + 1, 3]，列为 x/(grid_size-1)、
                y/(grid_size-1)、报告测量时间/max_steps；没有位移。
      route_seed: 路线种子，用于追溯和重新生成路线。
      report_times: 原始测量时间，未加通信延迟。

    训练、验证、测试均使用各自组内的完整路线。预测标签的时间平移由
    后续训练代码处理；整条序列训练时，每个时刻的预测只能使用此前及
    当时的报告，不能读取后续位置。循环网络的隐藏状态需在路线间重置。
    """
    for seed in seeds:
        route, _ = generate_vehicle_trajectory(config, np.random.default_rng(seed))
        positions = route.astype(np.float32) / (config.grid_size - 1)
        report_times = np.arange(len(route), dtype=np.int64)
        times = report_times.astype(np.float32) / config.max_steps
        yield {
            "features": np.column_stack((positions, times)),
            "route_seed": int(seed),
            "report_times": report_times,
        }
