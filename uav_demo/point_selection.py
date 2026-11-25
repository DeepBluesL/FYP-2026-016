"""从车位概率图取一个点预测，供规则追踪和 DQN 输入使用。

mode：概率最大的格子。
disk：半径内概率最大点，即车落在该格欧氏半径 radius 内概率最大的格子。采集看的是车是否在半径内，disk 让这个事件的概率最大；
mode 只让「恰好在这一格」的概率最大。做法同 HOME（Gilles et al., ITSC 2021）的 MR 取点只取一个点（K=1）时的形式。
"""

from __future__ import annotations

import numpy as np

POINT_RULES = ("mode", "disk")


def most_likely_positions(maps) -> np.ndarray:
    """概率图 [n, x, y] 中每张图概率最大的格子 [n, 2]；并列时取下标最小的格子。"""
    maps = np.asarray(maps)
    flat = maps.reshape(len(maps), -1).argmax(axis=1)
    return np.stack(np.unravel_index(flat, maps.shape[-2:]), axis=-1).astype(np.float32)


def point_positions(maps, point: str = "mode", radius: float = 2.0) -> np.ndarray:
    """概率图 [n, x, y] → 点预测 [n, 2]。

    disk 并列时取本格概率最大的（例如概率全在一格时取这一格本身），仍并列再取下标最小的格子；mode 并列取下标最小的格子。
    """
    if point == "mode":
        return most_likely_positions(maps)
    if point != "disk":
        raise ValueError(f"point 只能取 {POINT_RULES}")
    scores = []
    for probability in maps:
        probability = np.asarray(probability, dtype=np.float64)
        total = disk_sum(probability, radius)
        tied = np.isclose(total, total.max(), rtol=1e-12, atol=0.0)
        scores.append(np.where(tied, probability, -1.0))
    return most_likely_positions(scores)


def disk_sum(probability, radius: float) -> np.ndarray:
    """S[x, y] = 以 (x, y) 为圆心、欧氏半径 radius 内（与环境的采集判定相同）各格概率之和。"""
    size = probability.shape[0]
    reach = int(np.floor(radius))
    total = np.zeros_like(probability)
    for dx in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            # total[x, y] += probability[x + dx, y + dy]，越界部分不加。
            target = (slice(max(-dx, 0), size - max(dx, 0)), slice(max(-dy, 0), size - max(dy, 0)))
            source = (slice(max(dx, 0), size + min(dx, 0)), slice(max(dy, 0), size + min(dy, 0)))
            total[target] += probability[source]
    return total
