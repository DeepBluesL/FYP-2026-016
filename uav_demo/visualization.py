"""把一次完整任务保存为轨迹图和 GIF；也可直接回放保存的 episode.json。

通过独立进程执行本模块，可将绘图依赖与 PyTorch 的 OpenMP 运行库隔离。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import EnvConfig


def render_episode(
    config: EnvConfig,
    car_trajectory: np.ndarray,
    car_goal: tuple[int, int],
    records: list[dict],
    output_dir: Path,
    *,
    save_gif: bool = True,
    show: bool = False,
) -> dict[str, str]:
    """输出 ``trajectory.png`` 和可选 ``episode.gif``，返回文件绝对路径。

    records 应包含 t=0 和每个 step 后的状态。浅橙色虚线展示车辆
    整条已知轨迹，实线只展示当前时刻已经执行的部分。
    """
    if not records:
        raise ValueError("records must include at least the initial state")

    # pyplot 延迟导入，使服务器/测试环境也可以直接生成图片。
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    uav = np.asarray([record["uav_pos"] for record in records], dtype=float)
    car = np.asarray([record["car_pos"] for record in records], dtype=float)
    contact = np.asarray([record["contact"] for record in records], dtype=bool)
    completed = [i for i, record in enumerate(records) if record["sensing_complete"]]
    completion_index = completed[0] if completed else None
    trajectory = np.asarray(car_trajectory, dtype=float)

    blue, orange, green = "#2463CC", "#D98216", "#1C9469"
    fig = plt.figure(figsize=(10.4, 7.7), facecolor="#FAFBFD")
    layout = fig.add_gridspec(
        1, 2, width_ratios=(4.8, 1.65), left=0.07, right=0.98,
        bottom=0.09, top=0.88, wspace=0.11,
    )
    ax = fig.add_subplot(layout[0])
    panel = fig.add_subplot(layout[1])
    panel.axis("off")
    fig.suptitle(
        f"{config.grid_size} x {config.grid_size} GridWorld | UAV sensing mission",
        x=0.07, y=0.96, ha="left", fontsize=16, fontweight="bold", color="#192D45",
    )
    fig.text(
        0.07, 0.915, "Full vehicle route known  /  No obstacles  /  Eight-direction UAV",
        fontsize=10, color="#657386",
    )

    ax.set_facecolor("#F4F7FB")
    ax.set_xlim(-1, config.grid_size)
    ax.set_ylim(-1, config.grid_size)
    ax.set_aspect("equal")
    major_ticks = np.unique(np.r_[np.arange(0, config.grid_size, 5), config.grid_size - 1])
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticks(np.arange(config.grid_size), minor=True)
    ax.set_yticks(np.arange(config.grid_size), minor=True)
    ax.grid(which="major", color="#D8E0EA", linewidth=0.65)
    ax.grid(which="minor", color="#E8EDF3", linewidth=0.35)
    ax.tick_params(which="both", labelsize=8, colors="#64748B")
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("x (cells)", color="#64748B", fontsize=9)
    ax.set_ylabel("y (cells)", color="#64748B", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#CFD8E5")

    ax.plot(
        trajectory[:, 0], trajectory[:, 1], "--", color=orange,
        alpha=0.28, linewidth=2.2, zorder=2,
    )
    ax.scatter(*config.uav_start, s=120, color=green, edgecolors="white", zorder=7)
    ax.scatter(
        *config.uav_goal, s=175, color="#AC4260", marker="*",
        edgecolors="white", linewidths=0.8, zorder=7,
    )
    ax.scatter(
        *car_goal, s=95, marker="D", facecolors="none", edgecolors="#995D18",
        linewidths=1.5, zorder=6,
    )
    # 两个 UAV 端点的标签向图内偏移，避免右上角被裁切。
    ax.annotate(
        "START", config.uav_start, xytext=(8, 9), textcoords="offset points",
        fontsize=8, color=green, fontweight="bold", zorder=8,
    )
    ax.annotate(
        "STOP", config.uav_goal, xytext=(-8, -15), textcoords="offset points",
        ha="right", fontsize=8, color="#AC4260", fontweight="bold", zorder=8,
    )

    car_line, = ax.plot([], [], color=orange, linewidth=1.8, alpha=0.9, zorder=3)
    uav_line, = ax.plot([], [], color=blue, linewidth=2.0, zorder=4)
    contacts = ax.scatter([], [], s=22, color=green, alpha=0.9, zorder=5)
    sensing_finished = ax.scatter(
        [], [], s=110, marker="o", facecolors="none", edgecolors=green,
        linewidths=2, zorder=6,
    )
    uav_marker, = ax.plot(
        [], [], marker="^", markersize=10, color=blue,
        markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=9,
    )
    car_marker, = ax.plot(
        [], [], marker="s", markersize=8, color=orange,
        markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=8,
    )
    sensing_circle = Circle(
        tuple(uav[0]), config.sensing_radius, facecolor=blue, edgecolor=blue,
        alpha=0.12, linewidth=1.5, zorder=3,
    )
    ax.add_patch(sensing_circle)

    panel.text(0, 0.98, "MISSION STATUS", fontsize=10, fontweight="bold", color="#64748B")
    phase_text = panel.text(0, 0.915, "", fontsize=15, fontweight="bold", color=blue)
    status_text = panel.text(
        0, 0.85, "", va="top", fontsize=10, color="#34465C", linespacing=1.65,
    )
    legend_handles = [
        Line2D([], [], color=blue, linewidth=2, label="UAV path"),
        Line2D([], [], color=orange, linewidth=2, label="Car path so far"),
        Line2D([], [], color=orange, alpha=0.35, linestyle="--", label="Full car route"),
        Line2D([], [], color=green, marker="o", markersize=5, linestyle="none", label="Sensing contact"),
        Line2D([], [], color=green, marker="o", fillstyle="none", markersize=8, linestyle="none", label="Sensing finished"),
        Line2D([], [], color="#995D18", marker="D", fillstyle="none", linestyle="none", label="Car destination"),
    ]
    panel.legend(
        handles=legend_handles, loc="lower left", bbox_to_anchor=(-0.05, 0.13),
        frameon=False, fontsize=9, handlelength=2.4, labelspacing=1.1,
    )
    panel.text(
        0, 0.045,
        f"Car: 1 cell / {getattr(config, 'car_move_every', 2)} steps\n"
        "UAV grid step: 1 or 2\nCircle: sensing radius",
        fontsize=8.5, color="#64748B", linespacing=1.5,
    )

    def update(index: int):
        record = records[index]
        uav_line.set_data(uav[: index + 1, 0], uav[: index + 1, 1])
        car_line.set_data(car[: index + 1, 0], car[: index + 1, 1])
        uav_marker.set_data([uav[index, 0]], [uav[index, 1]])
        car_marker.set_data([car[index, 0]], [car[index, 1]])
        contacts.set_offsets(uav[: index + 1][contact[: index + 1]])
        if completion_index is not None and completion_index <= index:
            sensing_finished.set_offsets(uav[completion_index : completion_index + 1])
        else:
            sensing_finished.set_offsets(np.empty((0, 2)))
        sensing_circle.center = tuple(uav[index])
        sensing_circle.set_color(green if record["contact"] else blue)

        if record["success"]:
            phase, color = "SUCCESS", green
        elif index == len(records) - 1 and record["time"] >= config.max_steps:
            phase, color = "TIME LIMIT", "#AC4260"
        elif record["sensing_complete"]:
            phase, color = "To destination", blue
        elif record["sensing_progress"]:
            phase, color = "Collecting data", green
        else:
            phase, color = "Intercepting", blue
        phase_text.set_text(phase)
        phase_text.set_color(color)
        ux, uy = (int(value) for value in record["uav_pos"])
        cx, cy = (int(value) for value in record["car_pos"])
        mode = "Consecutive" if config.sensing_mode == "consecutive" else "Cumulative"
        status_text.set_text(
            f"Step       {record['time']} / {config.max_steps}\n"
            f"UAV       ({ux}, {uy})\n"
            f"Car        ({cx}, {cy})\n"
            f"Grid step  {record['uav_speed']}\n"
            f"Distance  {record['travel_distance']:.1f} cells\n\n"
            f"Sensing   {record['sensing_progress']} / {config.sensing_steps}\n"
            f"Radius     {config.sensing_radius:g} cells\n"
            f"{mode} sensing"
        )
        return uav_line, car_line, contacts, sensing_finished, uav_marker, car_marker, sensing_circle, phase_text, status_text

    paths: dict[str, str] = {}
    try:
        update(len(records) - 1)
        png_path = output_dir / "trajectory.png"
        fig.savefig(png_path, dpi=160, facecolor=fig.get_facecolor())
        paths["trajectory"] = str(png_path)
        animation = None
        if save_gif or show:
            # 尾帧保留一秒，方便看清是否完成采集并抵达终点。
            frames = [*range(len(records)), *([len(records) - 1] * 12)]
            animation = FuncAnimation(fig, update, frames=frames, interval=1000 / 12, blit=False)
        if save_gif:
            gif_path = output_dir / "episode.gif"
            animation.save(gif_path, writer=PillowWriter(fps=12), dpi=100)
            paths["animation"] = str(gif_path)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return paths


def main() -> None:
    """读取主程序保存的轨迹 JSON，不加载模型或导入 PyTorch。"""
    import argparse
    import json

    from .config import EnvConfig

    parser = argparse.ArgumentParser(description="Render a saved UAV episode without loading a model.")
    parser.add_argument("--episode", type=Path, required=True, help="Path to episode.json")
    parser.add_argument("--no-gif", action="store_true", help="Save only the trajectory PNG")
    parser.add_argument("--show", action="store_true", help="Also open an animation window")
    args = parser.parse_args()
    episode_path = args.episode.resolve()
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    paths = render_episode(
        EnvConfig(**episode["config"]),
        np.asarray(episode["car_trajectory"]),
        tuple(episode["car_goal"]),
        episode["records"],
        episode_path.parent,
        save_gif=not args.no_gif,
        show=args.show,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
