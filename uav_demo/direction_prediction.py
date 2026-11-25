"""方向概率 LSTM 预测器：逐步输出车辆下一步走北 / 东 / 南 / 西的概率，点预测取 k 步后最可能的格子（或 2 格内概率最大的格子）。

做法对应 CSSRNN（Wu et al., IJCAI 2017）在网格上的形式：RNN 读历史，对下一步的合法移动做屏蔽 softmax，用交叉熵训练；
不学平均位置，所以车直走时点预测不会落在车后面。往前推 k 步时按模型概率逐步抽 samples 条路径（自回归抽样，Graves 2013），
取第 k 步出现次数最多的格子（并列取下标最小）；point=disk 时改取「车落在半径 radius 内」的抽样比例最大的格子
（半径内概率最大点，见 point_selection.py）。每次预测用固定的随机种子抽样，同样的历史总是给出同样的点。
只建模随机路线、车辆每步移动一格（car_route=random，car_move_every=1）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .point_selection import POINT_RULES, point_positions
from .vehicle import DIRECTIONS, generate_vehicle_trajectory


def prediction_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def encode_positions(positions, grid_size: int) -> np.ndarray:
    """连续报告位置 [T, 2] → 网络输入 [T, 4]：位置 / (grid_size − 1)、到达该位置的那一步移动 dx/dy（首个为 0）。"""
    positions = np.asarray(positions, dtype=np.float32)
    previous = np.vstack((np.zeros((1, 2), dtype=np.float32), np.diff(positions, axis=0)))
    return np.column_stack((positions / (grid_size - 1), previous)).astype(np.float32)


def legal_directions(positions, grid_size: int) -> np.ndarray:
    """[..., 2] 位置上四个方向是否仍在地图内 → [..., 4]。"""
    targets = np.asarray(positions)[..., None, :] + DIRECTIONS
    return np.all((targets >= 0) & (targets < grid_size), axis=-1)


def route_examples(config, seeds) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """每条路线一个样本：各时刻输入 [T, 4]、下一步方向序号 [T]、下一步合法方向 [T, 4]。"""
    inputs, targets, legal = [], [], []
    for seed in seeds:
        route, _ = generate_vehicle_trajectory(config, np.random.default_rng(seed))
        moves = np.diff(route, axis=0)
        inputs.append(encode_positions(route[:-1], config.grid_size))
        targets.append((moves[:, None, :] == DIRECTIONS).all(axis=-1).argmax(axis=1))
        legal.append(legal_directions(route[:-1], config.grid_size))
    return tuple(torch.as_tensor(np.stack(column)) for column in (inputs, targets, legal))


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)


class DirectionPredictor(nn.Module):
    model_type = "direction_lstm"

    def __init__(self, hidden_size: int = 128, head_size: int = 64, samples: int = 2000, sample_seed: int = 0,
                 point: str = "mode", radius: float = 2.0):
        super().__init__()
        if point not in POINT_RULES:
            raise ValueError(f"point 只能取 {POINT_RULES}")
        self.point, self.radius = point, float(radius)
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.samples = samples
        self.sample_seed = sample_seed
        self._noise, self._constant_tables, self._graphs = {}, {}, {}
        self.lstm = nn.LSTM(4, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_size), nn.ReLU(), nn.Linear(head_size, 4))

    def forward(self, inputs, state=None):
        output, state = self.lstm(inputs, state)
        return self.head(output), state

    @staticmethod
    def _check_route(config) -> None:
        if config.car_route != "random" or config.car_move_every != 1:
            raise ValueError("方向概率 LSTM 只建模随机路线、每步移动一格（car_route=random，car_move_every=1）")

    @torch.no_grad()
    def _sample_cells(self, history, grid_size: int, steps) -> np.ndarray:
        """按模型概率从最后报告位置抽 samples 条路径，返回 steps 各步所在格子的扁平序号 x × grid_size + y，[len(steps), samples]。

        先用 nn.LSTM 读完报告历史，再由 _rollout 往前抽样。GPU 上把 _rollout 录成 CUDA graph 重放：
        抽样每步都是很小的算子，逐个启动时耗时主要在启动上，几个训练进程共用一块 GPU 时尤其明显。
        """
        self.eval()
        device = next(self.parameters()).device
        report = int(history[-1][1][0]) * grid_size + int(history[-1][1][1])
        inputs = encode_positions([position for _, position in history], grid_size)
        logits, (h, c) = self(torch.as_tensor(inputs[None], device=device))
        start = (logits[0, -1], h[0, 0], c[0, 0])
        if device.type == "cuda":
            key = (grid_size, steps, str(device), tuple(p.data_ptr() for p in self.parameters()))
            if key not in self._graphs:
                self._graphs[key] = _SamplingGraph(self, grid_size, steps, device)
            return self._graphs[key].run(report, *start)
        return self._rollout(grid_size, steps, torch.tensor(report, device=device), *start).cpu().numpy()

    def _rollout(self, grid_size: int, steps, report, logits, h, c) -> torch.Tensor:
        """从扁平序号 report、历史读完后的 logits [4] 与 LSTM 状态 h, c [hidden] 出发抽样，只用张量运算（可录成 CUDA graph）。

        少用算子：输入（到达格 / (grid_size − 1)、这一步移动）乘 W_ih 的结果按（格子, 方向）一次算成表，逐步查表；
        单步 LSTM 按 nn.LSTM 的门顺序（输入、遗忘、候选、输出）直接算；
        抽方向用 Gumbel-max（logits 加 Gumbel 噪声后取最大，与按 softmax 概率抽样等价），噪声按 sample_seed 预先抽好，
        每次预测共用同一份，所以同样的历史总是给出同样的结果。
        """
        device = logits.device
        rows, illegal, moves = self._tables(grid_size, device)
        lstm, hidden = self.lstm, self.hidden_size
        input_gates = torch.addmm(lstm.bias_ih_l0 + lstm.bias_hh_l0, rows, lstm.weight_ih_l0.t())
        recurrent = lstm.weight_hh_l0.t()
        flat = report.expand(self.samples)
        h, c = h.expand(self.samples, -1), c.expand(self.samples, -1)
        step_logits = logits.expand(self.samples, -1)
        recorded, horizon = {0: flat}, max(steps)
        noise = self._gumbel_noise(horizon, device)
        for k in range(1, horizon + 1):
            choice = (step_logits.masked_fill(illegal[flat], -torch.inf) + noise[k - 1]).argmax(dim=-1)
            flat = flat + moves[choice]
            recorded[k] = flat
            if k < horizon:
                gates = torch.addmm(input_gates[flat * 4 + choice], h, recurrent)
                gate = gates.sigmoid()
                c = torch.addcmul(gate[:, hidden:2 * hidden] * c, gate[:, :hidden], gates[:, 2 * hidden:3 * hidden].tanh())
                h = gate[:, 3 * hidden:] * c.tanh()
                step_logits = self.head(h)
        return torch.stack([recorded[k] for k in steps])

    def _tables(self, grid_size: int, device):
        """抽样用的常量表：（格子, 方向）对应的网络输入 [4 × 格子数, 4]、各格越界方向 [格子数, 4]、四个方向的扁平位移。"""
        key = (grid_size, str(device))
        if key not in self._constant_tables:
            cells = np.indices((grid_size, grid_size)).reshape(2, -1).T
            rows = np.column_stack((np.repeat(cells / (grid_size - 1), 4, axis=0), np.tile(DIRECTIONS, (len(cells), 1))))
            self._constant_tables[key] = (
                torch.as_tensor(rows, dtype=torch.float32, device=device),
                torch.as_tensor(~legal_directions(cells, grid_size), device=device),
                torch.as_tensor(DIRECTIONS[:, 0] * grid_size + DIRECTIONS[:, 1], device=device),
            )
        return self._constant_tables[key]

    def _gumbel_noise(self, horizon: int, device) -> torch.Tensor:
        """各步各条路径四个方向的 Gumbel 噪声 [horizon, samples, 4]；同一 horizon 总是同一份。"""
        key = (horizon, str(device))
        if key not in self._noise:
            generator = torch.Generator().manual_seed(self.sample_seed)
            uniform = torch.rand((horizon, self.samples, 4), generator=generator).clamp_min(1e-12)
            self._noise[key] = (-torch.log(-torch.log(uniform))).to(device)
        return self._noise[key]

    def sample_positions(self, history, grid_size: int, steps) -> np.ndarray:
        """抽样路径在 steps 各步的位置 [len(steps), samples, 2]（整数格子）。"""
        cells = self._sample_cells(history, grid_size, tuple(steps))
        return np.stack(np.divmod(cells, grid_size), axis=-1)

    def predict_positions(self, history, grid_size: int, steps, move_every: int = 1) -> np.ndarray:
        """各步的点预测 [len(steps), 2]（抽样直方图上按 point 取格），steps 相对最后报告时刻；第 0 步就是报告位置。"""
        if move_every != 1:
            raise ValueError("方向概率 LSTM 只支持车辆每步移动一格")
        cells = self._sample_cells(history, grid_size, tuple(steps))
        counts = [np.bincount(row, minlength=grid_size * grid_size).reshape(grid_size, grid_size) for row in cells]
        return point_positions(counts, self.point, self.radius)

    def log_likelihood(self, config, seeds) -> float:
        """路线上每次实际移动在模型下的平均对数概率（精确值，越大越好）。"""
        device = next(self.parameters()).device
        return self._mean_log_likelihood(*(tensor.to(device) for tensor in route_examples(config, seeds)))

    @torch.no_grad()
    def _mean_log_likelihood(self, inputs, targets, legal) -> float:
        self.eval()
        log_p = masked_log_softmax(self(inputs)[0], legal)
        return float(log_p.gather(2, targets[..., None]).mean())

    @classmethod
    def fit(cls, config, groups, epochs: int = 40, seed: int = 0, *, learning_rate: float = 0.001,
            batch_size: int = 32, device="auto", hidden_size: int = 128, head_size: int = 64,
            samples: int = 2000) -> tuple["DirectionPredictor", dict]:
        """在训练路线上用交叉熵拟合方向概率，取验证路线上对数概率最好的一轮；验证 / 测试路线不参与拟合。"""
        cls._check_route(config)
        torch.manual_seed(seed)
        device = prediction_device(device)
        model = cls(hidden_size, head_size, samples).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_x, train_y, train_legal = (tensor.to(device) for tensor in route_examples(config, groups["train"]))
        validation_data = tuple(tensor.to(device) for tensor in route_examples(config, groups["validation"]))
        generator = torch.Generator().manual_seed(seed)
        best, best_epoch, best_state, history = -float("inf"), 0, None, []
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in torch.randperm(len(train_x), generator=generator).split(batch_size):
                batch = batch.to(device)
                log_p = masked_log_softmax(model(train_x[batch])[0], train_legal[batch])
                loss = nn.functional.nll_loss(log_p.reshape(-1, 4), train_y[batch].reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            validation = model._mean_log_likelihood(*validation_data)
            history.append(validation)
            if validation > best:
                best, best_epoch = validation, epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            print(f"epoch={epoch} validation log-likelihood per move={validation:.4f}", flush=True)
        model.load_state_dict(best_state)
        model.eval()
        metrics = {
            "model_type": cls.model_type, "best_validation_log_likelihood": best, "best_epoch": best_epoch,
            "validation_log_likelihood_by_epoch": history, "seed_groups": groups, "epochs": epochs,
            "learning_rate": learning_rate, "batch_size": batch_size, "hidden_size": hidden_size,
            "head_size": head_size, "samples": samples, "seed": seed, "device": str(device),
        }
        return model, metrics

    def export_state(self) -> dict:
        return {
            "model_type": self.model_type, "hidden_size": self.hidden_size, "head_size": self.head_size,
            "samples": self.samples, "sample_seed": self.sample_seed, "point": self.point, "radius": self.radius,
            "state_dict": {name: value.detach().cpu().clone() for name, value in self.state_dict().items()},
        }

    @classmethod
    def from_state(cls, saved: dict, device="auto") -> "DirectionPredictor":
        model = cls(saved["hidden_size"], saved["head_size"], saved["samples"], saved["sample_seed"],
                    saved.get("point", "mode"), saved.get("radius", 2.0))
        model.load_state_dict(saved["state_dict"])
        model.to(prediction_device(device))
        model.eval()
        return model


class _SamplingGraph:
    """把一种（地图大小, 时距）的 _rollout 录成 CUDA graph；每次预测只把起点写进固定的输入张量再重放。

    graph 记的是参数张量的地址，参数换了地址（例如 .to() 搬设备）就要重录，所以缓存键里带参数地址。
    """

    def __init__(self, model: DirectionPredictor, grid_size: int, steps, device):
        self.report = torch.zeros((), dtype=torch.int64, device=device)
        self.logits = torch.zeros(4, device=device)
        self.h = torch.zeros(model.hidden_size, device=device)
        self.c = torch.zeros(model.hidden_size, device=device)
        arguments = (grid_size, steps, self.report, self.logits, self.h, self.c)
        # 录制前先在旁路流上跑几次，让常量表、噪声和显存分配都就绪（录制期间不能有主机到设备的拷贝）。
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                model._rollout(*arguments)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model._rollout(*arguments)

    def run(self, report: int, logits, h, c) -> np.ndarray:
        self.report.fill_(report)
        self.logits.copy_(logits)
        self.h.copy_(h)
        self.c.copy_(c)
        self.graph.replay()
        return self.output.cpu().numpy()
