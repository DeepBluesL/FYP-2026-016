from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .prediction_data import iter_prediction_routes


def prediction_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class RouteDataset(Dataset):
    """只保存种子，每次索引生成一条完整路线。"""

    def __init__(self, config, seeds):
        self.config = config
        self.seeds = list(seeds)

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, index):
        sample = next(iter_prediction_routes(self.config, [self.seeds[index]]))
        return torch.from_numpy(sample["features"])


class SequencePredictor(nn.Module):
    def __init__(
            self,
            model_type="lstm",
            hidden_size=256,
            head_size=128,
            num_layers=1,
            max_steps=150,
    ):
        super().__init__()

        self.model_type = model_type
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_layers = num_layers
        self.max_steps = max_steps

        recurrent_layer = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[model_type]

        self.encoder = recurrent_layer(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, head_size), nn.ReLU(), nn.Linear(head_size, 2),
        )

    def forward(self, features, state=None):
        hidden_sequence, state = self.encoder(features, state)
        position = self.position_head(hidden_sequence)
        return position, state

    @torch.no_grad()
    def predict_positions(self, history, grid_size, steps, move_every=1):
        self.eval()
        scale = grid_size - 1
        device = next(self.parameters()).device
        features = torch.tensor([
            [x / scale, y / scale, time / self.max_steps]
            for time, (x, y) in history
        ], dtype=torch.float32, device=device).unsqueeze(0)

        predicted, state = self(features)

        forecasts = [np.asarray(history[-1][1], dtype=np.float32)]

        report_time = history[-1][0]
        next_position = predicted[:, -1:, :]

        for offset in range(1, max(steps) + 1):
            next_position = next_position.clamp(0.0, 1.0)
            forecasts.append(next_position[0, 0].cpu().numpy() * scale)

            if offset < max(steps):
                next_time = next_position.new_full((1, 1, 1), (report_time + offset) / self.max_steps)
                next_input = torch.cat((next_position, next_time), dim=-1)
                next_position, state = self(next_input, state)

        return np.stack([forecasts[h] for h in steps])

    # 保存与恢复
    def export_state(self):
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "num_layers": self.num_layers,
            "max_steps": self.max_steps,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_state(cls, saved, device="auto"):
        model = cls(
            model_type=saved["model_type"],
            hidden_size=saved["hidden_size"],
            head_size=saved["head_size"],
            num_layers=saved["num_layers"],
            max_steps=saved["max_steps"],
        )
        model.load_state_dict(saved["state_dict"])
        model.to(prediction_device(device))
        model.eval()
        return model


def run_epoch(model, config, seeds, optimizer=None, *, batch_size=32, generator=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    route_count = 0

    device = next(model.parameters()).device
    loader = DataLoader(RouteDataset(config, seeds), batch_size=batch_size,
                        shuffle=training, generator=generator, drop_last=False,
                        pin_memory=device.type == "cuda")
    # 每个 batch 包含多条完整路线，路线之间不传递隐藏状态。
    with torch.set_grad_enabled(training):
        for features in loader:
            features = features.to(device, non_blocking=device.type == "cuda")

            predicted, _ = model(features)

            loss = nn.functional.mse_loss(predicted[:, :-1, :], features[:, 1:, :2])

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * features.size(0)
            route_count += features.size(0)

        return total_loss / route_count


def fit_sequence_predictor(config, groups, model_type="lstm", epochs=20, seed=42, *,
                           learning_rate=0.001, batch_size=32, device="auto",
                           hidden_size=256, head_size=128, num_layers=1,
                           tcn_blocks=4, tcn_kernel_size=3):
    torch.manual_seed(seed)

    if model_type == "tcn":
        from .tcn_prediction import TCNPredictor
        model = TCNPredictor(hidden_size=hidden_size, head_size=head_size,
                             num_blocks=tcn_blocks, kernel_size=tcn_kernel_size,
                             max_steps=config.max_steps)
    else:
        model = SequencePredictor(model_type=model_type, max_steps=config.max_steps,
                                  hidden_size=hidden_size, head_size=head_size,
                                  num_layers=num_layers)
    model = model.to(prediction_device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = run_epoch(model, config, groups["train"], optimizer,
                               batch_size=batch_size, generator=generator)
        validation_loss = run_epoch(model, config, groups["validation"], batch_size=batch_size)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())

        print(
            f"epoch={epoch + 1} "
            f"train={train_loss:.6f} "
            f"validation={validation_loss:.6f}"
        )

    model.load_state_dict(best_state)
    model.eval()

    metrics = {
        "model_type": model_type,
        "best_validation_mse": best_loss,
        "seed_groups": groups,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "device": str(next(model.parameters()).device),
        "hidden_size": hidden_size,
        "head_size": head_size,
        "seed": seed,
        "epochs": epochs,
    }
    if model_type == "tcn":
        metrics.update(tcn_blocks=tcn_blocks, tcn_kernel_size=tcn_kernel_size,
                       receptive_field=model.receptive_field)
    else:
        metrics["num_layers"] = num_layers
    return model, metrics
