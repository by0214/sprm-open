from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TrajectoryBiLSTMValueModel(nn.Module):
    """Outcome-supervised trajectory value function used by SPRM Stage1."""

    def __init__(
        self,
        *,
        input_size: int,
        lstm_hidden_dim: int = 512,
        lstm_num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        loss_type: str = "bce",
    ):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if loss_type not in {"bce", "mse"}:
            raise ValueError("loss_type must be 'bce' or 'mse'")

        self.config = {
            "input_size": int(input_size),
            "lstm_hidden_dim": int(lstm_hidden_dim),
            "lstm_num_layers": int(lstm_num_layers),
            "bidirectional": bool(bidirectional),
            "dropout": float(dropout),
            "loss_type": str(loss_type),
        }
        self.input_norm = nn.LayerNorm(int(input_size))
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(lstm_hidden_dim),
            num_layers=int(lstm_num_layers),
            batch_first=True,
            dropout=float(dropout) if int(lstm_num_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        feature_dim = int(lstm_hidden_dim) * (2 if bidirectional else 1)
        self.value_head = nn.Linear(feature_dim, 1)
        self.loss_type = str(loss_type)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, steps, hidden]")

        x = self.input_norm(hidden_states.float())
        if lengths is None:
            lengths = torch.full(
                (x.shape[0],),
                x.shape[1],
                dtype=torch.long,
                device=x.device,
            )
        lengths = lengths.to(torch.long).clamp(min=1)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.config["bidirectional"]:
            trajectory_feature = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            trajectory_feature = h_n[-1]

        logits = self.value_head(trajectory_feature).squeeze(-1)
        values = torch.sigmoid(logits)
        output = {"logits": logits, "values": values}
        if labels is not None:
            labels = labels.float().view(-1)
            if self.loss_type == "bce":
                output["loss"] = self.bce_loss(logits, labels)
            else:
                output["loss"] = self.mse_loss(values, labels)
        return output

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "trajectory_value_model.pt")
        (output_dir / "trajectory_value_config.json").write_text(
            json.dumps(self.config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, map_location: str | torch.device = "cpu"):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "trajectory_value_config.json").read_text(encoding="utf-8"))
        model = cls(**config)
        state = torch.load(model_dir / "trajectory_value_model.pt", map_location=map_location)
        model.load_state_dict(state)
        return model


def save_trajectory_value_head(model: TrajectoryBiLSTMValueModel, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "value_head.weight": model.value_head.weight.detach().cpu(),
            "value_head.bias": model.value_head.bias.detach().cpu(),
        },
        output_path,
    )
