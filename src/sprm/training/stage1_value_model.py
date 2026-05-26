from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sprm.data import HiddenStateBinReader, HiddenStateRef
from sprm.models import TrajectoryBiLSTMValueModel, save_trajectory_value_head

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage1Sample:
    hidden_states: torch.Tensor
    final_reward: torch.Tensor


class TrajectoryValueDataset(Dataset):
    def __init__(self, samples: list[Stage1Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Stage1Sample:
        return self.samples[index]


def as_vector(value: Any, expected_hidden_size: int | None) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim != 1:
        array = array.reshape(-1)
    if expected_hidden_size is not None and array.shape[0] != int(expected_hidden_size):
        return None
    return np.asarray(array, dtype=np.float32)


def load_step_hidden_state(
    step: dict[str, Any],
    *,
    reader: HiddenStateBinReader,
    base_dir: Path,
    hidden_size: int | None,
) -> np.ndarray | None:
    if "hidden_state" in step:
        return as_vector(step["hidden_state"], hidden_size)

    if isinstance(step.get("hidden_state_ref"), dict):
        ref = HiddenStateRef.from_dict(step["hidden_state_ref"])
        return as_vector(reader.read_ref(ref, base_dir=base_dir), hidden_size)

    # Multi-layer refs are supported by concatenating all refs in their stored order.
    refs = step.get("hidden_state_refs")
    if isinstance(refs, list) and refs:
        parts: list[np.ndarray] = []
        for ref_obj in refs:
            if not isinstance(ref_obj, dict):
                return None
            ref = HiddenStateRef.from_dict(ref_obj)
            parts.append(as_vector(reader.read_ref(ref, base_dir=base_dir), None))
        concat = np.concatenate(parts, axis=0)
        return as_vector(concat, hidden_size)

    return None


def iter_stage1_samples(
    data_file: str | Path,
    *,
    hidden_size: int | None = None,
    max_samples: int = 0,
) -> list[Stage1Sample]:
    path = Path(data_file)
    reader = HiddenStateBinReader()
    samples: list[Stage1Sample] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skip invalid json at %s:%d", path, line_no)
                continue

            steps = record.get("steps")
            final_reward = record.get("final_reward")
            if not isinstance(steps, list) or not steps or final_reward is None:
                continue

            vectors: list[np.ndarray] = []
            for step in steps:
                if not isinstance(step, dict):
                    vectors = []
                    break
                vector = load_step_hidden_state(
                    step,
                    reader=reader,
                    base_dir=path.parent,
                    hidden_size=hidden_size,
                )
                if vector is None:
                    vectors = []
                    break
                vectors.append(vector)

            if not vectors:
                continue
            stacked = torch.tensor(np.stack(vectors, axis=0), dtype=torch.float32)
            samples.append(
                Stage1Sample(
                    hidden_states=stacked,
                    final_reward=torch.tensor(float(final_reward), dtype=torch.float32),
                )
            )
            if max_samples > 0 and len(samples) >= int(max_samples):
                break

    return samples


def collate_trajectory_values(samples: list[Stage1Sample]) -> dict[str, torch.Tensor]:
    max_steps = max(sample.hidden_states.shape[0] for sample in samples)
    hidden_size = samples[0].hidden_states.shape[1]
    hidden_states = torch.zeros((len(samples), max_steps, hidden_size), dtype=torch.float32)
    lengths = torch.zeros((len(samples),), dtype=torch.long)
    labels = torch.zeros((len(samples),), dtype=torch.float32)
    for idx, sample in enumerate(samples):
        steps = sample.hidden_states.shape[0]
        hidden_states[idx, :steps] = sample.hidden_states
        lengths[idx] = int(steps)
        labels[idx] = sample.final_reward
    return {"hidden_states": hidden_states, "lengths": lengths, "labels": labels}


def split_samples(
    samples: list[Stage1Sample],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[Stage1Sample], list[Stage1Sample]]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if validation_ratio <= 0 or len(shuffled) < 2:
        return shuffled, []
    val_size = max(1, int(round(len(shuffled) * float(validation_ratio))))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def compute_binary_metrics(values: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    labels = labels.detach().float().cpu()
    preds = (values >= 0.5).float()
    targets = (labels >= 0.5).float()
    return {
        "loss_mse": float(torch.mean((values - labels) ** 2).item()),
        "accuracy": float(torch.mean((preds == targets).float()).item()),
        "mean_value": float(values.mean().item()),
        "mean_label": float(labels.mean().item()),
    }


def run_epoch(
    *,
    model: TrajectoryBiLSTMValueModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    all_values: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in dataloader:
        hidden_states = batch["hidden_states"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            output = model(hidden_states=hidden_states, lengths=lengths, labels=labels)
            loss = output["loss"]
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        all_values.append(output["values"].detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total_items == 0:
        return {}
    metrics = compute_binary_metrics(torch.cat(all_values), torch.cat(all_labels))
    metrics["loss"] = total_loss / total_items
    return metrics


def train_stage1(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    samples = iter_stage1_samples(
        args.data_file,
        hidden_size=args.hidden_size if args.hidden_size > 0 else None,
        max_samples=args.max_samples,
    )
    if not samples:
        raise ValueError(f"No valid Stage1 samples found in {args.data_file}")

    inferred_hidden_size = int(samples[0].hidden_states.shape[1])
    train_samples, val_samples = split_samples(
        samples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    logger.info(
        "stage1 samples: total=%d train=%d val=%d hidden_size=%d",
        len(samples),
        len(train_samples),
        len(val_samples),
        inferred_hidden_size,
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TrajectoryBiLSTMValueModel(
        input_size=inferred_hidden_size,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        loss_type=args.loss_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TrajectoryValueDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectory_values,
    )
    val_loader = (
        DataLoader(
            TrajectoryValueDataset(val_samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_trajectory_values,
        )
        if val_samples
        else None
    )

    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = int(args.early_stop_patience)
    for epoch in range(1, int(args.num_train_epochs) + 1):
        train_metrics = run_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device)
        if val_loader is not None:
            val_metrics = run_epoch(model=model, dataloader=val_loader, optimizer=None, device=device)
            monitor = float(val_metrics.get("loss", train_metrics["loss"]))
            logger.info("epoch=%d train=%s val=%s", epoch, train_metrics, val_metrics)
        else:
            monitor = float(train_metrics["loss"])
            logger.info("epoch=%d train=%s", epoch, train_metrics)

        if monitor < best_metric:
            best_metric = monitor
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(args.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("early stop at epoch=%d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    stage1_dir = Path(args.output_dir) / "stage1"
    full_model_dir = stage1_dir / "full_model"
    model.save_pretrained(full_model_dir)
    head_path = stage1_dir / "trajectory_value_head.bin"
    save_trajectory_value_head(model, head_path)
    (stage1_dir / "metrics.json").write_text(
        json.dumps({"best_loss": best_metric, "num_samples": len(samples)}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("stage1 saved full model to %s", full_model_dir)
    logger.info("stage1 saved value head to %s", head_path)
    return stage1_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage1: train SPRM's outcome-supervised trajectory value model."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--hidden-size", type=int, default=0, help="0 means infer from data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--lstm-hidden-dim", type=int, default=512)
    parser.add_argument("--lstm-num-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=["bce", "mse"], default="bce")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_stage1(args)


if __name__ == "__main__":
    main()
