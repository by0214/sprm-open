from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sprm.data import HiddenStateBinReader
from sprm.models import TrajectoryBiLSTMValueModel
from sprm.training.stage1_value_model import load_step_hidden_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage2Trajectory:
    record: dict[str, Any]
    hidden_states: torch.Tensor
    final_reward: float
    trajectory_id: int


def load_stage2_trajectories(
    data_file: str | Path,
    *,
    hidden_size: int | None = None,
    max_samples: int = 0,
) -> list[Stage2Trajectory]:
    path = Path(data_file)
    reader = HiddenStateBinReader()
    trajectories: list[Stage2Trajectory] = []

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
            trajectories.append(
                Stage2Trajectory(
                    record=record,
                    hidden_states=torch.tensor(np.stack(vectors, axis=0), dtype=torch.float32),
                    final_reward=float(final_reward),
                    trajectory_id=len(trajectories),
                )
            )
            if max_samples > 0 and len(trajectories) >= int(max_samples):
                break

    return trajectories


def trajectory_score(
    model: TrajectoryBiLSTMValueModel,
    hidden_states: torch.Tensor,
    *,
    target: str = "prob",
) -> torch.Tensor:
    output = model(
        hidden_states=hidden_states.unsqueeze(0),
        lengths=torch.tensor([hidden_states.shape[0]], device=hidden_states.device),
    )
    if target == "logit":
        return output["logits"][0]
    if target == "prob":
        return output["values"][0]
    raise ValueError("target must be 'prob' or 'logit'")


def step_value_integration(
    model: TrajectoryBiLSTMValueModel,
    hidden_states: torch.Tensor,
    *,
    steps: int = 64,
    target: str = "prob",
) -> torch.Tensor:
    """Compute per-step SVI credit with an Aumann-Shapley path integral."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    baseline = torch.zeros_like(hidden_states)
    total_grad = torch.zeros_like(hidden_states)

    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=hidden_states.device):
        interpolated = (baseline + alpha * (hidden_states - baseline)).detach()
        interpolated.requires_grad_(True)
        score = trajectory_score(model, interpolated, target=target)
        grad = torch.autograd.grad(score, interpolated, retain_graph=False)[0]
        total_grad = total_grad + grad.detach()

    attribution = (hidden_states - baseline) * (total_grad / float(steps))
    return attribution.sum(dim=1)


def compute_causal_consistency(
    model: TrajectoryBiLSTMValueModel,
    hidden_states: torch.Tensor,
    *,
    target: str = "prob",
) -> torch.Tensor:
    """Value drop after zero-ablating each step representation."""

    with torch.no_grad():
        origin = trajectory_score(model, hidden_states, target=target)
        values: list[torch.Tensor] = []
        for idx in range(hidden_states.shape[0]):
            ablated = hidden_states.clone()
            ablated[idx] = 0
            values.append(origin - trajectory_score(model, ablated, target=target))
    return torch.stack(values)


def compute_refined_causal_consistency(
    model: TrajectoryBiLSTMValueModel,
    hidden_states: torch.Tensor,
    *,
    base_causal: torch.Tensor,
    target: str = "prob",
    window_size: int = 2,
    beta: float = 0.5,
) -> torch.Tensor:
    """Add a local pairwise interaction correction to causal consistency."""

    if window_size <= 0 or beta == 0:
        return base_causal

    with torch.no_grad():
        origin = trajectory_score(model, hidden_states, target=target)
        single_ablations: list[torch.Tensor] = []
        for idx in range(hidden_states.shape[0]):
            ablated = hidden_states.clone()
            ablated[idx] = 0
            single_ablations.append(trajectory_score(model, ablated, target=target))

        refined = base_causal.clone()
        for idx in range(hidden_states.shape[0]):
            correction = torch.tensor(0.0, device=hidden_states.device)
            weight_sum = 0.0
            for prev in range(max(0, idx - window_size), idx):
                double_ablated = hidden_states.clone()
                double_ablated[idx] = 0
                double_ablated[prev] = 0
                double_value = trajectory_score(model, double_ablated, target=target)
                interaction = origin - single_ablations[idx] - single_ablations[prev] + double_value
                weight = 1.0 / float(idx - prev)
                correction = correction + float(weight) * interaction
                weight_sum += weight
            if weight_sum > 0:
                refined[idx] = refined[idx] + float(beta) * correction / float(weight_sum)
    return refined


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def compute_cti_scores(
    trajectories: list[Stage2Trajectory],
    *,
    topk: int = 50,
    search_k: int = 400,
) -> tuple[list[list[float | None]], float]:
    """Compute cross-trajectory inherent scores with cosine neighbors."""

    rewards = np.asarray([traj.final_reward for traj in trajectories], dtype=np.float32)
    base_rate = float(rewards.mean()) if rewards.size else 0.0
    if not trajectories:
        return [], base_rate

    vectors: list[np.ndarray] = []
    traj_ids: list[int] = []
    step_offsets: list[tuple[int, int]] = []
    for traj_idx, traj in enumerate(trajectories):
        start = len(vectors)
        arr = traj.hidden_states.detach().cpu().numpy().astype(np.float32)
        vectors.extend(arr)
        traj_ids.extend([traj_idx] * arr.shape[0])
        step_offsets.append((start, len(vectors)))

    matrix = np.stack(vectors, axis=0)
    normalized = l2_normalize(matrix)
    sims = normalized @ normalized.T
    traj_ids_arr = np.asarray(traj_ids, dtype=np.int64)
    out: list[list[float | None]] = []
    effective_search_k = max(int(search_k), int(topk))

    for traj_idx, (start, end) in enumerate(step_offsets):
        traj_scores: list[float | None] = []
        for global_step in range(start, end):
            order = np.argsort(-sims[global_step])[: effective_search_k + (end - start)]
            selected: list[int] = []
            selected_sims: list[float] = []
            for candidate in order:
                if int(traj_ids_arr[candidate]) == traj_idx:
                    continue
                selected.append(int(candidate))
                selected_sims.append(float(sims[global_step, candidate]))
                if len(selected) >= int(topk):
                    break
            if not selected:
                traj_scores.append(None)
                continue
            weights = np.maximum(np.asarray(selected_sims, dtype=np.float32), 0.0)
            if float(weights.sum()) <= 1e-12:
                weights = np.ones_like(weights, dtype=np.float32)
            neighbor_rewards = rewards[traj_ids_arr[np.asarray(selected, dtype=np.int64)]]
            neighborhood_rate = float((weights * neighbor_rewards).sum() / weights.sum())
            traj_scores.append(neighborhood_rate - base_rate)
        out.append(traj_scores)
    return out, base_rate


def add_stage2_labels(
    trajectory: Stage2Trajectory,
    *,
    model: TrajectoryBiLSTMValueModel,
    target: str,
    svi_steps: int,
    enable_refined_causal_consistency: bool,
    pairwise_window_size: int,
    pairwise_beta: float,
    cti_score: list[float | None] | None,
    cti_base_rate: float | None,
    device: torch.device,
) -> dict[str, Any]:
    hidden_states = trajectory.hidden_states.to(device)
    model = model.to(device)
    model.eval()

    marginal = step_value_integration(model, hidden_states, steps=svi_steps, target=target)
    causal = compute_causal_consistency(model, hidden_states, target=target)
    with torch.no_grad():
        value = trajectory_score(model, hidden_states, target=target)

    record = dict(trajectory.record)
    record["trajectory_value"] = float(value.detach().cpu().item())
    record["marginal_contribution"] = [float(x) for x in marginal.detach().cpu().tolist()]
    record["causal_consistency"] = [float(x) for x in causal.detach().cpu().tolist()]

    if enable_refined_causal_consistency:
        refined = compute_refined_causal_consistency(
            model,
            hidden_states,
            base_causal=causal,
            target=target,
            window_size=pairwise_window_size,
            beta=pairwise_beta,
        )
        record["refined_causal_consistency"] = [float(x) for x in refined.detach().cpu().tolist()]
        record["pairwise_beta"] = float(pairwise_beta)

    if cti_score is not None:
        record["cti_score"] = cti_score
        record["cti_base_rate"] = cti_base_rate

    return record


def construct_stage2_labels(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TrajectoryBiLSTMValueModel.from_pretrained(args.stage1_model_dir, map_location=device)
    model.to(device)
    model.eval()
    hidden_size = int(model.config["input_size"])

    trajectories = load_stage2_trajectories(
        args.data_file,
        hidden_size=hidden_size,
        max_samples=args.max_samples,
    )
    if not trajectories:
        raise ValueError(f"No valid trajectories found in {args.data_file}")

    cti_scores: list[list[float | None]] | None = None
    cti_base_rate: float | None = None
    if args.enable_cti:
        cti_scores, cti_base_rate = compute_cti_scores(
            trajectories,
            topk=args.cti_topk,
            search_k=args.cti_search_k,
        )

    output_dir = Path(args.output_dir) / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(args.data_file).stem
    output_path = output_dir / f"{input_stem}-sprm-labels.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, trajectory in enumerate(trajectories):
            record = add_stage2_labels(
                trajectory,
                model=model,
                target=args.svi_target,
                svi_steps=args.svi_steps,
                enable_refined_causal_consistency=args.enable_refined_causal_consistency,
                pairwise_window_size=args.pairwise_window_size,
                pairwise_beta=args.pairwise_beta,
                cti_score=(cti_scores[idx] if cti_scores is not None else None),
                cti_base_rate=cti_base_rate,
                device=device,
            )
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    logger.info("stage2 wrote SPRM labels to %s", output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage2: construct SPRM labels from a trained trajectory value model."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--stage1-model-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--svi-steps", type=int, default=64)
    parser.add_argument("--svi-target", choices=["prob", "logit"], default="prob")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument(
        "--enable-refined-causal-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pairwise-window-size", type=int, default=2)
    parser.add_argument("--pairwise-beta", type=float, default=0.5)

    parser.add_argument("--enable-cti", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cti-topk", type=int, default=50)
    parser.add_argument("--cti-search-k", type=int, default=400)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    construct_stage2_labels(args)


if __name__ == "__main__":
    main()
