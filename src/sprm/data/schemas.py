from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundaryStep:
    """One question or reasoning-step boundary used by SPRM."""

    prefix_text: str
    hidden_state_ref: dict[str, Any]
    role: str = "step"

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_text": self.prefix_text,
            "hidden_state_ref": self.hidden_state_ref,
            "role": self.role,
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    """Output schema produced by `sprm.data.collect_hidden_states`."""

    final_reward: float
    steps: list[BoundaryStep]
    question: str = ""
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question": self.question,
            "final_reward": float(self.final_reward),
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.sample_id is not None:
            out["id"] = self.sample_id
        if self.metadata:
            out["metadata"] = self.metadata
        return out


def require_trajectory_record(obj: dict[str, Any]) -> None:
    """Validate the minimal schema consumed by Stage1 and Stage2."""

    if "final_reward" not in obj:
        raise ValueError("trajectory record missing final_reward")
    try:
        float(obj["final_reward"])
    except Exception as exc:
        raise ValueError("final_reward must be numeric") from exc

    steps = obj.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory record must contain non-empty steps")

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{idx}] must be an object")
        if not isinstance(step.get("prefix_text"), str) or not step["prefix_text"]:
            raise ValueError(f"steps[{idx}].prefix_text must be a non-empty string")
        if not isinstance(step.get("hidden_state_ref"), dict):
            raise ValueError(f"steps[{idx}].hidden_state_ref must be an object")
