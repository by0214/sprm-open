from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from sprm.models import AutoModelForCausalLMWithDualValueHead


def build_sprm_input_text(question: str, steps: Sequence[str], st_token: str = "<ST_END>") -> str:
    question = (question or "").strip()
    parts = [question + st_token]
    for idx, step in enumerate(steps, start=1):
        text = (step or "").strip()
        if not text.lower().startswith("step"):
            text = f"Step {idx}: {text}"
        parts.append(text + st_token)
    return "".join(parts)


def combine_step_scores(
    marginal_scores: torch.Tensor,
    cti_scores: torch.Tensor,
    *,
    combine: str = "sprm",
    marginal_weight: float = 1.0,
    cti_weight: float = 1.0,
) -> torch.Tensor:
    mode = combine.strip().lower()
    if mode == "marginal":
        return marginal_scores
    if mode == "cti":
        return cti_scores
    if mode == "mul":
        return marginal_scores * cti_scores
    if mode == "sprm":
        return (
            float(marginal_weight) * (marginal_scores - 0.5)
            + float(cti_weight) * (cti_scores - 0.5)
            + 0.5
        )
    raise ValueError(f"Unknown combine mode: {combine}")


@dataclass(frozen=True)
class StepScores:
    marginal: list[float]
    cti: list[float]
    combined: list[float]


class SPRMScorer:
    """Inference helper that extracts SPRM scores at `<ST_END>` boundaries."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        st_token: str = "<ST_END>",
        device: str | torch.device = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.st_token = st_token
        self.device = self._resolve_device(device)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        st_token: str = "<ST_END>",
        device: str | torch.device = "auto",
        **model_kwargs: Any,
    ) -> "SPRMScorer":
        from transformers import AutoTokenizer

        subfolder = model_kwargs.get("subfolder")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=model_kwargs.pop("trust_remote_code", True),
            subfolder=subfolder,
            token=model_kwargs.get("token", model_kwargs.get("use_auth_token", None)),
        )
        model = AutoModelForCausalLMWithDualValueHead.from_pretrained(
            model_id_or_path,
            **model_kwargs,
        )
        return cls(model, tokenizer, st_token=st_token, device=device)

    def score_steps(
        self,
        *,
        question: str,
        steps: Sequence[str],
        combine: str = "sprm",
        marginal_weight: float = 1.0,
        cti_weight: float = 1.0,
        apply_sigmoid: bool = True,
        max_length: int | None = None,
    ) -> StepScores:
        if not steps:
            return StepScores(marginal=[], cti=[], combined=[])

        text = build_sprm_input_text(question, steps, st_token=self.st_token)
        boundary_positions = self._boundary_positions(question, steps)
        if len(boundary_positions) != len(steps) + 1:
            raise RuntimeError("Failed to align all SPRM boundary positions.")

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = getattr(tokenized, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        marginal_values, cti_values = self._extract_value_heads(outputs)

        # Drop the first boundary, which corresponds to the question boundary.
        step_positions = boundary_positions[1:]
        max_position = int(marginal_values.shape[1]) - 1
        if any(pos > max_position for pos in step_positions):
            raise RuntimeError("SPRM boundary position exceeds model output length.")

        positions = torch.tensor(step_positions, dtype=torch.long, device=marginal_values.device)
        marginal = marginal_values[0, positions]
        cti = cti_values[0, positions]
        if apply_sigmoid:
            marginal = torch.sigmoid(marginal)
            cti = torch.sigmoid(cti)
        combined = combine_step_scores(
            marginal,
            cti,
            combine=combine,
            marginal_weight=marginal_weight,
            cti_weight=cti_weight,
        )
        return StepScores(
            marginal=[float(x) for x in marginal.detach().cpu().tolist()],
            cti=[float(x) for x in cti.detach().cpu().tolist()],
            combined=[float(x) for x in combined.detach().cpu().tolist()],
        )

    def _boundary_positions(self, question: str, steps: Sequence[str]) -> list[int]:
        prefixes: list[str] = [(question or "").strip() + self.st_token]
        current = prefixes[0]
        for idx, step in enumerate(steps, start=1):
            text = (step or "").strip()
            if not text.lower().startswith("step"):
                text = f"Step {idx}: {text}"
            current = current + text + self.st_token
            prefixes.append(current)

        positions: list[int] = []
        for prefix in prefixes:
            tokenized = self.tokenizer(prefix, return_tensors="pt")
            positions.append(int(tokenized.input_ids.shape[1] - 1))
        return positions

    @staticmethod
    def _extract_value_heads(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 4:
            return outputs[2], outputs[3]
        marginal = getattr(outputs, "marginal_values", None)
        cti = getattr(outputs, "cti_values", None)
        if marginal is not None and cti is not None:
            return marginal, cti
        raise RuntimeError("Model output does not contain SPRM dual value heads.")

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
