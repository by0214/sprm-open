from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sprm.models import AutoModelForCausalLMWithDualValueHead

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage3Sample:
    text: str
    marginal_labels: list[float]
    cti_labels: list[float]
    marginal_mask: list[float]
    cti_mask: list[float]


def sigmoid_scalar(value: float, temperature: float = 1.0) -> float:
    temperature = max(float(temperature), 1e-6)
    x = max(min(float(value) / temperature, 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-x)))


def transform_signal(value: float, *, mode: str, threshold: float, temperature: float) -> float:
    if mode == "binary":
        return 1.0 if float(value) > float(threshold) else 0.0
    if mode == "prob":
        return sigmoid_scalar(float(value), temperature=temperature)
    if mode == "regression":
        return float(value)
    raise ValueError("label mode must be binary, prob, or regression")


def finite_float_list(raw: Any, expected_len: int) -> list[float | None] | None:
    if not isinstance(raw, list):
        return None
    out: list[float | None] = []
    for idx in range(expected_len):
        if idx >= len(raw) or raw[idx] is None:
            out.append(None)
            continue
        try:
            value = float(raw[idx])
        except Exception:
            out.append(None)
            continue
        out.append(value if math.isfinite(value) else None)
    return out


def recover_boundary_text(prefixes: Sequence[str], st_token: str) -> str:
    if not prefixes:
        return ""
    text = str(prefixes[0]).strip() + st_token
    previous = str(prefixes[0])
    for prefix in prefixes[1:]:
        current = str(prefix)
        if current.startswith(previous):
            segment = current[len(previous) :].strip()
        else:
            segment = current.strip()
        if segment:
            text += segment + st_token
        else:
            text += st_token
        previous = current
    return text


def build_stage3_sample(
    raw: dict[str, Any],
    *,
    st_token: str = "<ST_END>",
    label_mode: str = "binary",
    marginal_threshold: float = 0.0,
    cti_threshold: float = 0.0,
    label_temperature: float = 1.0,
    causal_field: str = "refined_causal_consistency",
) -> Stage3Sample | None:
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    prefixes = [step.get("prefix_text", "") for step in steps if isinstance(step, dict)]
    prefixes = [str(prefix) for prefix in prefixes if str(prefix).strip()]
    if not prefixes:
        return None
    n = len(prefixes)

    marginal = finite_float_list(raw.get("marginal_contribution"), n)
    if marginal is None:
        return None
    cti = finite_float_list(raw.get("cti_score"), n)
    causal = finite_float_list(raw.get(causal_field), n)
    if causal is None:
        causal = finite_float_list(raw.get("causal_consistency"), n)

    marginal_labels: list[float] = []
    marginal_mask: list[float] = []
    for idx, value in enumerate(marginal):
        if value is None:
            marginal_labels.append(0.0)
            marginal_mask.append(0.0)
            continue
        if causal is not None and causal[idx] is not None:
            use_label = float(value) * float(causal[idx]) > 0.0
        else:
            use_label = True
        marginal_labels.append(
            transform_signal(
                value,
                mode=label_mode,
                threshold=marginal_threshold,
                temperature=label_temperature,
            )
        )
        marginal_mask.append(1.0 if use_label else 0.0)

    cti_labels: list[float] = []
    cti_mask: list[float] = []
    for value in cti or [None] * n:
        if value is None:
            cti_labels.append(0.0)
            cti_mask.append(0.0)
        else:
            cti_labels.append(
                transform_signal(
                    value,
                    mode=label_mode,
                    threshold=cti_threshold,
                    temperature=label_temperature,
                )
            )
            cti_mask.append(1.0)

    return Stage3Sample(
        text=recover_boundary_text(prefixes, st_token=st_token),
        marginal_labels=marginal_labels,
        cti_labels=cti_labels,
        marginal_mask=marginal_mask,
        cti_mask=cti_mask,
    )


class Stage3Dataset(Dataset):
    def __init__(
        self,
        label_path: str | Path,
        *,
        st_token: str = "<ST_END>",
        label_mode: str = "binary",
        marginal_threshold: float = 0.0,
        cti_threshold: float = 0.0,
        label_temperature: float = 1.0,
        causal_field: str = "refined_causal_consistency",
        max_samples: int = 0,
    ):
        self.samples: list[Stage3Sample] = []
        with Path(label_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                sample = build_stage3_sample(
                    raw,
                    st_token=st_token,
                    label_mode=label_mode,
                    marginal_threshold=marginal_threshold,
                    cti_threshold=cti_threshold,
                    label_temperature=label_temperature,
                    causal_field=causal_field,
                )
                if sample is not None:
                    self.samples.append(sample)
                if max_samples > 0 and len(self.samples) >= int(max_samples):
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Stage3Sample:
        return self.samples[index]


class DataCollatorForSPRM:
    def __init__(self, tokenizer, *, st_token: str = "<ST_END>", max_length: int = 2048):
        self.tokenizer = tokenizer
        self.st_token = st_token
        self.max_length = int(max_length)
        self.st_token_id = self._resolve_st_token_id(tokenizer, st_token)

    def __call__(self, samples: list[Stage3Sample]) -> dict[str, torch.Tensor]:
        texts = [sample.text for sample in samples]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized.input_ids
        attention_mask = getattr(tokenized, "attention_mask", torch.ones_like(input_ids))
        targets_marginal = torch.zeros_like(input_ids, dtype=torch.float32)
        targets_cti = torch.zeros_like(input_ids, dtype=torch.float32)
        mask_marginal = torch.zeros_like(input_ids, dtype=torch.float32)
        mask_cti = torch.zeros_like(input_ids, dtype=torch.float32)

        for row, sample in enumerate(samples):
            positions = torch.nonzero(input_ids[row] == int(self.st_token_id), as_tuple=False).view(-1)
            n = min(
                int(positions.numel()),
                len(sample.marginal_labels),
                len(sample.cti_labels),
                len(sample.marginal_mask),
                len(sample.cti_mask),
            )
            for idx in range(n):
                pos = int(positions[idx].item())
                targets_marginal[row, pos] = float(sample.marginal_labels[idx])
                targets_cti[row, pos] = float(sample.cti_labels[idx])
                mask_marginal[row, pos] = float(sample.marginal_mask[idx])
                mask_cti[row, pos] = float(sample.cti_mask[idx])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marginal_targets": targets_marginal,
            "cti_targets": targets_cti,
            "marginal_mask": mask_marginal,
            "cti_mask": mask_cti,
        }

    @staticmethod
    def _resolve_st_token_id(tokenizer, st_token: str) -> int:
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            token_id = tokenizer.convert_tokens_to_ids(st_token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                return int(token_id)
        encoded = tokenizer(st_token, return_tensors="pt", add_special_tokens=False)
        return int(encoded.input_ids.view(-1)[0].item())


class SPRMDualHeadTrainingModel(nn.Module):
    def __init__(
        self,
        base_model: AutoModelForCausalLMWithDualValueHead,
        *,
        lambda_marginal: float = 1.0,
        lambda_cti: float = 1.0,
        loss_type: str = "bce",
    ):
        super().__init__()
        if loss_type not in {"bce", "mse"}:
            raise ValueError("loss_type must be bce or mse")
        self.base_model = base_model
        self.lambda_marginal = float(lambda_marginal)
        self.lambda_cti = float(lambda_cti)
        self.loss_type = loss_type

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        marginal_targets: torch.Tensor | None = None,
        cti_targets: torch.Tensor | None = None,
        marginal_mask: torch.Tensor | None = None,
        cti_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        marginal_logits = outputs[2]
        cti_logits = outputs[3]
        result = {"marginal_logits": marginal_logits, "cti_logits": cti_logits}
        if marginal_targets is None or cti_targets is None:
            return result

        marginal_loss = masked_value_loss(
            marginal_logits,
            marginal_targets,
            marginal_mask,
            loss_type=self.loss_type,
        )
        cti_loss = masked_value_loss(
            cti_logits,
            cti_targets,
            cti_mask,
            loss_type=self.loss_type,
        )
        result["loss"] = self.lambda_marginal * marginal_loss + self.lambda_cti * cti_loss
        result["marginal_loss"] = marginal_loss.detach()
        result["cti_loss"] = cti_loss.detach()
        return result


def masked_value_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    loss_type: str,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.float32)
    mask = mask.float()
    if loss_type == "bce":
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    else:
        loss = F.mse_loss(torch.sigmoid(logits), targets.float(), reduction="none")
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def split_dataset(dataset: Stage3Dataset, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if len(indices) < 2 or validation_ratio <= 0:
        return indices, []
    val_size = max(1, int(round(len(indices) * float(validation_ratio))))
    val_size = min(val_size, len(indices) - 1)
    return indices[val_size:], indices[:val_size]


def run_epoch(
    *,
    model: SPRMDualHeadTrainingModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            output = model(**batch)
            loss = output["loss"]
            if is_train:
                loss.backward()
                optimizer.step()
        batch_size = int(batch["input_ids"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total += batch_size
    return {"loss": total_loss / max(total, 1)}


def build_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "add_special_tokens"):
        tokenizer.add_special_tokens({"additional_special_tokens": [args.st_token]})

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=dtype_map.get(args.torch_dtype, "auto"),
        local_files_only=bool(args.local_files_only),
    )
    if hasattr(base, "resize_token_embeddings"):
        base.resize_token_embeddings(len(tokenizer))

    if args.lora_enabled:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, config)
    elif not args.train_backbone:
        for param in base.parameters():
            param.requires_grad = False

    return tokenizer, AutoModelForCausalLMWithDualValueHead(base)


def train_stage3(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, dual_model = build_model_and_tokenizer(args)
    training_model = SPRMDualHeadTrainingModel(
        dual_model,
        lambda_marginal=args.lambda_marginal,
        lambda_cti=args.lambda_cti,
        loss_type=args.loss_type,
    ).to(device)

    dataset = Stage3Dataset(
        args.sprm_label_path,
        st_token=args.st_token,
        label_mode=args.label_mode,
        marginal_threshold=args.marginal_threshold,
        cti_threshold=args.cti_threshold,
        label_temperature=args.label_temperature,
        causal_field=args.causal_field,
        max_samples=args.max_samples,
    )
    if not dataset:
        raise ValueError(f"No valid Stage3 samples found in {args.sprm_label_path}")
    train_indices, val_indices = split_dataset(dataset, args.validation_ratio, args.seed)
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices) if val_indices else None
    collator = DataCollatorForSPRM(tokenizer, st_token=args.st_token, max_length=args.max_length)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = (
        DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if val_subset is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        [param for param in training_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = int(args.early_stop_patience)
    for epoch in range(1, int(args.num_train_epochs) + 1):
        train_metrics = run_epoch(
            model=training_model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        if val_loader is not None:
            val_metrics = run_epoch(
                model=training_model,
                dataloader=val_loader,
                optimizer=None,
                device=device,
            )
            monitor = val_metrics["loss"]
            logger.info("epoch=%d train=%s val=%s", epoch, train_metrics, val_metrics)
        else:
            monitor = train_metrics["loss"]
            logger.info("epoch=%d train=%s", epoch, train_metrics)
        if monitor < best_loss:
            best_loss = monitor
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in training_model.state_dict().items()
                if "v_head" in key or "lora" in key
            }
            patience_left = int(args.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("early stop at epoch=%d", epoch)
                break
    if best_state is not None:
        training_model.load_state_dict(best_state, strict=False)

    model_dir = Path(args.output_dir) / "stage3" / "sprm_prm_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.lora_enabled and hasattr(dual_model.pretrained_model, "save_pretrained"):
        dual_model.pretrained_model.save_pretrained(model_dir)
    elif args.train_backbone and hasattr(dual_model.pretrained_model, "save_pretrained"):
        dual_model.pretrained_model.save_pretrained(model_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(model_dir)
    dual_model.save_dual_heads(model_dir)
    (model_dir / "sprm_model_config.json").write_text(
        json.dumps(
            {
                "base_model_name": args.model_name,
                "st_token": args.st_token,
                "label_mode": args.label_mode,
                "causal_field": args.causal_field,
                "lora_enabled": bool(args.lora_enabled),
                "train_backbone": bool(args.train_backbone),
                "weights_note": (
                    "This smoke-test output stores dual value heads and tokenizer metadata only. "
                    "Load with the referenced base_model_name."
                    if (not args.lora_enabled and not args.train_backbone)
                    else "This directory includes trainable backbone/adapter artifacts plus dual value heads."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "stage3_metrics.json").write_text(
        json.dumps({"best_loss": best_loss, "num_samples": len(dataset)}, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("stage3 saved SPRM PRM model to %s", model_dir)
    return model_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage3: train SPRM dual-head PRM.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sprm-label-path", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--st-token", default="<ST_END>")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label-mode", choices=["binary", "prob", "regression"], default="binary")
    parser.add_argument("--marginal-threshold", type=float, default=0.0)
    parser.add_argument("--cti-threshold", type=float, default=0.0)
    parser.add_argument("--label-temperature", type=float, default=1.0)
    parser.add_argument("--causal-field", default="refined_causal_consistency")
    parser.add_argument("--loss-type", choices=["bce", "mse"], default="bce")
    parser.add_argument("--lambda-marginal", type=float, default=1.0)
    parser.add_argument("--lambda-cti", type=float, default=1.0)

    parser.add_argument("--lora-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_stage3(args)


if __name__ == "__main__":
    main()
