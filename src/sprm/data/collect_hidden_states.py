from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .hidden_state_store import HiddenStateBinWriter
from .schemas import BoundaryStep, TrajectoryRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedTrajectory:
    sample_id: str | None
    question: str
    full_text: str
    question_text: str
    step_texts: list[str]
    final_reward: float
    metadata: dict[str, Any]


def dumps_jsonl(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skip invalid json at %s:%d (%s)", path, line_no, exc)
                continue
            if isinstance(obj, dict):
                yield obj


def remove_trailing_reward_marks(text: str) -> str:
    """Remove Math-Shepherd-style trailing +/- labels from each line."""

    return re.sub(r"\s+[+-]\s*$", "", text, flags=re.MULTILINE).strip()


def infer_reward_from_suffix(text: str) -> float | None:
    stripped = text.rstrip()
    if stripped.endswith("+"):
        return 1.0
    if stripped.endswith("-"):
        return 0.0
    return None


def split_question_and_steps(text: str, step_regex: str) -> tuple[str, list[str]]:
    """Split a trajectory into question text and step texts.

    The default pattern recognizes spans beginning with `Step 1:`, `Step 2:`,
    etc. The returned step strings keep their `Step i:` prefix.
    """

    pattern = re.compile(step_regex)
    matches = list(pattern.finditer(text))
    if not matches:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return text.strip(), []
        return lines[0], lines[1:]

    question = text[: matches[0].start()].strip()
    steps: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        step = text[start:end].strip()
        if step:
            steps.append(step)
    return question, steps


def parse_reward(raw: dict[str, Any], *, reward_field: str, text: str, reward_from_suffix: bool) -> float | None:
    if reward_field and raw.get(reward_field) is not None:
        try:
            return float(raw[reward_field])
        except Exception:
            return None
    if reward_from_suffix:
        return infer_reward_from_suffix(text)
    return None


def parse_trajectory(
    raw: dict[str, Any],
    *,
    text_field: str,
    question_field: str,
    id_field: str,
    reward_field: str,
    reward_from_suffix: bool,
    step_regex: str,
) -> ParsedTrajectory | None:
    raw_text = raw.get(text_field)
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    final_reward = parse_reward(
        raw,
        reward_field=reward_field,
        text=raw_text,
        reward_from_suffix=reward_from_suffix,
    )
    if final_reward is None:
        return None

    clean_text = remove_trailing_reward_marks(raw_text)
    question_text, step_texts = split_question_and_steps(clean_text, step_regex)
    if not question_text or not step_texts:
        return None

    question = raw.get(question_field)
    if not isinstance(question, str) or not question.strip():
        question = question_text

    sample_id = raw.get(id_field)
    if sample_id is not None:
        sample_id = str(sample_id)

    metadata = {}
    for key in ("task", "source", "dataset"):
        if key in raw:
            metadata[key] = raw[key]

    return ParsedTrajectory(
        sample_id=sample_id,
        question=question,
        full_text=clean_text,
        question_text=question_text,
        step_texts=step_texts,
        final_reward=float(final_reward),
        metadata=metadata,
    )


def prefix_texts_for_boundaries(parsed: ParsedTrajectory) -> list[tuple[str, str]]:
    boundaries: list[tuple[str, str]] = [("question", parsed.question_text)]
    current = parsed.question_text
    for step in parsed.step_texts:
        current = f"{current}\n{step}" if current else step
        boundaries.append(("step", current))
    return boundaries


def token_end_index(tokenizer, text: str) -> int:
    tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return int(tokenized.input_ids.shape[1] - 1)


def get_last_hidden_state(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    return outputs[0]


def build_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    torch_dtype: str,
    *,
    local_files_only: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype_map.get(torch_dtype, "auto"),
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def collect_batch(
    batch: list[ParsedTrajectory],
    *,
    tokenizer,
    model,
    writer: HiddenStateBinWriter,
    hidden_state_file: str,
    device: torch.device,
    max_length: int,
    autocast_dtype: str,
) -> list[dict[str, Any]]:
    texts = [item.full_text for item in batch]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    dtype = torch.bfloat16 if autocast_dtype == "bfloat16" else torch.float16
    use_autocast = device.type == "cuda" and autocast_dtype in {"float16", "bfloat16"}
    with torch.inference_mode():
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = get_last_hidden_state(outputs)

    records: list[dict[str, Any]] = []
    for batch_idx, parsed in enumerate(batch):
        valid_len = int(attention_mask[batch_idx].sum().item())
        steps: list[BoundaryStep] = []

        for role, prefix in prefix_texts_for_boundaries(parsed):
            end_idx = token_end_index(tokenizer, prefix)
            if end_idx >= valid_len:
                logger.warning(
                    "skip truncated boundary sample_id=%s role=%s end_idx=%d valid_len=%d",
                    parsed.sample_id,
                    role,
                    end_idx,
                    valid_len,
                )
                continue

            vector = hidden_states[batch_idx, end_idx, :].detach().float().cpu().numpy()
            ref = writer.write_vector(vector).to_dict()
            ref["file"] = hidden_state_file
            steps.append(BoundaryStep(prefix_text=prefix, hidden_state_ref=ref, role=role))

        if len(steps) < 2:
            continue

        record = TrajectoryRecord(
            sample_id=parsed.sample_id,
            question=parsed.question,
            final_reward=parsed.final_reward,
            steps=steps,
            metadata=parsed.metadata,
        )
        records.append(record.to_dict())

    return records


def run_collection(args: argparse.Namespace) -> None:
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    hidden_state_dir = Path(args.hidden_state_dir) if args.hidden_state_dir else output_jsonl.parent
    hidden_state_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{output_jsonl} exists; pass --overwrite to replace it")

    hidden_state_file = args.hidden_state_file or f"{output_jsonl.stem}.hidden_states.f16.bin"
    hidden_state_path = hidden_state_dir / hidden_state_file
    if hidden_state_path.exists() and not args.overwrite:
        raise FileExistsError(f"{hidden_state_path} exists; pass --overwrite to replace it")

    device = build_device(args.device)
    logger.info("loading model=%s device=%s", args.model_name, device)
    tokenizer, model = load_model_and_tokenizer(
        args.model_name,
        device,
        args.torch_dtype,
        local_files_only=bool(args.local_files_only),
    )

    parsed_batch: list[ParsedTrajectory] = []
    written = 0
    skipped = 0

    with HiddenStateBinWriter(hidden_state_path, dtype=args.storage_dtype, mode="wb") as writer:
        with output_jsonl.open("wb") as out:
            iterator = iter_jsonl(input_jsonl)
            for raw in tqdm(iterator, desc="collect hidden states"):
                parsed = parse_trajectory(
                    raw,
                    text_field=args.text_field,
                    question_field=args.question_field,
                    id_field=args.id_field,
                    reward_field=args.reward_field,
                    reward_from_suffix=args.reward_from_suffix,
                    step_regex=args.step_regex,
                )
                if parsed is None:
                    skipped += 1
                    continue

                parsed_batch.append(parsed)
                if len(parsed_batch) < args.batch_size:
                    continue

                records = collect_batch(
                    parsed_batch,
                    tokenizer=tokenizer,
                    model=model,
                    writer=writer,
                    hidden_state_file=hidden_state_file,
                    device=device,
                    max_length=args.max_length,
                    autocast_dtype=args.autocast_dtype,
                )
                for record in records:
                    out.write(dumps_jsonl(record))
                written += len(records)
                parsed_batch = []

            if parsed_batch:
                records = collect_batch(
                    parsed_batch,
                    tokenizer=tokenizer,
                    model=model,
                    writer=writer,
                    hidden_state_file=hidden_state_file,
                    device=device,
                    max_length=args.max_length,
                    autocast_dtype=args.autocast_dtype,
                )
                for record in records:
                    out.write(dumps_jsonl(record))
                written += len(records)

    logger.info("wrote %d trajectories to %s (skipped=%d)", written, output_jsonl, skipped)
    logger.info("hidden states stored at %s", hidden_state_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect question and step-boundary hidden states for SPRM."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--hidden-state-dir", default="")
    parser.add_argument("--hidden-state-file", default="")

    parser.add_argument("--text-field", default="label")
    parser.add_argument("--question-field", default="input")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--reward-field", default="")
    parser.add_argument("--reward-from-suffix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-regex", default=r"(?i)\bStep\s+\d+\s*:")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--autocast-dtype",
        default="bfloat16",
        choices=["none", "float16", "bfloat16"],
    )
    parser.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_collection(args)


if __name__ == "__main__":
    main()
