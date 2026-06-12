from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Scalar value head applied to each token hidden state."""

    def __init__(self, config: Any, summary_dropout_prob: float = 0.1):
        super().__init__()
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from model config.")

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        self.summary = nn.Linear(int(hidden_size), 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class AutoModelForCausalLMWithDualValueHead(nn.Module):
    """CausalLM wrapper with SPRM's marginal-contribution and CTI value heads."""

    def __init__(self, pretrained_model: nn.Module, summary_dropout_prob: float = 0.1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.v_head1 = ValueHead(pretrained_model.config, summary_dropout_prob=summary_dropout_prob)
        self.v_head2 = ValueHead(pretrained_model.config, summary_dropout_prob=summary_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        return_past_key_values: bool = False,
        **kwargs: Any,
    ):
        kwargs["output_hidden_states"] = True
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values

        base_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = base_output.hidden_states[-1]
        marginal_values = self.v_head1(hidden_states).squeeze(-1)
        cti_values = self.v_head2(hidden_states).squeeze(-1)

        result = (
            getattr(base_output, "logits", None),
            getattr(base_output, "loss", None),
            marginal_values,
            cti_values,
        )
        if return_past_key_values:
            return (*result, getattr(base_output, "past_key_values", None))
        return result

    def generate(self, *args: Any, **kwargs: Any):
        return self.pretrained_model.generate(*args, **kwargs)

    def save_dual_heads(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {}
        for name, module in (("v_head1", self.v_head1), ("v_head2", self.v_head2)):
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value.detach().cpu().contiguous()
        try:
            from safetensors.torch import save_file

            save_file(state, str(output_dir / "dual_v_heads.safetensors"))
        except Exception:
            torch.save(state, output_dir / "dual_v_heads.bin")

    def load_dual_heads(
        self,
        model_id_or_path: str | Path,
        token: str | None = None,
        subfolder: str | None = None,
    ) -> None:
        state_path = _resolve_dual_head_path(model_id_or_path, token=token, subfolder=subfolder)
        if state_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(state_path))
        else:
            state = torch.load(str(state_path), map_location="cpu")

        self.v_head1.load_state_dict(_strip_prefix(state, "v_head1."), strict=True)
        self.v_head2.load_state_dict(_strip_prefix(state, "v_head2."), strict=True)

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path, *model_args: Any, **kwargs: Any):
        """Load a PEFT adapter repo or local directory with `dual_v_heads.*`.

        This method is intentionally inference-focused. Stage3 training can use
        its own Trainer wrapper while preserving the same saved head format.
        """

        token = kwargs.get("token", kwargs.get("use_auth_token", None))
        trust_remote_code = bool(kwargs.pop("trust_remote_code", True))
        summary_dropout_prob = float(kwargs.pop("summary_dropout_prob", 0.1))
        subfolder = kwargs.pop("subfolder", None)

        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_ref = str(model_id_or_path)
        peft_config = PeftConfig.from_pretrained(model_ref, subfolder=subfolder, token=token)
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
            token=token,
        )
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            *model_args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if hasattr(base, "resize_token_embeddings"):
            base.resize_token_embeddings(len(tokenizer))
        base = PeftModel.from_pretrained(
            base,
            model_ref,
            is_trainable=False,
            subfolder=subfolder,
            token=token,
        )

        model = cls(base, summary_dropout_prob=summary_dropout_prob)
        model.load_dual_heads(model_ref, token=token, subfolder=subfolder)
        return model


def _resolve_dual_head_path(
    model_id_or_path: str | Path,
    token: str | None,
    subfolder: str | None = None,
) -> Path:
    model_ref = Path(model_id_or_path)
    if model_ref.is_dir():
        if subfolder:
            model_ref = model_ref / subfolder
        for name in ("dual_v_heads.safetensors", "dual_v_heads.bin"):
            candidate = model_ref / name
            if candidate.exists():
                return candidate
    else:
        from huggingface_hub import hf_hub_download

        for name in ("dual_v_heads.safetensors", "dual_v_heads.bin"):
            try:
                return Path(hf_hub_download(str(model_id_or_path), name, subfolder=subfolder, token=token))
            except Exception:
                continue
    raise FileNotFoundError("Could not find dual_v_heads.safetensors or dual_v_heads.bin.")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    stripped = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not stripped:
        raise KeyError(f"No keys found with prefix {prefix!r}.")
    return stripped
