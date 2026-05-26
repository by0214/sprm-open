# SPRM

This repository contains the reference implementation for **SPRM: From Cooperative Games to Marginal-Contribution Process Reward Modeling**.

SPRM trains a process reward model from outcome-only trajectories. It learns a trajectory value function, converts terminal rewards into step-level marginal credit with Step Value Integration (SVI), applies causal-consistency filtering, estimates cross-trajectory inherent (CTI) scores, and trains a dual-head PRM on `<ST_END>` step boundaries.

## Pipeline

```text
Math-Shepherd trajectories
  -> collect step boundary hidden states
  -> Stage 1: train an outcome-supervised trajectory value model
  -> Stage 2: construct SPRM step labels with SVI, causal consistency, and CTI
  -> Stage 3: train a dual-head PRM on <ST_END> step boundaries
  -> evaluate on ProcessBench and PRMBench
```

## Installation

Create a virtual environment and install the package:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

For a CPU-only smoke environment, install CPU PyTorch first:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python -e .
```

Optional extras:

```bash
uv pip install --python .venv/bin/python -e ".[benchmarks]"
uv pip install --python .venv/bin/python -e ".[speedups]"
uv pip install --python .venv/bin/python -e ".[logging]"
```

## Data Layout

Large data and generated artifacts are intentionally ignored by git:

```text
data/
  math_shepherd/
    raw/      # raw Math-Shepherd-style jsonl
    steps/    # collected hidden-state jsonl and .bin files
  processbench/
    raw/
  prmbench/

artifacts/
  stage1/
  stage2/
  stage3/
```

The Math-Shepherd-style input should contain a reasoning trajectory and terminal correctness. The default collector reads `label` as the trajectory text and infers `final_reward` from a trailing `+` or `-`.

## Training

### 1. Collect Step Boundary Hidden States

```bash
.venv/bin/python -m sprm.data.collect_hidden_states \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --input-jsonl data/math_shepherd/raw/math_shepherd.jsonl \
  --output-jsonl data/math_shepherd/steps/math_shepherd_steps.jsonl \
  --hidden-state-dir data/math_shepherd/steps \
  --batch-size 8 \
  --max-length 2048 \
  --overwrite
```

Output:

```text
data/math_shepherd/steps/math_shepherd_steps.jsonl
data/math_shepherd/steps/math_shepherd_steps.hidden_states.f16.bin
```

### 2. Stage1: Trajectory Value Model

Paper-aligned defaults: single-layer BiLSTM, hidden size 512, bidirectional, BCE loss, learning rate `1e-4`, batch size 128, early stopping patience 5.

```bash
scripts/run_stage1.sh \
  --data-file data/math_shepherd/steps/math_shepherd_steps.jsonl \
  --output-dir artifacts \
  --hidden-size 2560
```

Output:

```text
artifacts/stage1/full_model/trajectory_value_model.pt
artifacts/stage1/full_model/trajectory_value_config.json
artifacts/stage1/trajectory_value_head.bin
```

### 3. Stage2: SPRM Credit Labels

Paper-aligned defaults: SVI with 64 integration steps, probability trajectory value target, zero-ablation causal consistency, refined local causal consistency, CTI top-k 50 and search-k 400.

```bash
scripts/run_stage2.sh \
  --data-file data/math_shepherd/steps/math_shepherd_steps.jsonl \
  --stage1-model-dir artifacts/stage1/full_model \
  --output-dir artifacts
```

Output:

```text
artifacts/stage2/math_shepherd_steps-sprm-labels.jsonl
```

The Stage2 jsonl contains:

- `trajectory_value`
- `marginal_contribution`
- `causal_consistency`
- `refined_causal_consistency`
- `cti_score`
- `cti_base_rate`

### 4. Stage3: Dual-Head PRM

Paper-aligned defaults: Qwen3-4B-Instruct-2507, `<ST_END>` step boundary supervision, binary SVI labels, CTI head, LoRA enabled, max length 4096, learning rate `5e-5`, batch size 32, one epoch.

```bash
scripts/run_stage3.sh \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --sprm-label-path artifacts/stage2/math_shepherd_steps-sprm-labels.jsonl \
  --output-dir artifacts
```

Output:

```text
artifacts/stage3/sprm_prm_model/
  adapter_config.json              # when LoRA is enabled
  adapter_model.safetensors        # when LoRA is enabled
  dual_v_heads.safetensors
  tokenizer files
  sprm_model_config.json
```

## Real Local Smoke Test

If `Qwen/Qwen3-4B-Instruct-2507` is already cached locally, run the same command path that users will run:

```bash
scripts/run_real_qwen3_e2e_smoke.sh
```

This writes persistent intermediate artifacts under:

```text
artifacts/real_qwen3_e2e/
  data/raw/tiny_math_shepherd.jsonl
  data/steps/tiny_steps.jsonl
  data/steps/tiny_steps.hidden_states.f16.bin
  artifacts/stage1/full_model/
  artifacts/stage2/tiny_steps-sprm-labels.jsonl
  artifacts/stage3/sprm_prm_model/
```

The smoke test uses `--local-files-only` and CPU by default. For this smoke path, Stage3 saves tokenizer metadata and `dual_v_heads.*` without copying the full 4B base model into the artifact directory.

## Inference

```python
from sprm.inference import SPRMScorer

scorer = SPRMScorer.from_pretrained("path-or-hf-repo")
scores = scorer.score_steps(
    question="What is 1+1?",
    steps=["Step 1: Add 1 and 1.", "Step 2: The answer is 2."],
    combine="sprm",
)
print(scores.combined)
```

The scorer combines the marginal-contribution head and CTI head. Supported combination modes are `marginal`, `cti`, `mul`, and `sprm`.

## Evaluation Utilities

The package includes shared utilities for ProcessBench and PRMBench-style evaluation:

- `sprm.eval.compute_processbench_metrics`
- `sprm.eval.compute_step_classification_metrics`
- `sprm.eval.compute_pairwise_accuracy`
- `sprm.eval.predict_first_error_from_scores`

Dataset-specific CLI adapters can be added on top of these helpers without changing the training pipeline.
