#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/artifacts/real_qwen3_e2e}"

RAW_DIR="${OUTPUT_ROOT}/data/raw"
STEPS_DIR="${OUTPUT_ROOT}/data/steps"
ARTIFACTS_DIR="${OUTPUT_ROOT}/artifacts"
RAW_JSONL="${RAW_DIR}/tiny_math_shepherd.jsonl"
STEPS_JSONL="${STEPS_DIR}/tiny_steps.jsonl"

mkdir -p "${RAW_DIR}" "${STEPS_DIR}" "${ARTIFACTS_DIR}"
cp "${ROOT_DIR}/examples/tiny_math_shepherd.jsonl" "${RAW_JSONL}"

echo "[SPRM smoke] output root: ${OUTPUT_ROOT}"
echo "[SPRM smoke] model: ${MODEL_NAME}"

echo "[SPRM smoke] collect hidden states"
"${PYTHON_BIN}" -m sprm.data.collect_hidden_states \
  --model-name "${MODEL_NAME}" \
  --input-jsonl "${RAW_JSONL}" \
  --output-jsonl "${STEPS_JSONL}" \
  --hidden-state-dir "${STEPS_DIR}" \
  --batch-size 1 \
  --max-length 96 \
  --device cpu \
  --torch-dtype bfloat16 \
  --autocast-dtype none \
  --storage-dtype float16 \
  --local-files-only \
  --overwrite

echo "[SPRM smoke] stage1 trajectory value model"
"${PYTHON_BIN}" -m sprm.training.stage1_value_model \
  --data-file "${STEPS_JSONL}" \
  --output-dir "${ARTIFACTS_DIR}" \
  --hidden-size 0 \
  --batch-size 1 \
  --learning-rate 1e-3 \
  --num-train-epochs 1 \
  --validation-ratio 0.5 \
  --early-stop-patience 1 \
  --device cpu \
  --lstm-hidden-dim 8

echo "[SPRM smoke] stage2 SPRM labels"
"${PYTHON_BIN}" -m sprm.training.stage2_credit_labels \
  --data-file "${STEPS_JSONL}" \
  --stage1-model-dir "${ARTIFACTS_DIR}/stage1/full_model" \
  --output-dir "${ARTIFACTS_DIR}" \
  --svi-steps 2 \
  --svi-target prob \
  --device cpu \
  --enable-refined-causal-consistency \
  --pairwise-window-size 1 \
  --pairwise-beta 0.5 \
  --enable-cti \
  --cti-topk 1 \
  --cti-search-k 2

SPRM_LABEL_PATH="${ARTIFACTS_DIR}/stage2/tiny_steps-sprm-labels.jsonl"

echo "[SPRM smoke] stage3 dual-head PRM"
"${PYTHON_BIN}" -m sprm.training.stage3_dual_prm \
  --model-name "${MODEL_NAME}" \
  --sprm-label-path "${SPRM_LABEL_PATH}" \
  --output-dir "${ARTIFACTS_DIR}" \
  --st-token "<ST_END>" \
  --max-length 128 \
  --batch-size 1 \
  --learning-rate 1e-3 \
  --num-train-epochs 1 \
  --validation-ratio 0.5 \
  --early-stop-patience 1 \
  --device cpu \
  --torch-dtype bfloat16 \
  --local-files-only \
  --no-lora-enabled \
  --label-mode binary \
  --causal-field refined_causal_consistency

echo "[SPRM smoke] done"
echo "[SPRM smoke] key outputs:"
echo "  raw data:      ${RAW_JSONL}"
echo "  step data:     ${STEPS_JSONL}"
echo "  stage1 model:  ${ARTIFACTS_DIR}/stage1/full_model"
echo "  stage2 labels: ${SPRM_LABEL_PATH}"
echo "  stage3 model:  ${ARTIFACTS_DIR}/stage3/sprm_prm_model"
