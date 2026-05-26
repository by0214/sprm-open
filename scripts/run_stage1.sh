#!/usr/bin/env bash
set -euo pipefail

# Stage 1: train the outcome-supervised trajectory value model.
python -m sprm.training.stage1_value_model "$@"
