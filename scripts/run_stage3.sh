#!/usr/bin/env bash
set -euo pipefail

# Stage 3: train the dual-head process reward model.
python -m sprm.training.stage3_dual_prm "$@"
