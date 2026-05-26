#!/usr/bin/env bash
set -euo pipefail

# Stage 2: construct SPRM step labels with SVI, causal consistency, and CTI.
python -m sprm.training.stage2_credit_labels "$@"
