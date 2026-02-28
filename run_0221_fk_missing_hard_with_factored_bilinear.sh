#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/0221_fk_missing_hard_with_factored_bilinear"
mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/0221_fk_missing_hard_with_factored_bilinear_${TS}.log"
echo "Logging to $LOG_FILE"
stdbuf -oL -eL python -u "$ROOT_DIR/0221_fk_missing_hard_with_factored_bilinear.py" "$@" 2>&1 | tee "$LOG_FILE"
