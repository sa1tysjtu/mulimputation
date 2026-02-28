#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/0222_fk_missing_hard_with_factored_bilinear_hardneg"
mkdir -p "$LOG_DIR"
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/0222_fk_missing_hard_with_factored_bilinear_hardneg_${TS}.log"
echo "Logging to $LOG_FILE"
stdbuf -oL -eL python -u "$ROOT_DIR/0222_fk_missing_hard_with_factored_bilinear_hardneg.py" "$@" 2>&1 | tee "$LOG_FILE"
