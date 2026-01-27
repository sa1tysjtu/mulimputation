#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/ours_${TS}.log"

echo "Logging to $LOG_FILE"
stdbuf -oL -eL python -u "$ROOT_DIR/train_multi.py" "$@" 2>&1 | tee "$LOG_FILE"
