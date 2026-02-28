#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/0201_fk_only_no_prop"
mkdir -p "$LOG_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/0201_fk_only_no_prop_${TS}.log"

echo "Logging to $LOG_FILE"
stdbuf -oL -eL python -u "$ROOT_DIR/0201_fk_only_no_prop.py" "$@" 2>&1 | tee "$LOG_FILE"
