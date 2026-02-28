#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$ROOT_DIR/logs/fk_complete"
mkdir -p "$LOG_ROOT"

TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="$LOG_ROOT/$TS"
mkdir -p "$RUN_DIR"

SUMMARY_TSV="$RUN_DIR/summary.tsv"

: "${DATASET:=rel-f1}"
: "${MISSING_RATIO:=0.3}"
: "${MISSING_MECHANISM:=MCAR}"
: "${SEED:=0}"
: "${EPOCHS:=200}"
: "${DEVICE:=0}"
: "${EVAL_EVERY:=1}"
: "${PATIENCE:=10}"
: "${MIN_DELTA:=1e-4}"
: "${KNOWN:=0.6}"
: "${USE_KNOWN_MASK:=1}"
: "${HIDDEN_DIM:=64}"
: "${GNN_LAYERS:=3}"
: "${DROPOUT:=0.0}"
: "${ACTIVATION:=relu}"
: "${LR:=1e-3}"
: "${WEIGHT_DECAY:=0.0}"
: "${K_NEAR:=50}"
: "${K_RAND:=30}"
: "${POOL_SIZE:=2000}"
: "${TEMPERATURE:=0.1}"
: "${REPORT_PER_TABLE:=1}"
: "${DRY_RUN:=0}"

EXTRA_ARGS=("$@")

BASE_ARGS=(
  "--dataset" "$DATASET"
  "--missing_ratio" "$MISSING_RATIO"
  "--missing_mechanism" "$MISSING_MECHANISM"
  "--seed" "$SEED"
  "--epochs" "$EPOCHS"
  "--device" "$DEVICE"
  "--eval_every" "$EVAL_EVERY"
  "--patience" "$PATIENCE"
  "--min_delta" "$MIN_DELTA"
  "--known" "$KNOWN"
  "--use_known_mask" "$USE_KNOWN_MASK"
  "--hidden_dim" "$HIDDEN_DIM"
  "--gnn_layers" "$GNN_LAYERS"
  "--dropout" "$DROPOUT"
  "--activation" "$ACTIVATION"
  "--lr" "$LR"
  "--weight_decay" "$WEIGHT_DECAY"
  "--k_near" "$K_NEAR"
  "--k_rand" "$K_RAND"
  "--pool_size" "$POOL_SIZE"
  "--temperature" "$TEMPERATURE"
  "--report_per_table" "$REPORT_PER_TABLE"
)

echo "Logging to $RUN_DIR"
echo -e "exp_name\texit_code\tduration_sec\tfinal_rmse\tfinal_mae\tval_rmse_last\tcmd" >"$SUMMARY_TSV"

extract_metric() {
  local line="$1"
  local key="$2"
  echo "$line" | sed -n "s/.*${key}=\\([0-9.]*\\).*/\\1/p"
}

run_exp() {
  local exp_name="$1"
  local exp_desc="$2"
  shift 2
  local -a exp_args=("$@")

  local log_file="$RUN_DIR/${exp_name}.log"
  local start_s end_s duration_s exit_code
  start_s="$(date +%s)"

  {
    echo "# fk_complete ablation"
    echo "# exp_name: $exp_name"
    echo "# exp_desc: $exp_desc"
    echo "# base_args: ${BASE_ARGS[*]}"
    echo "# exp_args: ${exp_args[*]}"
    echo "# extra_args: ${EXTRA_ARGS[*]}"
    echo
  } >"$log_file"

  local -a cmd=(python -u "$ROOT_DIR/train_multi.py")
  cmd+=("${BASE_ARGS[@]}")
  cmd+=("${exp_args[@]}")
  cmd+=("${EXTRA_ARGS[@]}")

  {
    echo "# ===== Command ====="
    echo "# ${cmd[*]}"
    echo "# ==================="
    echo
  } >>"$log_file"

  echo "==> [$exp_name] ${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$exp_name" "0" "0" "" "" "" "${cmd[*]}" >>"$SUMMARY_TSV"
    return 0
  fi

  set +e
  stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee -a "$log_file"
  exit_code="${PIPESTATUS[0]}"
  set -e

  end_s="$(date +%s)"
  duration_s="$((end_s - start_s))"

  local final_line val_line final_rmse final_mae val_rmse_last
  final_line="$(grep -E "final test" "$log_file" | tail -n 1 || true)"
  val_line="$(grep -E "val num=.*rmse=" "$log_file" | tail -n 1 || true)"
  final_rmse="$(extract_metric "$final_line" "rmse")"
  final_mae="$(extract_metric "$final_line" "mae")"
  val_rmse_last="$(extract_metric "$val_line" "rmse")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$exp_name" \
    "$exit_code" \
    "$duration_s" \
    "$final_rmse" \
    "$final_mae" \
    "$val_rmse_last" \
    "${cmd[*]}" >>"$SUMMARY_TSV"
}

# FK complete (do not mask FK), hard propagation only, no fk loss.
run_exp "fk_complete_hard_only" "FK not masked, hard FK propagation only" \
  "--mask_fk" "0" "--weight_fk" "0" "--use_fk_propagation" "1" "--use_soft_fk_propagation" "0"

echo "Done. Summary: $SUMMARY_TSV"
