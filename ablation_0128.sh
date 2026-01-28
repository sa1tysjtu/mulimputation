#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$ROOT_DIR/logs/ablation_0128"
mkdir -p "$LOG_ROOT"

TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="$LOG_ROOT/$TS"
mkdir -p "$RUN_DIR"

SUMMARY_TSV="$RUN_DIR/summary.tsv"

: "${DATASET:=rel-f1}"
: "${MISSING_RATIO:=0.3}"
: "${MISSING_MECHANISM:=MCAR}"
: "${SEED:=0}"
: "${EPOCHS:=4000}"
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

GIT_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "N/A")"
GIT_BRANCH="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "N/A")"
GIT_STATUS="$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null || true)"
GIT_DIFF_STAT="$(git -C "$ROOT_DIR" diff --stat 2>/dev/null || true)"

echo "Logging to $RUN_DIR"
echo -e "exp_name\texit_code\tduration_sec\tfinal_rmse\tfinal_mae\tval_rmse_last\tcmd" >"$SUMMARY_TSV"

log_env() {
  local log_file="$1"
  {
    echo "# ===== Environment ====="
    echo "# date: $(date -Is)"
    echo "# host: $(hostname || true)"
    echo "# user: $(whoami || true)"
    echo "# pwd:  $(pwd)"
    echo "# uname: $(uname -a || true)"
    echo "# git_commit: $GIT_COMMIT"
    echo "# git_branch: $GIT_BRANCH"
    if [[ -n "$GIT_STATUS" ]]; then
      echo "# git_status_porcelain:"
      printf '%s\n' "$GIT_STATUS" | sed 's/^/#   /'
    else
      echo "# git_status_porcelain: clean"
    fi
    if [[ -n "$GIT_DIFF_STAT" ]]; then
      echo "# git_diff_stat:"
      printf '%s\n' "$GIT_DIFF_STAT" | sed 's/^/#   /'
    fi
    echo "# python: $(command -v python || true)"
    echo "# python_version: $(python --version 2>&1 || true)"
    python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(f"# torch_version: {torch.__version__}")
    print(f"# torch_cuda_available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"# torch_version: N/A ({e})")
PY
    echo "# ======================="
    echo
  } >>"$log_file"
}

extract_metric() {
  # Usage: extract_metric <line> <key>
  # Example line: "final test num=0.1234 rmse=0.5678 mae=0.1111"
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
  local start_epoch_ts
  start_epoch_ts="$(date -Is)"
  local start_s end_s duration_s exit_code
  start_s="$(date +%s)"

  {
    echo "# ablation_0128.sh"
    echo "# start_time: $start_epoch_ts"
    echo "# exp_name: $exp_name"
    echo "# exp_desc: $exp_desc"
    echo "# base_args: ${BASE_ARGS[*]}"
    echo "# exp_args: ${exp_args[*]}"
    echo "# extra_args: ${EXTRA_ARGS[*]}"
    echo
  } >"$log_file"
  log_env "$log_file"

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
    echo "DRY_RUN=1: skip execution, wrote header to $log_file"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$exp_name" "0" "0" "" "" "" "${cmd[*]}" >>"$SUMMARY_TSV"
    return 0
  fi

  set +e
  stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee -a "$log_file"
  exit_code="${PIPESTATUS[0]}"
  set -e

  end_s="$(date +%s)"
  duration_s="$((end_s - start_s))"

  {
    echo
    echo "# ===== Run Result ====="
    echo "# exit_code: $exit_code"
    echo "# duration_sec: $duration_s"
    echo "# end_time: $(date -Is)"
    echo "# ======================"
  } >>"$log_file"

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

  if [[ "$exit_code" != "0" ]]; then
    echo "WARN: [$exp_name] failed (exit_code=$exit_code). See $log_file"
  fi
}

# Baseline (no cross-table / no FK) 你说已经跑过，这里默认不再跑：
# run_exp "baseline_no_fk" "No FK propagation, no fk loss" \
#   "--weight_fk" "0" "--use_fk_propagation" "0" "--use_soft_fk_propagation" "0"

run_exp "fkprop_only" "Observed-FK hard propagation only (no soft, no fk loss)" \
  "--weight_fk" "0" "--use_fk_propagation" "1" "--use_soft_fk_propagation" "0"

run_exp "fkprop_soft_only" "Observed-FK hard + missing-FK soft propagation (no fk loss)" \
  "--weight_fk" "0" "--use_fk_propagation" "1" "--use_soft_fk_propagation" "1"

run_exp "fkprop_fkloss_w0p2" "Observed-FK hard propagation + fk loss (no soft)" \
  "--weight_fk" "0.2" "--use_fk_propagation" "1" "--use_soft_fk_propagation" "0"

run_exp "fkprop_soft_fkloss_w0p2" "Hard + soft FK propagation + fk loss" \
  "--weight_fk" "0.2" "--use_fk_propagation" "1" "--use_soft_fk_propagation" "1"

run_exp "fkprop_soft_fkloss_w0p05" "Hard + soft FK propagation + fk loss (weight_fk=0.05)" \
  "--weight_fk" "0.05" "--use_fk_propagation" "1" "--use_soft_fk_propagation" "1"

run_exp "soft_only" "Missing-FK soft propagation only (no hard, no fk loss)" \
  "--weight_fk" "0" "--use_fk_propagation" "0" "--use_soft_fk_propagation" "1"

run_exp "fkloss_only_w0p2" "fk loss only (no propagation)" \
  "--weight_fk" "0.2" "--use_fk_propagation" "0" "--use_soft_fk_propagation" "0"

echo "Done. Summary: $SUMMARY_TSV"
