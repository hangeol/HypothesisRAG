#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/evaluate/run_suite_all_models.sh [model_name ...]

If model names are provided, only those models are queued.
If omitted, the default model list is used.

All model/dataset/prompt runs share a single global GPU queue.
When one GPU finishes, the next queued python job starts immediately on that GPU.
EOF
  exit 0
fi

if (( $# > 0 )); then
  MODELS=("$@")
else
  MODELS=(
    "Qwen/Qwen3-30B-A3B-Instruct-2507"
   # "meta-llama/Llama-3.2-3B-Instruct"
   # "google/medgemma-4b-it"
   # "Qwen/Qwen3-4B-Instruct-2507"
   # "meta-llama/Llama-3.1-8B-Instruct"
  )
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/suites/${RUN_TS}"
SUMMARY_LOG="$LOG_DIR/summary.log"

mkdir -p "$LOG_DIR/jobs"
touch "$SUMMARY_LOG"

GPU_SET="${CUDA_VISIBLE_DEVICES:-6,7}"
# vLLM + torch in this environment must use spawn to avoid CUDA re-init error.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

IFS=',' read -r -a GPU_IDS <<< "$GPU_SET"
GPU_COUNT="${#GPU_IDS[@]}"
if (( GPU_COUNT < 1 )); then
  echo "No GPUs available from CUDA_VISIBLE_DEVICES='$GPU_SET'" >&2
  exit 1
fi

declare -A MAX_QUESTIONS
MAX_QUESTIONS["medqa"]=1273
MAX_QUESTIONS["mmlu"]=1089
DATASETS=("mmlu") #"medqa" )

# One evaluate.py process uses exactly one GPU.
TP_SIZE=1
VLLM_MAX_CONCURRENT="${VLLM_MAX_CONCURRENT:-1}"
EVAL_MAX_CONCURRENT="${EVAL_MAX_CONCURRENT:-8}"
DIRECT_VLLM_MAX_CONCURRENT="${DIRECT_VLLM_MAX_CONCURRENT:-1}"
DIRECT_EVAL_MAX_CONCURRENT="${DIRECT_EVAL_MAX_CONCURRENT:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

log() {
  local msg="$1"
  echo "[$(date '+%F %T')] $msg" | tee -a "$SUMMARY_LOG"
}

slugify() {
  echo "$1" | sed 's#[^A-Za-z0-9._-]#_#g'
}

declare -a QUEUE_MODELS=()
declare -a QUEUE_MODEL_ALIASES=()
declare -a QUEUE_TAGS=()
declare -a QUEUE_CMDS=()

enqueue_job() {
  local model_name="$1"
  local tag="$2"
  local eval_conc="$3"
  local vllm_conc="$4"
  shift 4

  local model_alias
  local result_dir
  local cmd
  local escaped=""

  model_alias="$(slugify "$model_name")"
  result_dir="$ROOT_DIR/outputs/results/local/${model_alias}"
  mkdir -p "$result_dir"

  cmd=(
    python scripts/evaluate/evaluate.py
    "$@"
    --llm-provider vllm
    --model "$model_name"
    --retrieval-dataset textbooks
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm-max-model-len 8192
    --vllm-max-tokens 2048
    --output-dir "$result_dir"
    --max-concurrent "$eval_conc"
    --vllm-tensor-parallel-size "$TP_SIZE"
    --vllm-max-concurrent "$vllm_conc"
  )

  printf -v escaped '%q ' "${cmd[@]}"
  QUEUE_MODELS+=("$model_name")
  QUEUE_MODEL_ALIASES+=("$model_alias")
  QUEUE_TAGS+=("$tag")
  QUEUE_CMDS+=("$escaped")
}

queue_model_jobs() {
  local model_name="$1"

  for dataset in "${DATASETS[@]}"; do
    local max_q="${MAX_QUESTIONS[$dataset]}"

    # enqueue_job "$model_name" "${dataset}_cot" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
    #   --mode cot \
    #   --question-dataset "$dataset" \
    #   --max-questions "$max_q"

    # enqueue_job "$model_name" "${dataset}_directrag" "$DIRECT_EVAL_MAX_CONCURRENT" "$DIRECT_VLLM_MAX_CONCURRENT" \
    #   --mode directrag \
    #   --question-dataset "$dataset" \
    #   --max-questions "$max_q"

    # enqueue_job "$model_name" "${dataset}_directrewriting" "$DIRECT_EVAL_MAX_CONCURRENT" "$DIRECT_VLLM_MAX_CONCURRENT" \
    #   --mode directrewriting \
    #   --question-dataset "$dataset" \
    #   --max-questions "$max_q"

    enqueue_job "$model_name" "${dataset}_hv8_rv12_gv1" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
      --mode hypothesis \
      --question-dataset "$dataset" \
      --max-questions "$max_q" \
      --hypothesis-prompt v8 \
      --rewriting-prompt v12 \
      --generator-prompt v1

    enqueue_job "$model_name" "${dataset}_hv8_rv10_gv1" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
      --mode hypothesis \
      --question-dataset "$dataset" \
      --max-questions "$max_q" \
      --hypothesis-prompt v8 \
      --rewriting-prompt v10 \
      --generator-prompt v1

    enqueue_job "$model_name" "${dataset}_hv7_rv10_gv2" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
      --mode hypothesis \
      --question-dataset "$dataset" \
      --max-questions "$max_q" \
      --hypothesis-prompt v7 \
      --rewriting-prompt v10 \
      --generator-prompt v2

    enqueue_job "$model_name" "${dataset}_hv9_rv10_gv2" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
      --mode hypothesis \
      --question-dataset "$dataset" \
      --max-questions "$max_q" \
      --hypothesis-prompt v9 \
      --rewriting-prompt v10 \
      --generator-prompt v2

      enqueue_job "$model_name" "${dataset}_hv8_rv12_gv2" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
      --mode hypothesis \
      --question-dataset "$dataset" \
      --max-questions "$max_q" \
      --hypothesis-prompt v8 \
      --rewriting-prompt v12 \
      --generator-prompt v2

    enqueue_job "$model_name" "${dataset}_hv8_rv10_gv2" "$EVAL_MAX_CONCURRENT" "$VLLM_MAX_CONCURRENT" \
      --mode hypothesis \
      --question-dataset "$dataset" \
      --max-questions "$max_q" \
      --hypothesis-prompt v8 \
      --rewriting-prompt v10 \
      --generator-prompt v2
  done
}

declare -a SLOT_PIDS=()
declare -a SLOT_MODELS=()
declare -a SLOT_TAGS=()
declare -a SLOT_LOGS=()
declare -A PID_TO_SLOT=()

for slot in "${!GPU_IDS[@]}"; do
  SLOT_PIDS[$slot]=""
  SLOT_MODELS[$slot]=""
  SLOT_TAGS[$slot]=""
  SLOT_LOGS[$slot]=""
done

running_jobs=0

cleanup_running_jobs() {
  local signal_name="$1"
  log "RECEIVED ${signal_name}; stopping ${running_jobs} running job(s)"
  for slot in "${!GPU_IDS[@]}"; do
    local pid="${SLOT_PIDS[$slot]}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

trap 'cleanup_running_jobs INT; exit 130' INT
trap 'cleanup_running_jobs TERM; exit 143' TERM

launch_job() {
  local job_idx="$1"
  local slot="$2"
  local gpu="${GPU_IDS[$slot]}"
  local model_name="${QUEUE_MODELS[$job_idx]}"
  local model_alias="${QUEUE_MODEL_ALIASES[$job_idx]}"
  local tag="${QUEUE_TAGS[$job_idx]}"
  local cmd="${QUEUE_CMDS[$job_idx]}"
  local log_file="$LOG_DIR/jobs/$(printf '%03d' "$job_idx")_${model_alias}_${tag}_gpu${gpu}.log"
  local pid

  log "START model=${model_name} tag=${tag} gpu=${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    eval "$cmd"
  ) >"$log_file" 2>&1 &
  pid="$!"

  SLOT_PIDS[$slot]="$pid"
  SLOT_MODELS[$slot]="$model_name"
  SLOT_TAGS[$slot]="$tag"
  SLOT_LOGS[$slot]="$log_file"
  PID_TO_SLOT["$pid"]="$slot"
  running_jobs=$((running_jobs + 1))
}

for model in "${MODELS[@]}"; do
  queue_model_jobs "$model"
done

TOTAL_JOBS="${#QUEUE_TAGS[@]}"
if (( TOTAL_JOBS == 0 )); then
  log "No jobs queued."
  exit 0
fi

log "MODELS=${#MODELS[@]}"
for model in "${MODELS[@]}"; do
  log "MODEL=${model}"
done
log "CUDA_VISIBLE_DEVICES=${GPU_SET}"
log "GPU_COUNT=${GPU_COUNT}"
log "VLLM_TP_SIZE=${TP_SIZE}"
log "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
log "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
log "LOG_DIR=${LOG_DIR}"
log "TOTAL_JOBS=${TOTAL_JOBS}"
for idx in "${!QUEUE_TAGS[@]}"; do
  log "QUEUE[$idx]=model=${QUEUE_MODELS[$idx]} tag=${QUEUE_TAGS[$idx]}"
done

next_job=0
for slot in "${!GPU_IDS[@]}"; do
  if (( next_job >= TOTAL_JOBS )); then
    break
  fi
  launch_job "$next_job" "$slot"
  next_job=$((next_job + 1))
done

failed=0
while (( running_jobs > 0 )); do
  finished_pid=""
  wait_rc=0
  if ! wait -n -p finished_pid; then
    wait_rc=$?
  fi

  slot="${PID_TO_SLOT[$finished_pid]:-}"
  if [[ -z "$slot" ]]; then
    log "FAILED unknown child pid=${finished_pid:-unset} rc=${wait_rc}"
    failed=1
    continue
  fi

  if (( wait_rc == 0 )); then
    log "DONE model=${SLOT_MODELS[$slot]} tag=${SLOT_TAGS[$slot]} gpu=${GPU_IDS[$slot]}"
  else
    log "FAILED model=${SLOT_MODELS[$slot]} tag=${SLOT_TAGS[$slot]} gpu=${GPU_IDS[$slot]} rc=${wait_rc} (see ${SLOT_LOGS[$slot]})"
    failed=1
  fi

  unset 'PID_TO_SLOT[$finished_pid]'
  SLOT_PIDS[$slot]=""
  SLOT_MODELS[$slot]=""
  SLOT_TAGS[$slot]=""
  SLOT_LOGS[$slot]=""
  running_jobs=$((running_jobs - 1))

  if (( next_job < TOTAL_JOBS )); then
    launch_job "$next_job" "$slot"
    next_job=$((next_job + 1))
  fi
done

if (( failed != 0 )); then
  log "ONE OR MORE RUNS FAILED"
  exit 1
fi

log "ALL RUNS COMPLETED"
