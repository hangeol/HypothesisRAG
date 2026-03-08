#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="${1:-eval_3models}"
GPU_SET="${CUDA_VISIBLE_DEVICES:-0,1,2}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

mkdir -p "$ROOT_DIR/logs/tmux"
TMUX_LOG="$ROOT_DIR/logs/tmux/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES='$GPU_SET' bash scripts/evaluate/run_suite_all_models.sh 2>&1 | tee -a '$TMUX_LOG'"

echo "Started tmux session: $SESSION_NAME"
echo "GPU: $GPU_SET"
echo "tmux log: $TMUX_LOG"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f $TMUX_LOG"

