#!/bin/bash
# Wrapper script to run GRPO training with GPU isolation for vLLM and Training
#
# Usage examples:
#   # Use GPUs 0,1,2,3,4 (1 for vLLM, 4 for training)
#   bash run_training.sh --gpus 0,1,2,3,4 --num_vllm_gpus 1 --num_train_gpus 4
#
#   # Use GPUs 6,7 (1 for vLLM, 1 for training) for either rewriter or hypothesis mode
#   bash run_training.sh --gpus 6,7 --num_vllm_gpus 1 --num_train_gpus 1

#   # Hypothesis-only training mode (planner + final-answer reward)
#   bash run_training.sh --mode hypothesis --gpus 0,1,2 --num_vllm_gpus 1 --num_train_gpus 2 \
#       --per_device_bs 1 --total_batch_size 32 --group_size 8 \
#       --hypothesis_prompt v7 --rewriting_prompt v10 --generator_prompt v1 \
#       --base_model Qwen/Qwen3-4B-Instruct-2507

# Default settings
TRAINING_MODE="rewriter"
NUM_VLLM_GPUS=1
NUM_TRAIN_GPUS=2
GPUS_STR="3,4,5"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
TOTAL_BATCH_SIZE=32
PER_DEVICE_BS=1
GROUP_SIZE=8          # num_generations: completions per prompt for GRPO
HYPOTHESIS_PROMPT="v7"
REWRITING_PROMPT="v10"
GENERATOR_PROMPT="v1"
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) TRAINING_MODE="$2"; shift ;;
        --gpus) GPUS_STR="$2"; shift ;;
        --num_vllm_gpus) NUM_VLLM_GPUS="$2"; shift ;;
        --num_train_gpus) NUM_TRAIN_GPUS="$2"; shift ;;
        --base_model) BASE_MODEL="$2"; shift ;;
        --total_batch_size) TOTAL_BATCH_SIZE="$2"; shift ;;
        --per_device_bs) PER_DEVICE_BS="$2"; shift ;;
        --group_size) GROUP_SIZE="$2"; shift ;;
        --hypothesis_prompt) HYPOTHESIS_PROMPT="$2"; shift ;;
        --rewriting_prompt) REWRITING_PROMPT="$2"; shift ;;
        --generator_prompt) GENERATOR_PROMPT="$2"; shift ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

TOTAL_GPUS=$((NUM_VLLM_GPUS + NUM_TRAIN_GPUS))

# Parse comma-separated GPUS_STR into an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS_STR"

if [ ${#GPU_ARRAY[@]} -lt $TOTAL_GPUS ]; then
    echo "Error: You requested $TOTAL_GPUS total GPUs (vLLM:$NUM_VLLM_GPUS + Train:$NUM_TRAIN_GPUS),"
    echo "       but only provided ${#GPU_ARRAY[@]} GPUs in --gpus '$GPUS_STR'."
    exit 1
fi

# Validate numeric inputs
for n in "$NUM_VLLM_GPUS" "$NUM_TRAIN_GPUS" "$TOTAL_BATCH_SIZE" "$PER_DEVICE_BS" "$GROUP_SIZE"; do
    if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -lt 1 ]; then
        echo "Error: numeric arguments must be positive integers."
        exit 1
    fi
done

eval "$(conda shell.bash hook)"
conda activate hypothesisrag

# Ensure MedRAG is in the PYTHONPATH so python can resolve the MIRAGE submodules natively
export PYTHONPATH=$PYTHONPATH:$(pwd)/MIRAGE:$(pwd)/MIRAGE/MedRAG:$(pwd)/MIRAGE/MedRAG/src

# Stability settings for long-running weight sync between TRL trainer and vLLM server
# (TRL's vllm_serve.py already sets VLLM_WORKER_MULTIPROC_METHOD=spawn internally)
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# Force all inter-process communication to use localhost (avoid resolving to external IP)
export MASTER_ADDR=127.0.0.1
export VLLM_HOST_IP=127.0.0.1
export NCCL_SOCKET_IFNAME=lo

# ── 1. Allocate specific GPUs from the array ──
# Allocate vLLM GPUs
VLLM_GPUS_LIST=("${GPU_ARRAY[@]:0:$NUM_VLLM_GPUS}")
# Allocate Train GPUs
TRAIN_GPUS_LIST=("${GPU_ARRAY[@]:$NUM_VLLM_GPUS:$NUM_TRAIN_GPUS}")

# Join arrays back into comma-separated strings for CUDA_VISIBLE_DEVICES
VLLM_CUDA_VISIBLE=$(IFS=, ; echo "${VLLM_GPUS_LIST[*]}")
TRAIN_CUDA_VISIBLE=$(IFS=, ; echo "${TRAIN_GPUS_LIST[*]}")

# ── 2. Configure vLLM logic based on NUM_VLLM_GPUS ──
# Currently we assume vLLM processes map strictly to the isolated CUDA_VISIBLE_DEVICES.
# Because CUDA_VISIBLE_DEVICES remaps the indices for the process, 
# the internal vLLM indices will always be 0, 1, 2 regardless of physical hardware ID.

if [ "$NUM_VLLM_GPUS" -le 1 ]; then
    # Both servers share the 1 physical GPU allocated to them (seen as internal ID "0")
    ROLLOUT_VLLM_IDX="0"
    REWARD_VLLM_IDX="0"
    ROLLOUT_MEM=0.45
    REWARD_MEM=0.45
    ROLLOUT_TP=1
    REWARD_TP=1
    echo "========================================="
    echo "GPU Configuration (Requested $TOTAL_GPUS GPUs total):"
    echo "  Hardware GPUs to use: $GPUS_STR"
    echo "  vLLM Hardware GPUs : $VLLM_CUDA_VISIBLE (Shared by both servers, 0.45 mem each)"
    echo "  Train Hardware GPUs: $TRAIN_CUDA_VISIBLE"
    echo "========================================="
else
    # 2 GPUs for vLLM.
    # Server 1 gets internal ID "0" (Physical: 1st in VLLM_CUDA_VISIBLE)
    # Server 2 gets internal ID "1" (Physical: 2nd in VLLM_CUDA_VISIBLE)
    ROLLOUT_VLLM_IDX="0"
    REWARD_VLLM_IDX="1"
    ROLLOUT_MEM=0.90
    REWARD_MEM=0.90
    ROLLOUT_TP=1
    REWARD_TP=1
    echo "========================================="
    echo "GPU Configuration (Requested $TOTAL_GPUS GPUs total):"
    echo "  Hardware GPUs to use: $GPUS_STR"
    echo "  vLLM Hardware GPUs : $VLLM_CUDA_VISIBLE (Dedicated: 1 for rollout, 1 for reward, 0.90 mem each)"
    echo "  Train Hardware GPUs: $TRAIN_CUDA_VISIBLE"
    echo "========================================="
fi

# ── Server 1: TRL rollout generation server (port 18080) ──
echo "Starting TRL vllm-serve for rollouts (CUDA_VISIBLE_DEVICES=$VLLM_CUDA_VISIBLE/internal_idx=$ROLLOUT_VLLM_IDX, mem=$ROLLOUT_MEM), port 18080..."
# Here we use CUDA_VISIBLE_DEVICES to restrict hardware visibility,
# and CUDA_VISIBLE_DEVICES=$ROLLOUT_VLLM_IDX for the process logic if needed, 
# but vLLM inherently uses all visible devices if TP matches count.
# Since we want to isolate them internal to the visible subset, we can override CUDA_VISIBLE_DEVICES per server
# if internal masking is tricky, but it's much safer to just set CUDA_VISIBLE_DEVICES to the specific physical GPU.

if [ "$NUM_VLLM_GPUS" -le 1 ]; then
    # Shared physical GPU
    ROLLOUT_PHYSICAL="$VLLM_CUDA_VISIBLE"
    REWARD_PHYSICAL="$VLLM_CUDA_VISIBLE"
else
    # Dedicated physical GPUs
    ROLLOUT_PHYSICAL="${VLLM_GPUS_LIST[0]}"
    REWARD_PHYSICAL="${VLLM_GPUS_LIST[1]}"
fi

CUDA_VISIBLE_DEVICES=$ROLLOUT_PHYSICAL trl vllm-serve \
    --model "$BASE_MODEL" \
    --tensor-parallel-size $ROLLOUT_TP \
    --gpu-memory-utilization $ROLLOUT_MEM \
    --max-model-len 8192 \
    --port 18080 &
TRL_VLLM_PID=$!

echo "Waiting for TRL vLLM Server to be ready on port 18080..."
while ! curl -s http://localhost:18080/health/ > /dev/null; do
    sleep 5
done
echo "TRL vLLM Server is ready!"

# ── Server 2: Reward function's frozen generator (port 18081) ──
echo "Starting vLLM API Server for Reward Generator (CUDA_VISIBLE_DEVICES=$REWARD_PHYSICAL, mem=$REWARD_MEM), port 18081..."
CUDA_VISIBLE_DEVICES=$REWARD_PHYSICAL python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --tensor-parallel-size $REWARD_TP \
    --gpu-memory-utilization $REWARD_MEM \
    --max-model-len 8192 \
    --port 18081 &
VLLM_PID=$!

echo "Waiting for vLLM API Server to be ready on port 18081..."
while ! curl -s http://localhost:18081/v1/models > /dev/null; do
    sleep 5
done
echo "vLLM API Server is ready!"

# ── Training on remaining GPUs ──
# TRL GRPOTrainer: generation_batch_size = per_device_bs × gpus × grad_accum
# This must be divisible by num_generations (GROUP_SIZE).
# unique_prompts = generation_batch_size / GROUP_SIZE
BASE_DENOM=$((PER_DEVICE_BS * NUM_TRAIN_GPUS))
if [ "$BASE_DENOM" -lt 1 ]; then
    echo "Error: invalid base denominator for grad accumulation."
    exit 1
fi

if [ $((TOTAL_BATCH_SIZE % BASE_DENOM)) -ne 0 ]; then
    echo "Error: total_batch_size ($TOTAL_BATCH_SIZE) must be divisible by per_device_bs×num_train_gpus ($BASE_DENOM)."
    echo "       Suggested: change --total_batch_size or --per_device_bs/--num_train_gpus."
    exit 1
fi

GRAD_ACCUM=$((TOTAL_BATCH_SIZE / BASE_DENOM))
GEN_BATCH=$((PER_DEVICE_BS * NUM_TRAIN_GPUS * GRAD_ACCUM))

if [ "$GEN_BATCH" -ne "$TOTAL_BATCH_SIZE" ]; then
    echo "Error: generation_batch_size mismatch ($GEN_BATCH != $TOTAL_BATCH_SIZE)."
    exit 1
fi

if [ $((GEN_BATCH % GROUP_SIZE)) -ne 0 ]; then
    echo "Error: generation_batch_size ($GEN_BATCH) must be divisible by group_size ($GROUP_SIZE)."
    exit 1
fi

NUM_PROMPTS=$((GEN_BATCH / GROUP_SIZE))
echo "Batch config: generation_batch_size=$GEN_BATCH (=$PER_DEVICE_BS×$NUM_TRAIN_GPUS×$GRAD_ACCUM)"
echo "  → $NUM_PROMPTS unique prompts × $GROUP_SIZE generations = $GEN_BATCH total (target: $TOTAL_BATCH_SIZE)"
echo "Prompt config: hypothesis=$HYPOTHESIS_PROMPT, rewriting=$REWRITING_PROMPT, generator=$GENERATOR_PROMPT"

# Select training script by mode.
if [ "$TRAINING_MODE" = "hypothesis" ]; then
    TRAINING_SCRIPT="training/train_hypothesis_grpo.py"
elif [ "$TRAINING_MODE" = "rewriter" ]; then
    TRAINING_SCRIPT="training/train_rewriter_grpo.py"
else
    echo "Error: unknown training mode '$TRAINING_MODE'. Use rewriter or hypothesis."
    exit 1
fi

# Pre-generate plan cache only for rewriter mode.
if [ "$TRAINING_MODE" = "rewriter" ]; then
    # ── Pre-generate plan cache (single process, before accelerate) ──
    # This avoids creating offline vLLM inside accelerate's multi-GPU processes
    echo "Pre-generating plan cache on GPU ${TRAIN_GPUS_LIST[0]}..."
    CUDA_VISIBLE_DEVICES=${TRAIN_GPUS_LIST[0]} python -c "
from data.medqa_loader import generate_plans_vllm, load_medqa_benchmark
import hashlib, os
benchmark_path = '${BENCHMARK_PATH:-data/medqa_train.json}'
split = '${SPLIT:-medqa_train}'
base_model = '$BASE_MODEL'
raw_data = load_medqa_benchmark(benchmark_path, split)
question_ids = sorted(raw_data.keys())
questions = [raw_data[qid] for qid in question_ids]
model_hash = hashlib.md5(base_model.encode()).hexdigest()[:8]
cache_path = os.path.join('outputs', f'plan_cache_{split}_{model_hash}.json')
generate_plans_vllm(questions, question_ids, base_model, cache_path=cache_path,
                    gpu_memory_utilization=0.4, max_model_len=8192)
print('Plan cache ready:', cache_path)
"
    echo "Plan cache generation complete."
fi

echo "Starting accelerate training on GPUs (CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE)..."
CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE accelerate launch \
    --num_processes=$NUM_TRAIN_GPUS \
    --main_process_port=29500 \
    $TRAINING_SCRIPT \
    --base_model "$BASE_MODEL" \
    --generator_tp $REWARD_TP \
    --batch_size $PER_DEVICE_BS \
    --grad_accum $GRAD_ACCUM \
    --group_size $GROUP_SIZE \
    --hypothesis_prompt "$HYPOTHESIS_PROMPT" \
    --rewriting_prompt "$REWRITING_PROMPT" \
    --generator_prompt "$GENERATOR_PROMPT" \
    "${EXTRA_ARGS[@]}"

# Cleanup vLLM servers after training finishes
kill $VLLM_PID || true
kill $TRL_VLLM_PID || true
