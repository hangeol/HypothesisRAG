#!/bin/bash
# GPU 0: Qwen3 hypothesis v4 × rewriting v4
set -e
cd /home/bispl_02/hangeol/HypothesisRAG

export CUDA_VISIBLE_DEVICES=0

echo "===== GPU0: Qwen3 hypothesis=v4, rewriting=v4, generator=v1 ====="
python evaluate_medqa_v2.py --mode hypothesis \
    --hypothesis-prompt v4 --rewriting-prompt v4 --generator-prompt v1 \
    --model Qwen/Qwen3-4B-Instruct-2507 --max-questions 1273
echo "GPU0 DONE"
