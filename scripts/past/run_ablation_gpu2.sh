#!/bin/bash
# GPU 2: Llama-3.2-3B hypothesis v2×r2/r3, v3×r1/r2/r3 (5 combos)
set -e
cd /home/bispl_02/hangeol/HypothesisRAG

export CUDA_VISIBLE_DEVICES=2

# h_v2 × r_v2, r_v3
for rv in v2 v3; do
    echo "===== GPU2: Llama3 hypothesis=v2, rewriting=${rv} ====="
    python evaluate_medqa_v2.py --mode hypothesis \
        --hypothesis-prompt v2 --rewriting-prompt ${rv} --generator-prompt v1 \
        --model meta-llama/Llama-3.2-3B-Instruct --max-questions 1273
done

# h_v3 × r_v1, r_v2, r_v3
for rv in v1 v2 v3; do
    echo "===== GPU2: Llama3 hypothesis=v3, rewriting=${rv} ====="
    python evaluate_medqa_v2.py --mode hypothesis \
        --hypothesis-prompt v3 --rewriting-prompt ${rv} --generator-prompt v1 \
        --model meta-llama/Llama-3.2-3B-Instruct --max-questions 1273
done

echo "GPU2 DONE"
