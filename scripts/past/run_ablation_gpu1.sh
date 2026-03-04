#!/bin/bash
# GPU 1: Llama-3.2-3B hypothesis v1,v2 × rewriting v1,v2,v3 + v1×v4 (7 combos)
set -e
cd /home/bispl_02/hangeol/HypothesisRAG

export CUDA_VISIBLE_DEVICES=1

# h_v1 × r_v1, r_v2, r_v3
for rv in v1 v2 v3; do
    echo "===== GPU1: Llama3 hypothesis=v1, rewriting=${rv} ====="
    python evaluate_medqa_v2.py --mode hypothesis \
        --hypothesis-prompt v1 --rewriting-prompt ${rv} --generator-prompt v1 \
        --model meta-llama/Llama-3.2-3B-Instruct --max-questions 1273
done

# h_v2 × r_v1 (1 combo - first of the second half)
echo "===== GPU1: Llama3 hypothesis=v2, rewriting=v1 ====="
python evaluate_medqa_v2.py --mode hypothesis \
    --hypothesis-prompt v2 --rewriting-prompt v1 --generator-prompt v1 \
    --model meta-llama/Llama-3.2-3B-Instruct --max-questions 1273

echo "GPU1 DONE"
