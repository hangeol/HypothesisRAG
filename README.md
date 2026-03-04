# HypothesisRAG

MedQA 기반 HypothesisRAG 실험 코드입니다.  
현재 구조는 `평가(Evaluate) / 학습(Train) / 분석(Analysis) / 과거(Past)`로 분리되어 있습니다.

## 1. 폴더 구조

```text
HypothesisRAG/
├── scripts/
│   ├── evaluate/
│   │   ├── evaluate.py                 # (기존 evaluate_medqa_v2.py)
│   │   ├── evaluate_checkpoints.py     # 체크포인트 스윕 평가
│   ├── train/
│   │   └── run_training.sh             # 학습 실행 래퍼
│   ├── analysis/
│   │   ├── eval_grpo.py
│   │   ├── analyze_failure_modes.py
│   │   ├── analyze_best_guess_vs_gold.py
│   │   ├── compare_v4_v5_results.py
│   │   └── prune_checkpoints_for_inference.py
│   └── past/
│       ├── evaluate_past.py            # (기존 evaluate_medqa.py)
│       ├── run_ablation_gpu*.sh
│       ├── run_rag.py
│       ├── test_rag.py
│       └── etc/*                       # legacy 코드
├── training/
│   ├── train_hypothesis_grpo.py
│   ├── train_rewriter_grpo.py
│   └── reward.py
├── core/
│   ├── prompts.py                    # canonical prompt registry
│   └── rag_core.py                   # canonical RAG core module
├── data/
├── outputs/
├── retrieval/
│   └── retriever.py                  # canonical retriever module
└── requirements*.txt
```

## 2. 환경 설정

### 2.1 Conda

```bash
conda create -n hypothesisrag python=3.10 -y
conda activate hypothesisrag
```

### 2.2 패키지 설치

전체(평가+학습+분석):

```bash
pip install -r requirements.txt
```

선택 설치:

```bash
pip install -r requirements-eval.txt
pip install -r requirements-train.txt
pip install -r requirements-analysis.txt
```

### 2.3 환경변수

```bash
export OPENAI_API_KEY="<your_key>"
```

## 3. 결과 저장 규칙

`scripts/evaluate/evaluate.py`는 `--output-dir`를 지정하지 않으면 자동으로 아래에 저장합니다.

```text
outputs/results/<provider>/<model>/...
```

예시:
- OpenAI: `outputs/results/openai/gpt-4o-mini/...`
- Local(vLLM): `outputs/results/local/Qwen_Qwen3-4B-Instruct-2507/...`

`--output-dir`를 지정하면 해당 경로를 우선 사용합니다.

## 4. Evaluate 사용법

### 4.0 Prompt 로딩 규칙 (`load_mirage_prompts`)

- 기본 프롬프트 레지스트리는 `core/prompts.py`입니다.
- 실행 시 `MIRAGE/MedRAG/src/template.py`가 있으면 `load_mirage_prompts()`가 MIRAGE 원본 시스템 프롬프트(`general_medrag_system`, `general_cot_system`)를 읽어와 덮어씁니다.
- Direct rewriting용 query prompt는 `core/prompts.py`의 `DIRECT_REWRITING_PROMPT`입니다.
- Direct rewriting의 system prompt는 `DIRECT_REWRITING_SYSTEM_PROMPT=""`(빈 문자열)이며, `DIRECT_REWRITING_TARGET_QUERIES=3`, `DIRECT_REWRITING_DOCS_PER_QUERY=5`(총 15)로 고정됩니다.
- 목적: 로컬 복사본과 MIRAGE 원본 간 프롬프트 drift를 방지하고, CoT/Generator baseline을 MIRAGE와 동일하게 유지하기 위함입니다.

### 4.1 단일 평가 (`evaluate.py`)

OpenAI + Hypothesis 모드:

```bash
python scripts/evaluate/evaluate.py \
  --mode hypothesis \
  --llm-provider openai \
  --model gpt-4o-mini \
  --hypothesis-prompt v7 \
  --rewriting-prompt v10 \
  --generator-prompt v1 \
  --max-questions 1273
```

Local(vLLM) + Hypothesis 모드:

```bash
python scripts/evaluate/evaluate.py \
  --mode hypothesis \
  --llm-provider vllm \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --hypothesis-prompt v7 \
  --rewriting-prompt v10 \
  --generator-prompt v1 \
  --vllm-tensor-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.9 \
  --vllm-max-model-len 8192 \
  --vllm-max-tokens 2048 \
  --max-questions 1273
```

체크포인트 override (예: hypothesis/rewriter만 교체):

```bash
python scripts/evaluate/evaluate.py \
  --mode hypothesis \
  --llm-provider vllm \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --hypothesis-checkpoint /path/to/hyp/checkpoint-1400 \
  --rewriter-checkpoint /path/to/rew/checkpoint-1300 \
  --hypothesis-prompt v7 --rewriting-prompt v10 --generator-prompt v1
```

Baseline 모드:

```bash
python scripts/evaluate/evaluate.py --mode cot --llm-provider openai --model gpt-4o-mini
python scripts/evaluate/evaluate.py --mode directrag --llm-provider vllm --model Qwen/Qwen3-4B-Instruct-2507
python scripts/evaluate/evaluate.py --mode directrewriting --llm-provider vllm --model Qwen/Qwen3-4B-Instruct-2507
```

### 4.2 체크포인트 스윕 (`evaluate_checkpoints.py`)

```bash
python scripts/evaluate/evaluate_checkpoints.py \
  outputs/hypothesis_grpo/20260228_172844 \
  --mode hypothesis \
  --gpus 0,2 \
  --max-questions 1273 \
  --total-docs 15 \
  --gpu-mem 0.35 \
  --max-model-len 8192 \
  --max-tokens 2048 \
  --vllm-tensor-parallel-size 1 \
  --hypothesis-prompt v7 --rewriting-prompt v10 --generator-prompt v1
```

### 4.3 Prompt/Model Ablation (`evaluate.py` 단일 진입점)

```bash
python scripts/evaluate/evaluate.py \
  --mode hypothesis \
  --llm-provider openai \
  --ablation-models gpt-4o-mini gpt-4o \
  --ablation-combos v5-v5-v2 v7-v10-v2 \
  --max-questions 1273 \
  --max-concurrent 120
```

## 5. Train 사용법

메인 래퍼: `scripts/train/run_training.sh`  
(스크립트가 프로젝트 루트로 자동 이동하므로 어느 경로에서 실행해도 됩니다.)

### 5.1 Rewriter 학습

```bash
bash scripts/train/run_training.sh \
  --mode rewriter \
  --gpus 0,1,2 \
  --num_vllm_gpus 1 \
  --num_train_gpus 2 \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --per_device_bs 1 \
  --total_batch_size 32 \
  --group_size 8 \
  --hypothesis_prompt v7 --rewriting_prompt v10 --generator_prompt v1
```

### 5.2 Hypothesis 학습

```bash
bash scripts/train/run_training.sh \
  --mode hypothesis \
  --gpus 0,1,2 \
  --num_vllm_gpus 1 \
  --num_train_gpus 2 \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --per_device_bs 1 \
  --total_batch_size 32 \
  --group_size 8 \
  --hypothesis_prompt v7 --rewriting_prompt v10 --generator_prompt v1
```

학습 진입점 직접 실행:
- `training/train_hypothesis_grpo.py`
- `training/train_rewriter_grpo.py`

## 6. Analysis 사용법

```bash
python scripts/analysis/eval_grpo.py --help
python scripts/analysis/analyze_failure_modes.py --help
python scripts/analysis/analyze_best_guess_vs_gold.py --help
python scripts/analysis/compare_v4_v5_results.py
python scripts/analysis/prune_checkpoints_for_inference.py --help
```

## 7. Past 코드

사용하지 않는 과거 코드는 `scripts/past/`로 분리했습니다.

- `scripts/past/evaluate_past.py` (기존 `evaluate_medqa.py`)
- `scripts/past/run_gpt_ablation.py`
- `scripts/past/run_ablation_gpu*.sh`
- `scripts/past/run_rag.py`, `scripts/past/test_rag.py`
- `scripts/past/etc/*`
