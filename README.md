# RAG Comparison Experiment

LangGraph 기반의 RAG(Retrieval-Augmented Generation) 비교 실험 도구입니다.

## 목적

세 가지 RAG 전략을 비교하여 Query Rewriting(특히 Planning 기반)이 검색 품질에 어떻게 도움이 되는지 확인합니다:

- **(A) direct_rag**: Query rewriting 없이 사용자 입력 그대로 검색
- **(B) baseline_rewrite_rag**: LLM이 알아서 query rewriting (Planning 없음)
- **(C) planning_rewrite_rag**: 간단한 planning 후 plan 기반 query rewriting

**주의**: 이 도구는 연구 실험용입니다. 의학적 조언을 제공하지 않으며, 진단 확정을 하지 않습니다.

## 설치

### 1. 환경 설정

```bash
# Conda 환경 생성 및 활성화 (권장)
conda create -n rag_compare python=3.10
conda activate rag_compare

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 사용법

### 단일 쿼리 실행 (CLI)

#### Direct Mode (Query Rewriting 없음)
```bash
python run_rag.py --mode direct \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination."
```

#### Baseline Mode (LLM Query Rewriting)
```bash
python run_rag.py --mode baseline \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination."
```

#### Planning Mode (Planning 기반 Query Rewriting)
```bash
python run_rag.py --mode planning \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination."
```

### 3-Way 비교 실행

세 가지 모드를 한 번에 실행하고 비교:

```bash
python run_rag.py --compare \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination." \
    --out comparison_results.json
```

### Batch 실행

JSONL 파일로 여러 입력을 한 번에 처리:

```bash
# 모든 모드로 batch 실행
python run_rag.py --input_file sample_data.jsonl --batch

# 특정 모드만 실행
python run_rag.py --input_file sample_data.jsonl --batch --modes direct planning
```

## MedQA 데이터셋 평가 (Async)

MedQA testset을 사용하여 3-way RAG 비교 실험을 평가합니다. 비동기 처리를 통해 빠르게 수행됩니다.

### 평가 실행

```bash
# 빠른 테스트 (10개 문제)
python evaluate_medqa.py --max-questions 10

# 전체 평가 (모든 문제)
# 주의: 모든 문제를 평가하려면 시간이 오래 걸릴 수 있습니다.
python evaluate_medqa.py --max-questions 1273

# 특정 모드만 평가
python evaluate_medqa.py --modes direct planning --max-questions 50

# Evidence만 평가 (답변 생성 없음 - 검색 성능 측정용)
python evaluate_medqa.py --no-answers --max-questions 100
```

### 평가 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--max-questions`, `-n` | 평가할 최대 문제 수 | 100 |
| `--modes`, `-m` | 평가할 모드들 | 전체 |
| `--top-k`, `-k` | 쿼리당 검색 문서 수 | 5 |
| `--model` | OpenAI 모델명 | gpt-4o-mini |
| `--no-answers` | 답변 생성 건너뛰기 | False |
| `--output-dir`, `-o` | 결과 저장 디렉토리 | . |

## 분석 스크립트 (`scripts/`)

- `scripts/analyze_failure_modes.py`: 실패 유형 분석 (R-GAP, R-NOISE 등)
- `scripts/analyze_best_guess_vs_gold.py`: Planning 단계의 Best Guess 정확도 분석
- `scripts/compare_v4_v5_results.py`: V4 vs V5 결과 비교

## 주요 파일 구조

```text
/
├── rag_core.py                # Core Logic (Graph definition)
├── evaluate_medqa.py          # Main Evaluation Script (Async)
├── run_rag.py                 # CLI Runner
├── retriever.py               # Retriever Module
├── config.py                  # Configuration
├── scripts/                   # Analysis & Utility Scripts
│   ├── analyze_failure_modes.py
│   ├── analyze_best_guess_vs_gold.py
│   └── compare_v4_v5_results.py
└── etc/                       # Legacy & Test Files
    ├── evaluate_medqa_sync_legacy.py # Old synchronous evaluator
    └── test_json_parsing.py
```

## 참고 사항

1. **MIRAGE 검색 시스템**: 실제 MIRAGE MedRAG 코퍼스가 설정되어 있어야 합니다. 설정되지 않은 경우 mock retrieval이 사용됩니다.
2. **API 비용**: OpenAI API를 사용합니다. Planning 모드는 추가 LLM 호출이 필요합니다.
3. **연구 목적**: 이 도구는 연구 실험용입니다. 의료 진단이나 조언 목적으로 사용하지 마세요.

## 라이센스

연구 및 교육 목적으로 자유롭게 사용할 수 있습니다.
