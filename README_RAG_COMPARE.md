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

### 단일 쿼리 실행

#### Direct Mode (Query Rewriting 없음)
```bash
python rag_compare_runner.py --mode direct \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination."
```

#### Baseline Mode (LLM Query Rewriting)
```bash
python rag_compare_runner.py --mode baseline \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination."
```

#### Planning Mode (Planning 기반 Query Rewriting)
```bash
python rag_compare_runner.py --mode planning \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination."
```

### 3-Way 비교 실행

세 가지 모드를 한 번에 실행하고 비교:

```bash
python rag_compare_runner.py --compare \
    --input "A 45-year-old man presents with fatigue, increased thirst, and frequent urination." \
    --out comparison_results.json
```

### Batch 실행

JSONL 파일로 여러 입력을 한 번에 처리:

```bash
# 모든 모드로 batch 실행
python rag_compare_runner.py --input_file sample_data.jsonl --batch

# 특정 모드만 실행
python rag_compare_runner.py --input_file sample_data.jsonl --batch --modes direct planning
```

#### 입력 파일 형식 (JSONL)
```json
{"id": "q001", "text": "Patient presents with chest pain..."}
{"id": "q002", "text": "A woman with progressive memory loss..."}
```

## 명령줄 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--input`, `-i` | 단일 입력 쿼리 | - |
| `--input_file`, `-f` | 입력 파일 (JSONL) | - |
| `--mode`, `-m` | 실행 모드 (direct/baseline/planning) | - |
| `--compare`, `-c` | 3-way 비교 실행 | False |
| `--batch`, `-b` | Batch 모드 실행 | False |
| `--modes` | Batch에서 실행할 모드들 | 전체 |
| `--top_k`, `-k` | 쿼리당 검색 문서 수 | 5 |
| `--model` | OpenAI 모델명 | gpt-4o-mini |
| `--retriever` | 검색기 이름 | MedCPT |
| `--corpus` | 코퍼스 이름 | Textbooks |
| `--out`, `-o` | 출력 파일 경로 | - |
| `--output_dir` | Batch 출력 디렉토리 | . |
| `--quiet`, `-q` | 출력 최소화 | False |

## 출력 형식

### 단일 결과 (JSON)
```json
{
  "mode": "planning",
  "user_input": "A 45-year-old man presents with...",
  "plan": {
    "observed_features": ["fatigue", "increased thirst", "frequent urination", "fasting blood glucose 142"],
    "must_check_cooccurrence": [["increased thirst", "frequent urination"]],
    "need_disambiguation": []
  },
  "rewritten_queries": [
    "What symptoms indicate diabetes mellitus?",
    "Fasting blood glucose 142 mg/dL diagnostic criteria",
    "Fatigue polyuria polydipsia differential diagnosis"
  ],
  "final_queries": [...],
  "retrieved_docs": [
    {
      "id": "doc_001",
      "title": "Diabetes Mellitus",
      "content": "...",
      "fused_score": 2.45,
      "query_trace": ["query1", "query2"]
    }
  ],
  "metrics": {
    "num_queries": 3,
    "query_lengths": [42, 51, 48],
    "num_retrieved_docs": 10,
    "feature_coverage": 0.75,
    "cooccurrence_coverage": [...]
  }
}
```

### 비교 결과 (JSON)
```json
{
  "user_input": "...",
  "top_k": 5,
  "timestamp": "2026-01-21T...",
  "results": {
    "direct": {...},
    "baseline": {...},
    "planning": {...}
  }
}
```

## 예제 출력

### Direct Mode
```
======================================================================
Mode: DIRECT
======================================================================
Input: A 45-year-old man presents with fatigue, increased thirst...

📝 Final Queries (1):
   1. A 45-year-old man presents with fatigue, increased thirst...

📚 Retrieved Documents (10):
   1. [0.892] Diabetes Mellitus - Overview
   2. [0.856] Glucose Metabolism Disorders
   ...

📊 Metrics:
   num_queries: 1
   query_lengths: [156]
   num_retrieved_docs: 10
```

### Planning Mode
```
======================================================================
Mode: PLANNING
======================================================================
Input: A 45-year-old man presents with fatigue, increased thirst...

📝 Final Queries (3):
   1. What are symptoms of diabetes mellitus type 2?
   2. Fasting blood glucose 142 mg/dL diagnostic significance
   3. Polyuria polydipsia fatigue differential diagnosis

📚 Retrieved Documents (10):
   1. [1.245] Type 2 Diabetes Clinical Features
   2. [1.102] Diagnostic Criteria for Diabetes
   ...

🎯 Plan:
   Features: ['fatigue', 'increased thirst', 'frequent urination', 'blood glucose 142']
   Co-occurrence: [['increased thirst', 'frequent urination']]

📊 Metrics:
   num_queries: 3
   query_lengths: [42, 51, 48]
   num_retrieved_docs: 10
   feature_coverage: 0.750
```

## 아키텍처

```
┌─────────────────┐
│  normalize_input │
└────────┬────────┘
         │
┌────────▼────────┐
│   route_mode    │
└────────┬────────┘
         │
    ┌────┼────┬─────────────┐
    │    │    │             │
    ▼    │    ▼             ▼
┌───────┐│ ┌─────────┐  ┌──────────┐
│direct ││ │baseline │  │make_plan │
│queries││ │rewrite  │  └────┬─────┘
└───┬───┘│ └────┬────┘       │
    │    │      │       ┌────▼─────┐
    │    │      │       │planning  │
    │    │      │       │rewrite   │
    │    │      │       └────┬─────┘
    │    │      │            │
    └────┴──────┴────────────┘
                │
         ┌──────▼──────┐
         │  retrieve   │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ summarize   │
         └─────────────┘
```

## 주요 파일

- `rag_compare_graph.py`: LangGraph 그래프 정의 및 노드 구현
- `rag_compare_runner.py`: CLI 실행 도구
- `retriever.py`: MIRAGE 기반 검색 시스템 래퍼
- `config.py`: 설정 클래스
- `sample_data.jsonl`: 샘플 입력 데이터

## 참고 사항

1. **MIRAGE 검색 시스템**: 실제 MIRAGE MedRAG 코퍼스가 설정되어 있어야 합니다. 설정되지 않은 경우 mock retrieval이 사용됩니다.

2. **API 비용**: OpenAI API를 사용합니다. Planning 모드는 추가 LLM 호출이 필요합니다.

3. **연구 목적**: 이 도구는 연구 실험용입니다. 의료 진단이나 조언 목적으로 사용하지 마세요.

## MedQA 데이터셋 평가

MedQA testset을 사용하여 3-way RAG 비교 실험을 평가할 수 있습니다.

### 평가 실행

```bash
# 빠른 테스트 (10개 문제)
python rag_compare_medqa_eval.py --max-questions 10

# 전체 평가 (모든 문제)
python rag_compare_medqa_eval.py --max-questions 1273

# 특정 모드만 평가
python rag_compare_medqa_eval.py --modes direct planning --max-questions 50

# Evidence만 평가 (답변 생성 없음)
python rag_compare_medqa_eval.py --no-answers --max-questions 100
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

### 평가 결과 예시

```
================================================================================
FINAL RESULTS
================================================================================
Total questions evaluated: 30
Total time: 14.5 minutes
Average time per question: 29.0s

Accuracy by Mode:
----------------------------------------
  direct      :  12/ 30 (40.0%)
  baseline    :  18/ 30 (60.0%)
  planning    :  21/ 30 (70.0%)

Improvement over Direct:
----------------------------------------
  baseline    : +20.0%
  planning    : +30.0%
================================================================================
```

### 결과 파일 형식

결과는 `medqa_rag_compare_{timestamp}.json` 형태로 저장됩니다:

```json
{
  "summary": {
    "config": { ... },
    "timing": { ... },
    "mode_results": {
      "direct": {"correct": 12, "total": 30, "accuracy": 40.0},
      "baseline": {"correct": 18, "total": 30, "accuracy": 60.0},
      "planning": {"correct": 21, "total": 30, "accuracy": 70.0}
    }
  },
  "results": [
    {
      "question_id": 0,
      "question": "...",
      "correct_answer": "B",
      "modes": {
        "direct": {
          "final_queries": [...],
          "predicted_answer": "A",
          "is_correct": false
        },
        "planning": {
          "plan": {"observed_features": [...], ...},
          "final_queries": [...],
          "predicted_answer": "B",
          "is_correct": true
        }
      }
    }
  ]
}
```

## 주요 파일

- `rag_compare_graph.py`: LangGraph 그래프 정의 및 노드 구현
- `rag_compare_runner.py`: CLI 실행 도구
- `rag_compare_medqa_eval.py`: MedQA 평가 스크립트
- `retriever.py`: MIRAGE 기반 검색 시스템 래퍼
- `config.py`: 설정 클래스
- `sample_data.jsonl`: 샘플 입력 데이터

## 참고 사항

1. **MIRAGE 검색 시스템**: 실제 MIRAGE MedRAG 코퍼스가 설정되어 있어야 합니다. 설정되지 않은 경우 mock retrieval이 사용됩니다.

2. **API 비용**: OpenAI API를 사용합니다. Planning 모드는 추가 LLM 호출이 필요합니다.

3. **연구 목적**: 이 도구는 연구 실험용입니다. 의료 진단이나 조언 목적으로 사용하지 마세요.

## 라이센스

연구 및 교육 목적으로 자유롭게 사용할 수 있습니다.
