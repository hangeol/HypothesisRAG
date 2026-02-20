# HypothesisRAG

의료 도메인 질의응답을 위한 검색 증강 생성(RAG) 파이프라인 실험 프로젝트입니다.
현재 프로젝트의 **핵심 스크립트는 `evaluate_medqa.py`**이며, 이 중에서도 **`planning_v4`** 모드가 🩺가설(Hypothesis)을 세우고 이를 검증하기 위해 RAG를 수행하는 본 프로젝트의 **핵심 방법론**입니다.

## 목적

여러 RAG 전략을 비교하여 모델이 스스로 가설을 세우고 검증하는 방식(Hypothesis 기반 Planning)이 검색 품질 및 최종 답변에 어떤 도움을 주는지 확인합니다:

- **(A) direct**: Query rewriting 없이 사용자 입력 그대로 검색
- **(B) baseline**: LLM이 자체적으로 일반적인 query rewriting 수행
- **(C) planning_v4 (⭐️ 핵심 방법론)**: 주어진 의료 문제에 대해 초기 진단 가설(Best Guess)을 세우고, 이 가설을 확정짓거나 감별 진단하기 위해 필요한 구체적 증거를 찾도록 타겟팅된 검색 쿼리를 생성하는 HypothesisRAG 방법론


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


## MedQA 데이터셋 평가 (Async) - ⭐️ 핵심 스크립트 (`evaluate_medqa.py`)

`evaluate_medqa.py`를 통해 전체 파이프라인 및 HypothesisRAG(`planning_v4`)의 성능을 평가합니다. 이 스크립트가 현 RAG 실험 및 검증의 코어(Core) 역할을 수행하며 비동기 처리를 통해 빠르게 실행됩니다.

### 평가 실행

```bash
# 빠른 테스트 (10개 문제)
python evaluate_medqa.py --max-questions 10

# 전체 평가 (모든 문제)
# 주의: 모든 문제를 평가하려면 시간이 오래 걸릴 수 있습니다.
python evaluate_medqa.py --max-questions 1273

# 특정 모드만 평가 (예: 핵심 방법론인 planning_v4 집중 평가)
python evaluate_medqa.py --modes direct baseline planning_v4 --max-questions 50

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
