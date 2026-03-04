# JSON 구조 및 파싱 검증 결과

## 1. 입력 파일 구조

### 파일 형식
- **Direct**: `results/medqa_direct_full.json`
- **CoT**: `results/medqa_cot_full.json`

### 최상위 구조
```json
{
  "summary": {...},
  "results": [
    {
      "question_id": "...",
      "question": "...",
      "options": {...},
      "correct_answer": "...",
      "modes": {
        "direct": {...},
        "cot": {...}
      }
    }
  ]
}
```

### 각 항목의 구조
```json
{
  "question_id": "1",  // 또는 숫자
  "question": "...",
  "options": {
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  },
  "correct_answer": "D",
  "modes": {
    "direct": {
      "num_queries": ...,
      "num_docs": ...,
      "queries": [...],
      "plan": [...],
      "retrieved_docs": [
        {
          "id": "...",
          "title": "...",
          "content": "...",
          "query_trace": {...},
          "fused_score": ...
        }
      ],
      "raw_response": "...",
      "predicted_answer": "A",
      "is_correct": true/false
    },
    "cot": {
      "num_queries": ...,
      "num_docs": ...,
      "predicted_answer": "A",
      "is_correct": true/false,
      "retrieved_docs": [...] // 또는 null
    }
  }
}
```

## 2. 파싱 로직 검증

### ✅ `load_json_file` 함수
- **구조 처리**: `{"results": [...]}` 또는 `[...]` 형식 모두 처리
- **question_id 추출**: `question_id`, `id`, `qid` 순서로 시도
- **에러 처리**: `json.JSONDecodeError` 예외 처리 추가됨

### ✅ `get_mode_block` 함수
- **경로**: `item["modes"][mode]` 접근
- **안전성**: 타입 체크 및 기본값 반환

### ✅ `get_predicted_answer` 함수
- **우선순위**: `predicted_answer` > `prediction` > `answer_choice` > `pred`
- **fallback**: `raw_response`에서 JSON 파싱 시도
- **에러 처리**: 예외 발생 시 `None` 반환

### ✅ `get_correctness` 함수
- **우선순위**: `modes[mode]["is_correct"]` 직접 확인
- **fallback**: `predicted_answer`와 `correct_answer` 비교

### ✅ `get_retrieved_docs` 함수
- **구조 처리**: 리스트 또는 None 처리
- **필드 매핑**: 
  - `id` 또는 `doc_id` → `id`
  - `content`, `text`, `chunk` → `content`
  - `title` → `title`
- **문서 구조**: `{"id": "...", "title": "...", "content": "..."}` 형식으로 정규화

### ✅ `normalize_options` 함수
- **dict 처리**: `{"A": "...", "B": "..."}` 형식
- **list 처리**: `["...", "..."]` → `{"A": "...", "B": "..."}` 변환

## 3. 발견된 문제점

### ⚠️ API 응답 파싱 실패
출력 파일(`failure_v2_out/set2_v2.jsonl`)에서 많은 항목이 다음 오류를 보임:
```json
"audit": {"_error": "json_parse_failed", "_raw": ""}
```

**원인 분석**:
1. `output_text`가 비어있거나 None인 경우
2. API 응답이 JSON 형식이 아닌 경우
3. `extract_json_from_text` 함수가 JSON을 추출하지 못한 경우

**현재 처리**:
- `extract_json_from_text`가 `None`을 반환하면 `{"_error": "json_parse_failed", "_raw": out_text[:2000]}` 반환
- 하지만 `_raw`가 빈 문자열인 경우가 많음 → `output_text` 자체가 비어있을 가능성

### ✅ 입력 파일 파싱은 정상
테스트 결과 모든 입력 파일 파싱이 정상적으로 작동:
- ✓ 1272개 항목 모두 파싱 성공
- ✓ question_id, options, modes 등 모든 필드 정상 추출
- ✓ retrieved_docs 구조 정상 처리

## 4. 권장 사항

### 개선 사항
1. **API 응답 로깅 강화**: `output_text`가 비어있는 경우 원본 응답 전체를 로깅
2. **에러 분류**: `json_parse_failed`와 `empty_response` 구분
3. **재시도 로직**: 빈 응답인 경우 재시도 고려

### 현재 상태
- ✅ 입력 JSON 파일 파싱: **정상 작동**
- ✅ 데이터 구조 추출: **정상 작동**
- ⚠️ API 응답 파싱: **일부 실패** (API 응답 문제로 추정)
