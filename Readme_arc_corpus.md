# ARC Corpus Setup & Evaluation Guide

이 문서는 `arc_corpus`를 `evaluate_arc_challenge.py` 스크립트 및 `MIRAGE` 기반 시스템과 함께 사용하기 위해 다운로드, 초기 인덱싱, 그리고 발생할 수 있는 주요 에러들을 해결하는 방법을 설명합니다.

---

## 1. Corpus 데이터 준비 (다운로드 및 배치)
wget https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip

ARC Challenge를 위한 `arc_corpus` 데이터 세트는 여러 개의 큰 단위(분할된 `.jsonl` 파일들)로 제공됩니다. 대략 15개 정도의 청크 파일이 있으며 전체 용량은 2.0GB ~ 2.5GB 수준입니다. 

**다운로드 후 디렉토리 구조 맞추기**
1. 압축을 풀거나 다운로드 받은 청크 파일들(`arc_chunk_000.jsonl` ~ `arc_chunk_014.jsonl` 등)을 다음 경로에 올바르게 복사해야 합니다.
   ```bash
   MIRAGE/MedRAG/corpus/arc_corpus/chunk/
   ```
2. 각 라인은 유효한 JSON 객체여야 하며 `{ "id": "...", "title": "...", "contents": "..." }` 등의 포맷을 가집니다.

---

## 2. 초기 인덱싱 (Embedding)

항목들을 검색할 수 있도록 ARC Challenge 평가 스크립트를 처음 실행할 때, 로컬에 임베딩된 인덱스가 없다면 `facebook/contriever`와 같은 검색기를 통해 **전체 청크 파일에 대한 임베딩**을 새로 생성합니다.

- **명령어 예시:**
  ```bash
  python evaluate_arc_challenge.py --max-questions 10
  ```
- **소요 시간:** 
  15개 파일 처리 시, GPU (CUDA) 환경을 기준으로 파일당 약 10분씩 **총 2시간 30분 정도가 소요**됩니다. 임베딩이 비정상적으로 느리다면 `nvidia-smi`를 통해 GPU 할당 여부를 확인하세요.
- 인덱싱 작업은 **단 한 번만 수행**되며, 처리가 끝나면 `arc_corpus_id2text.json` 및 `arc_corpus_id2path.json` 파일이 캐싱되어 다음 실행부터는 즉시 검색과 평가를 진행할 수 있습니다.

---

## 3. 주요 트러블슈팅 (경험 기반 버그 픽스)

초기 설정이나 평가 중 다음과 같은 치명적인 문제들이 발생할 수 있습니다. 스크립트 실행 전에 미리 조치하면 훨씬 원활한 진행이 가능합니다.

### A. Document Retrieval 실패 버그 (`KeyError: arc_chunk_XXX`)
- **증상:** 평가 시 검색 결과인 `accuracy`가 0.00%로 나오며 `Warning: Retrieval failed for query ...` 에러 메시지가 무한 반복 출력됩니다.
- **원인:** 모델 기반 검색기(Contriever 등)가 쿼리 결과를 반환할 때 파일 이름과 라인 번호를 조합한 `Pseudo-ID`(예: `arc_chunk_008_168177`)를 반환합니다. 하지만, 딕셔너리를 구축하는 `DocExtracter` 클래스는 오직 데이터 내부의 JSON 고유 `id`만으로 딕셔너리를 만들기 때문입니다.
- **해결책 (`MIRAGE/MedRAG/src/utils.py` 수정):**
  `utils.py` 내 `DocExtracter` 초기화 로직을 수정하여 원래 JSON의 `id`와 더불어 검색 시스템이 사용하는 모델 ID(`pseudo_id`)도 함께 캐시 맵핑에 추가해야 합니다.
  
  ```python
  # 기존 로직 아래에 다음 pseudo_id 코드를 추가
  if "id" in item:
      self.dict[item["id"]] = ...
  
  # 추가할 부분:
  pseudo_id = f'{fname.replace(".jsonl", "")}_{i}'
  self.dict[pseudo_id] = ...
  ```
  *(수정 후에는 이미 생성된 잘못된 딕셔너리 캐시 파일 `*_id2text.json`과 `*_id2path.json`을 삭제 후 다시 스크립트를 실행해야 정상적으로 딕셔너리를 구축합니다.)*

### B. 비동기 HTTP 요청 `brotli` 디코딩 에러 (`Can not decode content-encoding: br`)
- **증상:** LLM 답변 생성(Phase 4) 중에 `Error: 400, message: Can not decode content-encoding: br` 에러가 응답으로 반환되며 API가 실패합니다.
- **원인:** Python 패키지 중 `brotli`가 설치되어 있으면 `aiohttp` 세션이 기본적으로 요청 헤더에 `Accept-Encoding: br`을 포함합니다. 그러나 특정 API 환경이나 프록시가 이를 제대로 인코딩하지 못한 브로틀리 포맷을 반환하여 디코딩에 충돌이 발생합니다.
- **해결책 (`evaluate_arc_challenge.py` 에러 픽스):**
  해당 스크립트 내부에서 `aiohttp.ClientSession` 객체를 생성하는 부분을 모두 찾아서, gzip만 사용하도록 헤더를 강제로 명시합니다.
  ```python
  # 수정 전
  async with aiohttp.ClientSession(connector=connector) as session:

  # 수정 후
  async with aiohttp.ClientSession(connector=connector, headers={"Accept-Encoding": "gzip, deflate"}) as session:
  ```

### C. OpenAI API Key 환경 변수 관리
- 생성 모델 평가에서 OpenAI API를 사용하기 때문에 터미널(또는 `~/.bashrc`)에 `OPENAI_API_KEY` 환경 변수가 제대로 활성화되어 있어야 에러가 발생하지 않습니다.
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

---

위 절차대로 `.jsonl` 파일을 준비하시고, 두 가지 핵심적인 버그(`utils.py` 및 `evaluate_arc_challenge.py` 내)를 수정하면, 빠르고 정확하게 데이터를 검색하고 ARC Challenge 문제를 평가(최대 90% 이상의 정답률)할 수 있습니다.
