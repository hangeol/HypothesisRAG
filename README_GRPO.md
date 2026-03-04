# README_GRPO (Archived)

이 문서는 과거 버전의 GRPO 가이드였습니다.  
최신 경로/명령은 메인 README를 사용하세요.

- 최신 통합 문서: `README.md`
- 학습 래퍼: `scripts/train/run_training.sh`
- 평가 스크립트: `scripts/evaluate/evaluate.py`
- 체크포인트 평가: `scripts/evaluate/evaluate_checkpoints.py`
- 분석 스크립트: `scripts/analysis/*`

## 빠른 예시

```bash
bash scripts/train/run_training.sh --mode rewriter --gpus 0,1,2 --num_vllm_gpus 1 --num_train_gpus 2
python scripts/evaluate/evaluate.py --mode hypothesis --llm-provider vllm --model Qwen/Qwen3-4B-Instruct-2507
```
