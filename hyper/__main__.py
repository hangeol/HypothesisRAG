"""Evaluate HYPER on MedQA or MMLU-Med."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Hugging Face model identifier or local model directory",
    )
    parser.add_argument(
        "--dataset",
        choices=("medqa", "mmlu"),
        default="medqa",
        help="mmlu selects the medical subset (default: medqa)",
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate the first N questions for a quick run"
    )
    parser.add_argument("--output", help="Result directory (default: outputs/<model>/)")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used by the generation model",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory fraction (default: 0.85)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without loading models",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    config = dict(
        base_model=args.model,
        question_dataset=args.dataset,
        max_questions=args.limit or (1273 if args.dataset == "medqa" else 1089),
        total_docs=15,
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        max_model_len=8192,
        max_tokens=2048,
        gpu_mem=args.gpu_memory_utilization,
        vllm_tensor_parallel_size=args.tensor_parallel_size,
        output_dir=args.output,
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from hyper.pipeline import evaluate

    evaluate(**config)


if __name__ == "__main__":
    main()
