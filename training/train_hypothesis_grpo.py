#!/usr/bin/env python3
"""
GRPO Training Entrypoint for Hypothesis generation.

Uses trl.GRPOTrainer to train ONLY the hypothesis/plan model's LoRA adapter.
Retriever and final-answer generator remain frozen.
"""

import os
import sys
import argparse
import json
import math

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training for Hypothesis generation"
    )

    parser.add_argument(
        "--base_model", type=str, required=True,
        help="HuggingFace model name or path for the base model"
    )
    parser.add_argument(
        "--adapter_out_dir", type=str, default="outputs/hypothesis_grpo_lora",
        help="Directory to save trained LoRA adapter"
    )

    parser.add_argument(
        "--benchmark_path", type=str, default="data/medqa_train.json",
        help="Path to benchmark JSON (default: data/medqa_train.json for training)"
    )
    parser.add_argument(
        "--split", type=str, default="medqa_train",
        help="Dataset split key in benchmark JSON (default: medqa_train)"
    )
    parser.add_argument(
        "--max_questions", type=int, default=None,
        help="Limit dataset size (for debugging)"
    )

    parser.add_argument(
        "--retriever_name", type=str, default="MedCPT",
        help="Retriever name (e.g., MedCPT)"
    )
    parser.add_argument(
        "--corpus_name", type=str, default="Textbooks",
        help="Corpus name (e.g., Textbooks, PubMed)"
    )
    parser.add_argument(
        "--total_docs", type=int, default=15,
        help="Total documents to retrieve per answer"
    )

    parser.add_argument(
        "--group_size", type=int, default=8,
        help="Number of completions sampled per prompt (G in GRPO)"
    )
    parser.add_argument(
        "--steps", type=int, default=1500,
        help="Total training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device training batch size (number of prompts)"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--beta_kl", type=float, default=0.0,
        help="KL regularization coefficient"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=2048,
        help="Max tokens for hypothesis completion"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for hypothesis generation (GRPO rollout)"
    )

    parser.add_argument(
        "--hypothesis_reward_weight", type=float, default=0.5,
        help="Weight for reward from hypothesis-best_guess (default: 0.5)."
    )
    parser.add_argument(
        "--final_answer_reward_weight", type=float, default=0.5,
        help="Weight for reward from final generated answer (default: 0.5)."
    )
    parser.add_argument(
        "--generator_temperature", type=float, default=0.0,
        help="Sampling temperature for frozen generator in reward function"
    )

    parser.add_argument("--use_lora", action="store_true", help="Use LoRA, if not set runs full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    parser.add_argument(
        "--num_vllm_gpus", type=int, default=1,
        help="Number of GPUs to use for vLLM servers"
    )
    parser.add_argument(
        "--generator_tp", type=int, default=1,
        help="Tensor parallel size for frozen generator vLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.4,
        help="GPU memory utilization for vLLM inference instances"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192,
        help="Max model context length for vLLM"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=100,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="hypothesis-grpo",
        help="W&B project name"
    )

    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run 5 questions with G=2, no weight updates — validate pipeline"
    )

    # Compatibility options if you run from run_training.sh with shared flags.
    parser.add_argument("--hypothesis_temperature", type=float, default=0.0)
    parser.add_argument("--planner_tp", type=int, default=1)
    parser.add_argument("--plan_cache_path", type=str, default=None)

    # Prompt version selection (matches evaluate_medqa_v2.py)
    parser.add_argument(
        "--hypothesis_prompt", type=str, default="v6",
        help="Hypothesis prompt version for plan generation (default: v6)"
    )
    parser.add_argument(
        "--rewriting_prompt", type=str, default="v6",
        help="Rewriting prompt version (not directly used, for consistency)"
    )
    parser.add_argument(
        "--generator_prompt", type=str, default="v1",
        help="Generator prompt version for reward function (default: v1)"
    )

    return parser.parse_args()


def check_rag_connection(retriever_name: str, corpus_name: str):
    print("\n[0/3] Checking RAG connection...")
    try:
        from retriever import create_retriever
        retriever_instance = create_retriever(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
        docs, _ = retriever_instance.retrieve("What is a headache?", k=1)
        if not docs:
            print("✗ RAG FAILED: Retrieved empty documents.")
            sys.exit(1)
        print("✓ RAG SUCCESS: Connection verified.")
    except Exception as e:
        print(f"✗ RAG FAILED with exception: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (local_rank == 0)

    if is_main:
        check_rag_connection(args.retriever_name, args.corpus_name)

    from datetime import datetime
    import time

    # Synchronize timestamp across ranks.
    ts_sync_file = os.path.join(args.adapter_out_dir, "_timestamp_sync.txt")
    if is_main:
        os.makedirs(args.adapter_out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(ts_sync_file, "w") as f:
            f.write(timestamp)
    else:
        for _ in range(60):
            if os.path.exists(ts_sync_file):
                with open(ts_sync_file, "r") as f:
                    timestamp = f.read().strip()
                if timestamp:
                    break
            time.sleep(0.5)

    output_dir = os.path.join(args.adapter_out_dir, timestamp)
    hyperparams_path = os.path.join(output_dir, "hyperparameters.json")

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        with open(hyperparams_path, "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        print(f"✓ Saved hyperparameters to {hyperparams_path}")

    if args.dry_run:
        if is_main:
            print("\n" + "=" * 60)
            print("DRY RUN MODE: validating hypothesis pipeline only")
            print("=" * 60 + "\n")
        args.max_questions = args.max_questions or 5
        args.steps = 1
        args.batch_size = 1
        args.group_size = 2
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if args.batch_size * world_size < 1:
            args.batch_size = 1
        generation_batch = args.batch_size * max(world_size, 1)
        if generation_batch % args.group_size != 0:
            args.grad_accum = max(
                1,
                math.ceil(args.group_size / generation_batch),
            )

    from data.medqa_loader import build_hypothesis_grpo_dataset

    if is_main:
        print("\n[1/3] Building hypothesis GRPO dataset...")
    dataset = build_hypothesis_grpo_dataset(
        benchmark_path=args.benchmark_path,
        split=args.split,
        max_questions=args.max_questions,
        hypothesis_prompt_version=args.hypothesis_prompt,
    )

    if is_main:
        print("\n[2/3] Initializing hypothesis reward function...")
    from training.reward import HypothesisRewardFunction

    reward_fn = HypothesisRewardFunction(
        generator_model_name=args.base_model,
        retriever_name=args.retriever_name,
        corpus_name=args.corpus_name,
        total_docs=args.total_docs,
        tensor_parallel_size=args.generator_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        final_reward_weight=args.final_answer_reward_weight,
        hypothesis_reward_weight=args.hypothesis_reward_weight,
        generator_temperature=args.generator_temperature,
        generator_prompt_version=args.generator_prompt,
    )

    if is_main:
        print("\n[3/3] Configuring GRPOTrainer...")
    from trl import GRPOTrainer, GRPOConfig
    from models.rewriter import get_lora_config

    peft_config = get_lora_config(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ) if args.use_lora else None

    report_to = ["tensorboard", "wandb"] if args.use_wandb else "tensorboard"

    training_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        beta=args.beta_kl,
        temperature=args.temperature,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://localhost:18080",
        vllm_server_timeout=600.0,

        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=report_to,
        run_name="hypothesis-grpo",
        seed=args.seed,

        gradient_checkpointing=True,
        optim="adamw_8bit" if not args.use_lora else "adamw_torch",

        remove_unused_columns=False,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=[reward_fn],
        args=training_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("\n" + "=" * 60)
    print("Trainable parameters:")
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print("=" * 60 + "\n")

    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        print("Starting hypothesis training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(output_dir)

    if is_main:
        print(f"\nSaving final model to {output_dir}...")

        metrics = reward_fn.get_metrics()
        final_info = vars(args).copy()
        final_info["output_dir"] = output_dir
        final_info["reward_metrics"] = metrics
        with open(hyperparams_path, "w") as f:
            json.dump(final_info, f, indent=2, ensure_ascii=False)
        print(f"✓ Updated hyperparameters with reward metrics to {hyperparams_path}")

        print("\nReward function metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        print("\n✓ Hypothesis training complete!")


if __name__ == "__main__":
    main()
