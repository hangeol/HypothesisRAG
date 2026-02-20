#!/usr/bin/env python3
"""
GRPO Training Entrypoint for Rewriter

Uses trl.GRPOTrainer to train ONLY the Rewriter's LoRA adapter.
Planner (plan pre-generation via vLLM) and Generator (reward via vLLM)
remain frozen throughout.

Usage:
    python training/train_rewriter_grpo.py \
        --base_model google/gemma-2-9b-it \
        --adapter_out_dir outputs/rewriter_grpo_lora \
        --retriever_name MedCPT \
        --corpus_name Textbooks \
        --total_docs 15 \
        --group_size 8 \
        --steps 2000 \
        --beta_kl 0.02 \
        --reward_type acc \
        --seed 42
"""

import os
import sys
import argparse
import json
import warnings

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO training for Rewriter (plan-conditioned query generator)"
    )

    # Model
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="HuggingFace model name or path for the base model"
    )
    parser.add_argument(
        "--adapter_out_dir", type=str, default="outputs/rewriter_grpo_lora",
        help="Directory to save trained LoRA adapter"
    )

    # Dataset
    parser.add_argument(
        "--benchmark_path", type=str, default=None,
        help="Path to MIRAGE benchmark.json"
    )
    parser.add_argument(
        "--split", type=str, default="medqa",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max_questions", type=int, default=None,
        help="Limit dataset size (for debugging)"
    )
    parser.add_argument(
        "--plan_cache_path", type=str, default=None,
        help="Path to cache pre-generated plans"
    )

    # Retriever
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

    # GRPO hyperparameters
    parser.add_argument(
        "--group_size", type=int, default=8,
        help="Number of completions sampled per prompt (G in GRPO)"
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
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
        "--beta_kl", type=float, default=0.02,
        help="KL regularization coefficient"
    )
    parser.add_argument(
        "--epsilon_clip", type=float, default=0.2,
        help="PPO clip range"
    )
    parser.add_argument(
        "--reward_type", type=str, default="acc", choices=["acc"],
        help="Reward function type"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=256,
        help="Max tokens for rewriter completion"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=2048,
        help="Max tokens for prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for rewriter during training"
    )

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # vLLM settings (for reward function's generator)
    parser.add_argument(
        "--generator_tp", type=int, default=1,
        help="Tensor parallel size for frozen generator vLLM"
    )
    parser.add_argument(
        "--planner_tp", type=int, default=1,
        help="Tensor parallel size for frozen planner vLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.4,
        help="GPU memory utilization for vLLM inference instances"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=8192,
        help="Max model context length for vLLM"
    )

    # Logging / checkpoints
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--logging_steps", type=int, default=1,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=200,
        help="Save checkpoint every N steps"
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
        "--wandb_project", type=str, default="rewriter-grpo",
        help="W&B project name"
    )

    # Quantization
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")

    # Dry run
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run 5 questions with G=2, no weight updates — validate pipeline"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Dry run overrides
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE: validating pipeline only")
        print("=" * 60 + "\n")
        args.max_questions = args.max_questions or 5
        args.group_size = 2
        args.steps = 1
        args.batch_size = 1
        args.grad_accum = 1

    # ========================================================================
    # 1. Build the dataset (pre-generate plans via vLLM)
    # ========================================================================
    print("\n[1/3] Building dataset with pre-generated plans...")
    from data.medqa_loader import build_grpo_dataset

    dataset = build_grpo_dataset(
        benchmark_path=args.benchmark_path,
        split=args.split,
        base_model=args.base_model,
        max_questions=args.max_questions,
        plan_cache_path=args.plan_cache_path,
        tensor_parallel_size=args.planner_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # ========================================================================
    # 2. Initialize reward function (retriever + frozen generator via vLLM)
    # ========================================================================
    print("\n[2/3] Initializing reward function...")
    from training.reward import RewriterRewardFunction

    reward_fn = RewriterRewardFunction(
        generator_model_name=args.base_model,
        retriever_name=args.retriever_name,
        corpus_name=args.corpus_name,
        total_docs=args.total_docs,
        reward_type=args.reward_type,
        tensor_parallel_size=args.generator_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # ========================================================================
    # 3. Configure and run GRPOTrainer
    # ========================================================================
    print("\n[3/3] Configuring GRPOTrainer...")
    from trl import GRPOTrainer, GRPOConfig
    from models.rewriter import get_lora_config

    # LoRA config (GRPOTrainer handles PEFT internally)
    peft_config = get_lora_config(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Report to
    report_to = "wandb" if args.use_wandb else "none"

    # GRPOConfig
    training_config = GRPOConfig(
        output_dir=args.adapter_out_dir,

        # GRPO specific
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta_kl,
        # temperature is set via generation_kwargs below

        # Training
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,

        # Logging / checkpoints
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=report_to,
        run_name=f"rewriter-grpo-{args.reward_type}",
        seed=args.seed,

        # Misc
        remove_unused_columns=False,
        log_completions=True,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=[reward_fn],
        args=training_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Print model summary
    print("\n" + "=" * 60)
    print("Trainable parameters:")
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    trainable_names = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    non_lora = [n for n in trainable_names if "lora" not in n.lower()]
    if non_lora:
        print(f"  ⚠ Non-LoRA trainable params: {non_lora}")
    else:
        print(f"  ✓ All {len(trainable_names)} trainable params are LoRA layers")
    print("=" * 60 + "\n")

    # Run training
    print("Starting training...")
    trainer.train()

    # Save final adapter
    print(f"\nSaving final adapter to {args.adapter_out_dir}...")
    trainer.save_model(args.adapter_out_dir)

    # Save training args
    args_path = os.path.join(args.adapter_out_dir, "training_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Saved training args to {args_path}")

    # Print reward metrics
    metrics = reward_fn.get_metrics()
    print(f"\nReward function metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
