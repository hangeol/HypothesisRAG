# GRPO Training for Rewriter

Train the Rewriter (query generator) with Group Relative Policy Optimization using `trl.GRPOTrainer`, while keeping Planner and Generator frozen.

## Architecture

```
Question + Options
       │
       ▼
  Frozen Planner (vLLM)  ──→  Plan JSON
       │
       ▼
  ┌─ Rewriter (LoRA) ──→  3 Search Queries ─┐ ← trained
  │                                          │
  │  (G completions sampled per prompt)      │
  │                                          │
  │  Retriever (MedCPT)  ←──────────────────┘
  │       │
  │       ▼
  │  Retrieved Documents
  │       │
  │       ▼
  │  Frozen Generator (vLLM)  ──→  Answer
  │       │
  │       ▼
  └── Reward = 1 if correct, 0 if wrong
```

Only the Rewriter's LoRA adapter parameters are updated. All other model weights remain frozen.

## Quick Start

### 1. Install Dependencies

```bash
pip install trl peft transformers datasets vllm accelerate bitsandbytes wandb
```

### 2. Train

```bash
python training/train_rewriter_grpo.py \
    --base_model google/gemma-2-9b-it \
    --adapter_out_dir outputs/rewriter_grpo_lora \
    --retriever_name MedCPT \
    --corpus_name Textbooks \
    --group_size 8 \
    --steps 2000 \
    --beta_kl 0.02 \
    --seed 42
```

### 3. Dry Run (validate pipeline without GPU training)

```bash
python training/train_rewriter_grpo.py \
    --base_model google/gemma-2-9b-it \
    --dry_run
```

### 4. Evaluate

Compare baseline rewriter vs GRPO-trained rewriter:

```bash
# Standalone comparison script
python scripts/eval_grpo.py \
    --adapter_path outputs/rewriter_grpo_lora \
    --base_model google/gemma-2-9b-it \
    -n 100

# Or within the main evaluation framework
python evaluate_medqa.py \
    --modes planning_v4 planning_v4_grpo \
    --llm-provider vllm \
    --model google/gemma-2-9b-it \
    --rewriter-adapter-path outputs/rewriter_grpo_lora \
    -n 100
```

## File Structure

```
training/
├── __init__.py
├── train_rewriter_grpo.py   # Main entrypoint (uses trl.GRPOTrainer)
└── reward.py                # Custom reward function (retriever + vLLM generator)

models/
├── __init__.py
└── rewriter.py              # LoRA configuration

data/
├── __init__.py
└── medqa_loader.py          # Dataset builder with vLLM plan pre-generation

scripts/
└── eval_grpo.py             # Standalone baseline vs GRPO comparison
```

## Key Design Decisions

1. **TRL GRPOTrainer** — No manual GRPO implementation; uses the official `trl.GRPOTrainer` directly
2. **vLLM for frozen inference** — Planner (plan pre-generation) and Generator (reward computation) both use vLLM for fast batched inference
3. **Plan caching** — Plans are pre-generated before training and cached to disk to avoid regeneration each epoch
4. **Reward = accuracy** — Binary reward: 1.0 if the generator's answer matches the gold answer, 0.0 otherwise
5. **Single base model** — Planner, Generator, and Rewriter share the same base model weights; only the Rewriter has a LoRA adapter

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Completions per prompt (G) |
| `beta_kl` | 0.02 | KL regularization |
| `epsilon_clip` | 0.2 | PPO clip range |
| `learning_rate` | 5e-6 | AdamW LR |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |
| `max_completion_length` | 256 | Max rewriter output tokens |
| `warmup_ratio` | 0.05 | LR warmup |
