#!/usr/bin/env python3
"""
Rewriter Model Setup for GRPO Training

Loads a HuggingFace CausalLM with PEFT LoRA adapter for the rewriter role.
Only LoRA parameters are trainable; the rest is frozen.
"""

import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_rewriter_model(
    base_model: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    use_4bit: bool = False,
    use_8bit: bool = False,
    torch_dtype: str = "bfloat16",
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load base model with LoRA adapter for GRPO training.

    Args:
        base_model: HuggingFace model name or path.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout for LoRA layers.
        target_modules: Which modules to apply LoRA to.
            Default: ["q_proj", "k_proj", "v_proj", "o_proj"]
        use_4bit: Use 4-bit quantization (QLoRA).
        use_8bit: Use 8-bit quantization.
        torch_dtype: Model dtype ("bfloat16", "float16", "float32").
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", etc.).

    Returns:
        (model, tokenizer)
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    model_kwargs = {
        "torch_dtype": model_dtype,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Default target modules
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*60}")
    print(f"Model: {base_model}")
    print(f"Total parameters:     {total_params:>15,}")
    print(f"Trainable parameters: {trainable_params:>15,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters:    {frozen_params:>15,}")
    print(f"LoRA rank: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"Target modules: {target_modules}")
    print(f"{'='*60}\n")

    # Verify only LoRA params are trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n in trainable_names if "lora" not in n.lower()]
    if non_lora:
        print(f"⚠ WARNING: Non-LoRA trainable params detected: {non_lora}")
    else:
        print(f"✓ All {len(trainable_names)} trainable params are LoRA layers")

    return model, tokenizer


def get_lora_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """Get a standalone LoRA config for use with TRL's GRPOTrainer.

    GRPOTrainer can accept a peft_config directly, so we expose this.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
