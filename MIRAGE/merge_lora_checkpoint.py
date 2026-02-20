#!/usr/bin/env python3
"""
Merge LoRA adapter with base model to create a complete fine-tuned model
"""

import os
import sys
import json
import argparse
from pathlib import Path

def merge_lora_adapter(adapter_path, output_path, base_model_override=None):
    """
    Merge LoRA adapter with its base model
    
    Args:
        adapter_path (str): Path to LoRA adapter checkpoint
        output_path (str): Path where merged model will be saved
        base_model_override (str): Override base model path if needed
    """
    print("=" * 80)
    print("LoRA Adapter + Base Model Merger")
    print("=" * 80)
    
    # Validate adapter path
    if not os.path.exists(adapter_path):
        print(f"✗ Error: Adapter path '{adapter_path}' does not exist!")
        return False
    
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"✗ Error: No adapter_config.json found in '{adapter_path}'")
        return False
    
    print(f"✓ Found LoRA adapter at: {adapter_path}")
    
    # Read adapter configuration
    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        print("✓ Loaded adapter configuration")
    except Exception as e:
        print(f"✗ Error reading adapter config: {e}")
        return False
    
    # Get base model path
    base_model_path = base_model_override or adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        print("✗ Error: No base model path found in adapter config")
        return False
    
    print(f"✓ Base model: {base_model_path}")
    print(f"✓ Output path: {output_path}")
    
    try:
        # Import required libraries
        print("\nLoading libraries...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        print("✓ Libraries loaded")
        
        # Load base model and tokenizer
        print(f"\nLoading base model: {base_model_path}")
        print("This may take a few minutes...")
        
        # Try different base model options if the quantized one fails
        base_models_to_try = [
            base_model_path,
            "google/medgemma-4b-it",  # Original non-quantized model
            "unsloth/medgemma-4b-it-bnb-4bit",  # Alternative quantized
        ]
        
        base_model = None
        for model_path in base_models_to_try:
            try:
                print(f"  → Trying: {model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_4bit=False,  # Disable quantization for merging
                    load_in_8bit=False   # Disable quantization for merging
                )
                print(f"  ✓ Successfully loaded: {model_path}")
                base_model_path = model_path  # Update the path for tokenizer
                break
            except Exception as e:
                print(f"  ✗ Failed with {model_path}: {str(e)[:100]}...")
                continue
        
        if base_model is None:
            raise Exception("Failed to load any compatible base model")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("✓ Base model loaded")
        
        # Load LoRA adapter
        print(f"\nLoading LoRA adapter: {adapter_path}")
        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16
        )
        print("✓ LoRA adapter loaded")
        
        # Merge adapter with base model
        print("\nMerging LoRA adapter with base model...")
        merged_model = model_with_adapter.merge_and_unload()
        print("✓ Models merged successfully")
        
        # Save merged model
        print(f"\nSaving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        merged_model.save_pretrained(
            output_path,
            torch_dtype=torch.bfloat16,
            safe_serialization=True
        )
        
        tokenizer.save_pretrained(output_path)
        print("✓ Merged model saved")
        
        # Create a simple config file to mark this as a merged model
        merge_info = {
            "merged_from": {
                "base_model": base_model_path,
                "adapter_path": adapter_path,
                "adapter_config": adapter_config
            },
            "merge_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        }
        
        with open(os.path.join(output_path, "merge_info.json"), 'w') as f:
            json.dump(merge_info, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✅ MERGE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Merged model saved to: {output_path}")
        print("You can now use this path with vLLM or any other inference engine.")
        print("=" * 80)
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing required library: {e}")
        print("Please install required packages:")
        print("pip install transformers peft torch")
        return False
        
    except Exception as e:
        print(f"✗ Error during merge: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Merge LoRA adapter with base model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge checkpoint-250 with its base model
  python merge_lora_checkpoint.py --adapter ./rag_grpo_outputs_20250904_005255/checkpoint-250 --output ./merged_checkpoint_250
  
  # Merge with custom base model
  python merge_lora_checkpoint.py --adapter ./rag_grpo_outputs_20250904_005255/checkpoint-250 --output ./merged_checkpoint_250 --base-model google/medgemma-4b-it
  
  # Then use merged model with test script
  python test_medgemma_medqa_vllm_full.py --rag --model-path ./merged_checkpoint_250
        """
    )
    
    parser.add_argument('--adapter', type=str, required=True,
                       help='Path to LoRA adapter checkpoint directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for merged model')
    parser.add_argument('--base-model', type=str, default=None,
                       help='Override base model path (default: use from adapter config)')
    
    args = parser.parse_args()
    
    # Perform merge
    success = merge_lora_adapter(
        adapter_path=args.adapter,
        output_path=args.output,
        base_model_override=args.base_model
    )
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"python test_medgemma_medqa_vllm_full.py --rag --model-path {args.output}")
        sys.exit(0)
    else:
        print("\n❌ Merge failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
