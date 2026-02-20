#!/usr/bin/env python3
"""
Test script for MedQA RAG with smaller models
Tests both vLLM and transformers pipeline approaches
"""

import os
import sys
sys.path.append("src")

from medrag import MedRAG

def test_medqa_rag():
    """Test MedQA RAG with smaller models"""
    
    # Test question from MedQA dataset
    question = "A 45-year-old woman presents with a 2-month history of progressive dyspnea and fatigue. She has no chest pain, cough, or fever. Physical examination reveals a regular heart rate of 110/min, blood pressure 90/60 mmHg, and oxygen saturation 92% on room air. Cardiac auscultation reveals a loud S1 and a holosystolic murmur at the apex radiating to the axilla. The most likely diagnosis is:"
    
    options = {
        "A": "Mitral stenosis",
        "B": "Mitral regurgitation", 
        "C": "Aortic stenosis",
        "D": "Aortic regurgitation"
    }
    
    print("=" * 80)
    print("Testing MedQA RAG with smaller models")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Options: {options}")
    print()
    
    # Test 1: Transformers Pipeline (Original)
    print("Test 1: Transformers Pipeline (Original)")
    print("-" * 50)
    try:
        # Use a smaller, more accessible model
        medrag = MedRAG(
            llm_name="microsoft/DialoGPT-small",  # Much smaller model
            use_vllm=False,
            cache_dir="./cache"
        )
        
        # Simple CoT test first
        prompt = f"Question: {question}\nOptions: {options}\nPlease think step by step and choose the correct answer."
        response = medrag.generate(prompt)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error with transformers pipeline: {e}")
        print("This is expected if the model is not available or there are memory issues")
    
    print()
    
    # Test 2: vLLM (Faster Inference)
    print("Test 2: vLLM (Faster Inference)")
    print("-" * 50)
    try:
        # Set environment variables to allow longer max model length and reduce memory usage
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.7"  # Reduce from 0.9 to 0.7
        
        medrag_vllm = MedRAG(
            llm_name="microsoft/DialoGPT-small",  # Much smaller model
            use_vllm=True,
            cache_dir="./cache"
        )
        
        prompt = f"Question: {question}\nOptions: {options}\nPlease think step by step and choose the correct answer."
        response = medrag_vllm.generate(prompt)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error with vLLM: {e}")
        print("vLLM may not be installed or there may be compatibility issues")
    
    print()
    print("=" * 80)
    print("Test completed!")
    print("=" * 80)

def test_simple_cot():
    """Test simple Chain-of-Thought with smaller models"""
    
    question = "What is the most common cause of community-acquired pneumonia?"
    options = {
        "A": "Streptococcus pneumoniae",
        "B": "Haemophilus influenzae", 
        "C": "Mycoplasma pneumoniae",
        "D": "Legionella pneumophila"
    }
    
    print("=" * 80)
    print("Testing Simple CoT with smaller models")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Options: {options}")
    print()
    
    try:
        # Use a smaller, more accessible model
        medrag = MedRAG(
            llm_name="microsoft/DialoGPT-small",  # Much smaller model
            use_vllm=False,
            cache_dir="./cache"
        )
        
        # Create a more specific prompt for better response
        prompt = f"Question: {question}\n\nOptions:\nA) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}\n\nPlease think step by step and choose the correct answer. Explain your reasoning."
        
        print("Generating response...")
        response = medrag.generate(prompt)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error with CoT: {e}")

def test_basic_generation():
    """Test basic text generation without medical content"""
    
    print("=" * 80)
    print("Testing Basic Text Generation")
    print("=" * 80)
    
    try:
        medrag = MedRAG(
            llm_name="microsoft/DialoGPT-small",
            use_vllm=False,
            cache_dir="./cache"
        )
        
        simple_prompt = "Hello, how are you today?"
        print(f"Prompt: {simple_prompt}")
        
        response = medrag.generate(simple_prompt)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error with basic generation: {e}")

def test_direct_transformers():
    """Test direct transformers pipeline to debug response issues"""
    
    print("=" * 80)
    print("Testing Direct Transformers Pipeline")
    print("=" * 80)
    
    try:
        import transformers
        import torch
        
        # Load model and tokenizer directly
        model_name = "microsoft/DialoGPT-small"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Test simple generation
        prompt = "Hello, how are you today?"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Full response: {response}")
        print(f"Generated part: {response[len(prompt):]}")
        
    except Exception as e:
        print(f"Error with direct transformers: {e}")

def test_medrag_debug():
    """Debug MedRAG generate method to understand why responses are empty"""
    
    print("=" * 80)
    print("Debugging MedRAG Generate Method")
    print("=" * 80)
    
    try:
        medrag = MedRAG(
            llm_name="microsoft/DialoGPT-small",
            use_vllm=False,
            cache_dir="./cache"
        )
        
        # Test with a very simple prompt
        prompt = "Say hello"
        print(f"Testing with prompt: '{prompt}'")
        
        # Check if the model is properly loaded
        print(f"Model type: {type(medrag.model)}")
        if hasattr(medrag, 'model'):
            print(f"Model loaded: {medrag.model is not None}")
        
        # Try to generate
        print("Calling generate method...")
        response = medrag.generate(prompt)
        print(f"Raw response: {repr(response)}")
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(response) if response else 0}")
        
    except Exception as e:
        print(f"Error in debug test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting MedRAG tests with smaller models...")
    print()
    
    # Test direct transformers first (debug response issues)
    test_direct_transformers()
    print()
    
    # Debug MedRAG generate method
    test_medrag_debug()
    print()
    
    # Test basic generation (simplest test)
    test_basic_generation()
    print()
    
    # Test simple CoT (faster, requires fewer resources)
    test_simple_cot()
    print()
    
    # Test RAG (slower, requires more resources)
    test_medqa_rag()
