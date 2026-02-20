#!/usr/bin/env python3
"""
Test script for Llama 3.2-1B with vLLM
Tests various memory optimization settings to get vLLM working
"""

import os
import sys
sys.path.append("src")

def test_vllm_llama32():
    """Test Llama 3.2-1B with vLLM using various memory settings"""
    
    print("=" * 80)
    print("Testing Llama 3.2-1B with vLLM")
    print("=" * 80)
    
    # Test different memory settings
    memory_settings = [
        {"VLLM_GPU_MEMORY_UTILIZATION": "0.5", "VLLM_MAX_MODEL_LEN": "1024"},
        {"VLLM_GPU_MEMORY_UTILIZATION": "0.6", "VLLM_MAX_MODEL_LEN": "1024"},
        {"VLLM_GPU_MEMORY_UTILIZATION": "0.7", "VLLM_MAX_MODEL_LEN": "1024"},
        {"VLLM_GPU_MEMORY_UTILIZATION": "0.8", "VLLM_MAX_MODEL_LEN": "1024"},
        {"VLLM_GPU_MEMORY_UTILIZATION": "0.5", "VLLM_MAX_MODEL_LEN": "512"},
        {"VLLM_GPU_MEMORY_UTILIZATION": "0.6", "VLLM_MAX_MODEL_LEN": "512"},
    ]
    
    for i, settings in enumerate(memory_settings):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: GPU Memory Utilization = {settings['VLLM_GPU_MEMORY_UTILIZATION']}, Max Model Len = {settings['VLLM_MAX_MODEL_LEN']}")
        print(f"{'='*60}")
        
        try:
            # Set environment variables
            for key, value in settings.items():
                os.environ[key] = value
                print(f"Set {key} = {value}")
            
            # Also set the allow long max model len
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
            
            print("Initializing MedRAG with vLLM...")
            
            from medrag import MedRAG
            
            medrag = MedRAG(
                llm_name="meta-llama/Llama-3.2-1B-Instruct",
                use_vllm=True,
                cache_dir="./cache"
            )
            
            print("✅ vLLM initialization successful!")
            
            # Test simple generation
            prompt = "Hello, how are you today?"
            print(f"Testing with prompt: '{prompt}'")
            
            response = medrag.generate(prompt)
            print(f"Response: {response}")
            
            # Test medical question
            medical_prompt = "What is the most common cause of community-acquired pneumonia?"
            print(f"\nTesting medical question: '{medical_prompt}'")
            
            medical_response = medrag.generate(medical_prompt)
            print(f"Medical response: {medical_response}")
            
            print("\n🎉 SUCCESS! vLLM with Llama 3.2-1B is working!")
            return True
            
        except Exception as e:
            print(f"❌ Failed with settings {settings}: {e}")
            print("Trying next configuration...")
            continue
    
    print("\n❌ All vLLM configurations failed. Trying alternative approach...")
    return False

def test_vllm_direct():
    """Test vLLM directly without MedRAG wrapper"""
    
    print("\n" + "="*80)
    print("Testing vLLM Directly (Bypassing MedRAG)")
    print("="*80)
    
    try:
        from vllm import LLM, SamplingParams
        
        # Try with very conservative memory settings
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.4"
        os.environ["VLLM_MAX_MODEL_LEN"] = "512"
        
        print("Loading Llama 3.2-1B directly with vLLM...")
        
        llm = LLM(
            model="meta-llama/Llama-3.2-1B-Instruct",
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=512,
            disable_log_stats=True
        )
        
        print("✅ vLLM direct loading successful!")
        
        # Test generation
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        prompt = "Hello, how are you today?"
        
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        print("\n🎉 SUCCESS! Direct vLLM with Llama 3.2-1B is working!")
        return True
        
    except Exception as e:
        print(f"❌ Direct vLLM failed: {e}")
        return False

def test_smaller_model():
    """Test with an even smaller model if Llama 3.2-1B fails"""
    
    print("\n" + "="*80)
    print("Testing with Smaller Model (Llama 3.2-1B Alternative)")
    print("="*80)
    
    try:
        from vllm import LLM, SamplingParams
        
        # Use a smaller model
        model_name = "microsoft/DialoGPT-small"  # Much smaller than Llama 3.2-1B
        
        print(f"Loading {model_name} with vLLM...")
        
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=1024,
            disable_log_stats=True
        )
        
        print("✅ vLLM with smaller model successful!")
        
        # Test generation
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        prompt = "Hello, how are you today?"
        
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        print("\n🎉 SUCCESS! vLLM with smaller model is working!")
        return True
        
    except Exception as e:
        print(f"❌ Smaller model also failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting vLLM Llama 3.2-1B tests...")
    print()
    
    # Test 1: MedRAG wrapper with vLLM
    if test_vllm_llama32():
        print("\n🎉 SUCCESS: MedRAG with vLLM and Llama 3.2-1B is working!")
        exit(0)
    
    # Test 2: Direct vLLM
    if test_vllm_direct():
        print("\n🎉 SUCCESS: Direct vLLM with Llama 3.2-1B is working!")
        exit(0)
    
    # Test 3: Smaller model with vLLM
    if test_smaller_model():
        print("\n🎉 SUCCESS: vLLM with smaller model is working!")
        exit(0)
    
    print("\n❌ All vLLM tests failed. Please check GPU memory and try again.")
    print("Consider:")
    print("1. Closing other GPU applications")
    print("2. Using a smaller model")
    print("3. Reducing max_model_len")
    print("4. Reducing GPU memory utilization")
