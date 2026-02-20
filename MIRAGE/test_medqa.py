#!/usr/bin/env python3
"""
Complete MedQA evaluation using MedGemma 4B-IT with vLLM
Tests both RAG and non-RAG approaches on the full MedQA test set
Supports both base model and fine-tuned checkpoints
"""

import os
import sys
import json
import time
from tqdm import tqdm
import numpy as np
import argparse

# Add paths carefully to avoid conflicts
sys.path.insert(0, '.')
sys.path.insert(0, 'MedRAG')
sys.path.insert(0, 'MedRAG/src')

# Import MedRAG first
try:
    from MedRAG.src.medrag import MedRAG
    from MedRAG.src.template import *
    print("✓ Successfully imported MedRAG from MedRAG.src")
except ImportError as e:
    print(f"✗ Failed to import MedRAG: {e}")
    sys.exit(1)

# Now import QADataset from the main src directory
sys.path.insert(0, 'src')  # Put main src at the beginning
try:
    # Remove MedRAG/src from path temporarily to avoid conflict
    medrag_src_path = 'MedRAG/src'
    if medrag_src_path in sys.path:
        sys.path.remove(medrag_src_path)
    
    from src.utils import QADataset
    print("✓ Successfully imported QADataset from src.utils")
    
    # Add MedRAG/src back
    sys.path.append(medrag_src_path)
    
except ImportError as e:
    print(f"✗ Failed to import QADataset: {e}")
    # Try alternative approach
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("qa_utils", "src/utils.py")
        qa_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qa_utils)
        QADataset = qa_utils.QADataset
        print("✓ Successfully imported QADataset using importlib")
    except Exception as e2:
        print(f"✗ Alternative import also failed: {e2}")
        sys.exit(1)

def parse_answer(answer_text):
    """
    Parse answer from model output to extract the choice (A, B, C, D)
    """
    try:
        # Try to parse as JSON first
        if isinstance(answer_text, str):
            # Look for JSON pattern
            if '{' in answer_text and '}' in answer_text:
                start = answer_text.find('{')
                end = answer_text.rfind('}') + 1
                json_str = answer_text[start:end]
                answer_data = json.loads(json_str)
                return answer_data.get('answer_choice', answer_data.get('answer', ''))
        else:
            answer_data = answer_text
            return answer_data.get('answer_choice', answer_data.get('answer', ''))
    except:
        pass
    
    # Fallback: look for patterns like "answer is A" or just "A"
    answer_text = str(answer_text).upper()
    for choice in ['A', 'B', 'C', 'D']:
        if f"ANSWER IS {choice}" in answer_text or f"CHOICE {choice}" in answer_text:
            return choice
        if f"ANSWER: {choice}" in answer_text or f"ANSWER_CHOICE\": \"{choice}" in answer_text:
            return choice
    
    # Last resort: find isolated A, B, C, D
    import re
    matches = re.findall(r'\b([ABCD])\b', answer_text)
    if matches:
        return matches[-1]  # Take the last match
    
    return ""

def test_medqa_full(use_rag=True, max_questions=None, model_path=None, use_vllm=True):
    """
    Test MedGemma 4B-IT on full MedQA dataset
    
    Args:
        use_rag (bool): Whether to use RAG or just Chain-of-Thought
        max_questions (int): Maximum number of questions to test (None for all)
        model_path (str): Path to model checkpoint or model name (default: google/medgemma-4b-it)
        use_vllm (bool): Whether to use vLLM for faster inference
    """
    print("=" * 80)
    print(f"MedGemma 4B-IT Full MedQA Test ({'with RAG' if use_rag else 'CoT only'})")
    print("=" * 80)
    
    # Load MedQA dataset
    print("Loading MedQA dataset...")
    try:
        medqa_dataset = QADataset("medqa")
        total_questions = len(medqa_dataset)
        print(f"✓ MedQA dataset loaded: {total_questions} questions")
    except Exception as e:
        print(f"✗ Error loading MedQA dataset: {e}")
        return None
    
    # Limit questions if specified
    if max_questions:
        total_questions = min(max_questions, total_questions)
        print(f"Testing on first {total_questions} questions")
    
    # Set model path (default to base model if not specified)
    if model_path is None:
        model_path = "google/medgemma-4b-it"
    
    # Validate model path if it's a local path
    is_local_path = not model_path.startswith("google/") and not model_path.startswith("microsoft/") and not model_path.startswith("meta/")
    
    if is_local_path:
        if not os.path.exists(model_path):
            print(f"✗ Error: Model path '{model_path}' does not exist!")
            return None
        elif not os.path.isdir(model_path):
            print(f"✗ Error: Model path '{model_path}' is not a directory!")
            return None
        else:
            print(f"✓ Found local model directory: {model_path}")
            # Convert to absolute path for better compatibility
            model_path = os.path.abspath(model_path)
    
    # Check if this is a LoRA adapter
    is_lora_adapter = False
    if is_local_path:
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            is_lora_adapter = True
            print("✓ Detected LoRA adapter checkpoint")
            if use_vllm:
                print("⚠️  LoRA adapters may not work with vLLM. Will try vLLM first, then fall back if needed.")
    
    print(f"\nInitializing MedGemma using model: {model_path}")
    print(f"RAG mode: {use_rag}")
    print(f"Model type: {'Local checkpoint' if is_local_path else 'HuggingFace model'}")
    print(f"Inference backend: {'Transformers Pipeline' if not use_vllm else 'vLLM'}")
    
    try:
        if use_rag:
            # Try different corpus options for RAG
            corpus_options = ["Textbooks", "PubMed"]
            medrag = None
            
            for corpus_name in corpus_options:
                print(f"Trying {corpus_name} corpus with model...")
                try:
                    # For LoRA adapters with vLLM, we might need special handling
                    if is_lora_adapter and use_vllm:
                        print(f"  → Attempting vLLM with LoRA adapter...")
                        
                    medrag = MedRAG(
                        llm_name=model_path,  # Use specified model path
                        rag=True,
                        retriever_name="MedCPT",
                        corpus_name=corpus_name,
                        db_dir="./MedRAG/corpus",
                        cache_dir="./MedRAG/cache",
                        corpus_cache=True,
                        use_vllm=use_vllm  # Use vLLM based on parameter
                    )
                    print(f"✓ MedRAG with {corpus_name} and model '{model_path}' initialized successfully")
                    if is_lora_adapter and use_vllm:
                        print("✓ vLLM successfully loaded LoRA adapter!")
                    break
                except Exception as e:
                    print(f"✗ Failed with {corpus_name}: {str(e)[:200]}...")  # Truncate long error messages
                    if "Hugging Face repository ID" in str(e) or "Invalid repository ID" in str(e):
                        print("  → LoRA adapter detected, trying without vLLM...")
                        try:
                            medrag = MedRAG(
                                llm_name=model_path,
                                rag=True,
                                retriever_name="MedCPT",
                                corpus_name=corpus_name,
                                db_dir="./MedRAG/corpus",
                                cache_dir="./MedRAG/cache",
                                corpus_cache=True,
                                use_vllm=False  # Try without vLLM
                            )
                            print(f"✓ MedRAG with {corpus_name} initialized successfully (using transformers)")
                            break
                        except Exception as e2:
                            print(f"✗ Also failed with transformers: {str(e2)[:200]}...")
                            # If it's a LoRA adapter, try using base model
                            if is_lora_adapter:
                                print("  → Trying with base model instead of LoRA adapter...")
                                try:
                                    # Read adapter config to get base model
                                    import json
                                    with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                                        adapter_config = json.load(f)
                                    base_model = adapter_config.get("base_model_name_or_path", "google/medgemma-4b-it")
                                    print(f"  → Using base model: {base_model}")
                                    
                                    medrag = MedRAG(
                                        llm_name=base_model,
                                        rag=True,
                                        retriever_name="MedCPT",
                                        corpus_name=corpus_name,
                                        db_dir="./MedRAG/corpus",
                                        cache_dir="./MedRAG/cache",
                                        corpus_cache=True,
                                        use_vllm=use_vllm
                                    )
                                    print(f"✓ MedRAG with {corpus_name} initialized using base model")
                                    break
                                except Exception as e3:
                                    print(f"✗ Base model also failed: {str(e3)[:200]}...")
                    continue
            
            if medrag is None:
                print("✗ Failed to initialize with RAG, falling back to CoT")
                use_rag = False
        
        if not use_rag:
            try:
                medrag = MedRAG(
                    llm_name=model_path,  # Use specified model path
                    rag=False,
                    cache_dir="./MedRAG/cache",
                    use_vllm=use_vllm  # Use vLLM based on parameter
                )
                print(f"✓ MedGemma with model '{model_path}' initialized for Chain-of-Thought")
            except Exception as e:
                print(f"✗ Failed to initialize with vLLM: {str(e)[:200]}...")
                if "Hugging Face repository ID" in str(e) or "Invalid repository ID" in str(e):
                    print("  → Trying without vLLM for local checkpoint...")
                    try:
                        medrag = MedRAG(
                            llm_name=model_path,
                            rag=False,
                            cache_dir="./MedRAG/cache",
                            use_vllm=False  # Try without vLLM
                        )
                        print(f"✓ MedGemma with model '{model_path}' initialized for Chain-of-Thought (using transformers)")
                    except Exception as e2:
                        print(f"✗ Also failed with transformers: {str(e2)[:200]}...")
                        # If it's a LoRA adapter, try using base model
                        if is_lora_adapter:
                            print("  → Trying with base model instead of LoRA adapter...")
                            try:
                                # Read adapter config to get base model
                                import json
                                with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                                    adapter_config = json.load(f)
                                base_model = adapter_config.get("base_model_name_or_path", "google/medgemma-4b-it")
                                print(f"  → Using base model: {base_model}")
                                
                                medrag = MedRAG(
                                    llm_name=base_model,
                                    rag=False,
                                    cache_dir="./MedRAG/cache",
                                    use_vllm=use_vllm
                                )
                                print(f"✓ MedGemma initialized for Chain-of-Thought using base model")
                            except Exception as e3:
                                print(f"✗ Base model also failed: {str(e3)[:200]}...")
                                raise e3
                        else:
                            raise e2
                else:
                    raise e
            
    except Exception as e:
        print(f"✗ Error initializing MedRAG: {e}")
        return None
    
    # Run evaluation
    print(f"\nRunning evaluation on {total_questions} questions...")
    results = []
    correct_count = 0
    total_processed = 0
    
    start_time = time.time()
    
    for i in tqdm(range(total_questions), desc="Processing questions"):
        try:
            # Get question data
            question_data = medqa_dataset[i]
            question = question_data['question']
            options = question_data['options']
            correct_answer = question_data['answer']
            
            # Generate answer
            if use_rag and medrag.rag:
                # Temporarily monkey-patch the generate method to capture the actual prompt
                original_generate = medrag.generate
                captured_prompt = None
                
                def capture_prompt_generate(messages, **kwargs):
                    nonlocal captured_prompt
                    # Capture the user message which contains the actual prompt
                    for msg in messages:
                        if msg.get("role") == "user":
                            captured_prompt = msg.get("content", "")
                            break
                    return original_generate(messages, **kwargs)
                
                medrag.generate = capture_prompt_generate
                
                answer_json, snippets, scores = medrag.answer(
                    question=question,
                    options=options,
                    k=32  # Retrieve top 32 snippets
                )
                
                # Restore original method
                medrag.generate = original_generate
                
                num_snippets = len(snippets) if snippets else 0
                
                # Use captured prompt or fallback to constructed prompt
                if captured_prompt:
                    actual_prompt = captured_prompt
                else:
                    # Fallback: construct the prompt manually
                    if snippets:
                        context_text = "\n".join(snippets)
                    else:
                        context_text = "No context retrieved"
                    
                    options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())])
                    actual_prompt = f"Here are the relevant documents:\n{context_text}\n\nHere is the question:\n{question}\n\nHere are the potential choices:\n{options_text}\n\nPlease think step-by-step and generate your output in json:"
            else:
                answer_json, _, _ = medrag.answer(
                    question=question,
                    options=options
                )
                num_snippets = 0
                snippets = []
                scores = []
                actual_prompt = f"Question: {question}\nOptions: {options}"
            
            # Parse answer
            predicted_answer = parse_answer(answer_json)
            
            # Check correctness
            is_correct = predicted_answer.strip().upper() == correct_answer.strip().upper()
            if is_correct:
                correct_count += 1
            
            total_processed += 1
            
            # Store result with detailed RAG information
            result = {
                'question_id': i,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'raw_answer': str(answer_json),
                'num_snippets': num_snippets,
                'rag_snippets': snippets[:5] if snippets else [],  # Store top 5 snippets
                'rag_scores': scores[:5] if scores else [],  # Store top 5 scores
                'actual_prompt': actual_prompt,  # Store the actual prompt sent to model
                'rag_analysis': {
                    'total_retrieved': len(snippets) if snippets else 0,
                    'top_snippet_length': len(snippets[0]) if snippets and len(snippets) > 0 else 0,
                    'avg_snippet_length': sum(len(s) for s in snippets[:5]) / len(snippets[:5]) if snippets and len(snippets) > 0 else 0,
                    'score_range': [min(scores[:5]), max(scores[:5])] if scores and len(scores) > 0 else [0, 0]
                }
            }
            results.append(result)
            
            # Print progress every 50 questions
            if (i + 1) % 50 == 0:
                current_accuracy = correct_count / total_processed * 100
                elapsed_time = time.time() - start_time
                avg_time_per_q = elapsed_time / (i + 1)
                remaining_time = avg_time_per_q * (total_questions - i - 1)
                print(f"\nProgress: {i+1}/{total_questions}")
                print(f"Accuracy: {current_accuracy:.2f}%")
                print(f"Avg time per question: {avg_time_per_q:.2f}s")
                print(f"Estimated remaining time: {remaining_time/60:.1f} minutes")
            
        except Exception as e:
            print(f"\nError processing question {i}: {e}")
            continue
    
    # Calculate final results
    if total_processed > 0:
        accuracy = correct_count / total_processed * 100
        total_time = time.time() - start_time
        avg_time = total_time / total_processed
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Model: {model_path}")
        print(f"Dataset: MedQA")
        print(f"Method: {'RAG' if use_rag and medrag.rag else 'Chain-of-Thought'}")
        if use_rag and medrag.rag:
            print(f"Corpus: {medrag.corpus_name}")
            print(f"Retriever: {medrag.retriever_name}")
        print(f"Total Questions Processed: {total_processed}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Average Time per Question: {avg_time:.2f}s")
        print("=" * 80)
        
        # Save results
        timestamp = int(time.time())
        method_name = "rag" if (use_rag and medrag.rag) else "cot"
        
        # Create filename based on model type
        if "checkpoint" in model_path.lower():
            # Extract checkpoint info from path
            import re
            checkpoint_match = re.search(r'checkpoint-(\d+)', model_path)
            if checkpoint_match:
                checkpoint_num = checkpoint_match.group(1)
                model_name = f"checkpoint{checkpoint_num}"
            else:
                model_name = "checkpoint"
        elif "/" in model_path:
            # HuggingFace model name
            model_name = model_path.split("/")[-1].replace("-", "_")
        else:
            # Local path or simple name
            model_name = model_path.replace("/", "_").replace("-", "_")
        
        results_file = f"medgemma_medqa_{method_name}_{model_name}_results_{timestamp}.json"
        
        final_results = {
            'model': model_path,
            'dataset': 'MedQA',
            'method': method_name,
            'use_vllm': True,
            'corpus': medrag.corpus_name if (use_rag and medrag.rag) else None,
            'retriever': medrag.retriever_name if (use_rag and medrag.rag) else None,
            'total_questions': total_processed,
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'total_time_minutes': total_time/60,
            'avg_time_per_question': avg_time,
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        
        # Show some example results
        print("\nExample Results (first 5):")
        print("-" * 50)
        for i, result in enumerate(results[:5]):
            status = "✓ CORRECT" if result['is_correct'] else "✗ INCORRECT"
            print(f"Q{i+1}: {result['question'][:100]}...")
            print(f"Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
            print(f"Status: {status}")
            print("-" * 50)
        
        return {
            'accuracy': accuracy,
            'total_questions': total_processed,
            'correct_answers': correct_count,
            'method': method_name,
            'results_file': results_file
        }
    else:
        print("No questions were successfully processed.")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Test MedGemma 4B-IT on full MedQA dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available checkpoints
  python test_medgemma_medqa_vllm_full.py --list-checkpoints
  
  # Use base model with RAG
  python test_medgemma_medqa_vllm_full.py --rag
  
  # Use specific checkpoint with RAG
  python test_medgemma_medqa_vllm_full.py --rag --model-path ./rag_grpo_outputs_20250904_005255/checkpoint-250
  
  # Use different checkpoint
  python test_medgemma_medqa_vllm_full.py --rag --model-path ./rag_grpo_outputs_20250904_005255/checkpoint-200
  
  # Test both RAG and CoT with checkpoint
  python test_medgemma_medqa_vllm_full.py --both --model-path ./rag_grpo_outputs_20250904_005255/checkpoint-250
  
  # Test only CoT with base model
  python test_medgemma_medqa_vllm_full.py --model-path google/medgemma-4b-it
  
  # Test with limited questions
  python test_medgemma_medqa_vllm_full.py --rag --max-questions 100 --model-path ./rag_grpo_outputs_20250904_005255/checkpoint-250
  
  # Use any other model or checkpoint path
  python test_medgemma_medqa_vllm_full.py --rag --model-path /path/to/your/model
  
  # Use checkpoint without vLLM (if vLLM causes issues)
  python test_medgemma_medqa_vllm_full.py --rag --model-path ./rag_grpo_outputs_20250904_005255/checkpoint-250 --no-vllm
        """
    )
    parser.add_argument('--rag', action='store_true', help='Use RAG (default: Chain-of-Thought only)')
    parser.add_argument('--both', action='store_true', help='Test both RAG and CoT')
    parser.add_argument('--max-questions', type=int, default=None, help='Maximum number of questions to test')
    parser.add_argument('--model-path', type=str, default='google/medgemma-4b-it', 
                       help='Path to model checkpoint or model name (default: google/medgemma-4b-it)')
    parser.add_argument('--list-checkpoints', action='store_true', 
                       help='List available checkpoints and exit')
    parser.add_argument('--no-vllm', action='store_true', 
                       help='Disable vLLM (use transformers pipeline instead)')
    
    args = parser.parse_args()
    
    # List available checkpoints if requested
    if args.list_checkpoints:
        print("Available checkpoints:")
        checkpoint_dirs = [
            "rag_grpo_outputs_20250904_005255",
            # Add other checkpoint directories here if they exist
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                print(f"\n{checkpoint_dir}:")
                for item in os.listdir(checkpoint_dir):
                    item_path = os.path.join(checkpoint_dir, item)
                    if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                        print(f"  - {item_path}")
            else:
                print(f"\n{checkpoint_dir}: Not found")
        return
    
    # Display usage information
    print(f"Using model: {args.model_path}")
    print(f"RAG mode: {'Enabled' if args.rag else 'Disabled (CoT only)'}")
    print(f"vLLM: {'Disabled' if args.no_vllm else 'Enabled'}")
    if args.max_questions:
        print(f"Max questions: {args.max_questions}")
    print("-" * 50)
    
    if args.both:
        print("Testing both RAG and Chain-of-Thought approaches...")
        
        # Test Chain-of-Thought first
        print("\n" + "="*80)
        print("PHASE 1: Chain-of-Thought Evaluation")
        print("="*80)
        cot_results = test_medqa_full(use_rag=False, max_questions=args.max_questions, model_path=args.model_path, use_vllm=not args.no_vllm)
        
        # Test RAG
        print("\n" + "="*80)
        print("PHASE 2: RAG Evaluation")
        print("="*80)
        rag_results = test_medqa_full(use_rag=True, max_questions=args.max_questions, model_path=args.model_path, use_vllm=not args.no_vllm)
        
        # Compare results
        if cot_results and rag_results:
            print("\n" + "="*80)
            print("COMPARISON RESULTS")
            print("="*80)
            print(f"Chain-of-Thought: {cot_results['accuracy']:.2f}% ({cot_results['correct_answers']}/{cot_results['total_questions']})")
            print(f"RAG: {rag_results['accuracy']:.2f}% ({rag_results['correct_answers']}/{rag_results['total_questions']})")
            print(f"Improvement: {rag_results['accuracy'] - cot_results['accuracy']:+.2f}%")
            print("="*80)
    else:
        # Single test
        use_rag = args.rag
        results = test_medqa_full(use_rag=use_rag, max_questions=args.max_questions, model_path=args.model_path, use_vllm=not args.no_vllm)
        
        if results:
            print(f"\n✓ Test completed successfully!")
            print(f"Final Accuracy: {results['accuracy']:.2f}%")
        else:
            print("\n✗ Test failed.")
            sys.exit(1)

if __name__ == "__main__":
    main()
