#!/usr/bin/env python3
"""
Test script for MedGemma 4B-IT with MedQA using PubMed RAG in MIRAGE environment
Tests MedGemma 4B-IT model on MedQA dataset with PubMed corpus retrieval
"""

import os
import sys
import json
import time
import argparse
import torch
from tqdm import tqdm
import numpy as np

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

def test_medgemma_medqa_pubmed():
    """
    Test MedGemma 4B-IT on MedQA with PubMed RAG
    """
    print("=" * 80)
    print("Testing MedGemma 4B-IT on MedQA with PubMed RAG")
    print("=" * 80)
    
    # Model configuration
    model_name = "google/medgemma-4b-it"
    
    # Initialize MedRAG with available corpus options
    print(f"Initializing MedRAG with model: {model_name}")
    print("Setting up RAG retrieval...")
    
    corpus_options = ["Textbooks", "PubMed"]  # Try available corpus options
    medrag = None
    
    for corpus_name in corpus_options:
        print(f"\nTrying to initialize MedRAG with {corpus_name} corpus...")
        try:
            medrag = MedRAG(
                llm_name=model_name,
                rag=True,
                retriever_name="MedCPT",  # Best performing retriever for biomedical domain
                corpus_name=corpus_name,
                db_dir="./MedRAG/corpus",
                cache_dir="./MedRAG/cache",
                corpus_cache=True,        # Cache corpus for faster retrieval
                use_vllm=False           # Use transformers for better compatibility
            )
            print(f"✓ MedRAG initialized successfully with {corpus_name}")
            break
        except Exception as e:
            print(f"✗ Failed with {corpus_name}: {e}")
            continue
    
    if medrag is None:
        print("✗ Failed to initialize MedRAG with any corpus")
        return False
    
    # Load MedQA dataset
    print("\nLoading MedQA dataset...")
    try:
        medqa_dataset = QADataset("medqa")
        print(f"✓ MedQA dataset loaded: {len(medqa_dataset)} questions")
    except Exception as e:
        print(f"✗ Error loading MedQA dataset: {e}")
        return False
    
    # Test on a subset first (for faster testing)
    test_size = min(50, len(medqa_dataset))  # Test on first 50 questions or all if less
    print(f"\nTesting on {test_size} questions from MedQA...")
    
    results = []
    correct_count = 0
    total_questions = 0
    
    # Test each question
    for i in tqdm(range(test_size), desc="Processing questions"):
        try:
            # Get question data
            question_data = medqa_dataset[i]
            question = question_data['question']
            options = question_data['options']
            correct_answer = question_data['answer']
            
            # Generate answer using MedRAG
            start_time = time.time()
            answer_json, snippets, scores = medrag.answer(
                question=question,
                options=options,
                k=32  # Retrieve top 32 snippets
            )
            inference_time = time.time() - start_time
            
            # Parse the answer
            try:
                if isinstance(answer_json, str):
                    answer_data = json.loads(answer_json)
                else:
                    answer_data = answer_json
                    
                predicted_answer = answer_data.get('answer_choice', answer_data.get('answer', ''))
                reasoning = answer_data.get('step_by_step_thinking', answer_data.get('reasoning', ''))
                
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing fails, try to extract answer directly
                predicted_answer = str(answer_json)
                reasoning = ""
            
            # Check if answer is correct
            is_correct = predicted_answer.strip().upper() == correct_answer.strip().upper()
            if is_correct:
                correct_count += 1
            
            total_questions += 1
            
            # Store results
            result = {
                'question_id': i,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'reasoning': reasoning,
                'inference_time': inference_time,
                'num_snippets': len(snippets) if snippets else 0,
                'snippets': snippets[:3] if snippets else []  # Store top 3 snippets
            }
            results.append(result)
            
            # Print progress every 10 questions
            if (i + 1) % 10 == 0:
                current_accuracy = correct_count / total_questions * 100
                print(f"Progress: {i+1}/{test_size}, Accuracy so far: {current_accuracy:.2f}%")
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue
    
    # Calculate final metrics
    if total_questions > 0:
        accuracy = correct_count / total_questions * 100
        avg_inference_time = np.mean([r['inference_time'] for r in results])
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Dataset: MedQA")
        print(f"Corpus: {medrag.corpus_name}")
        print(f"Retriever: {medrag.retriever_name}")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average Inference Time: {avg_inference_time:.2f}s")
        print("=" * 80)
        
        # Save detailed results
        results_file = f"results_medgemma_medqa_pubmed_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model': model_name,
                'dataset': 'MedQA',
                'corpus': medrag.corpus_name,
                'retriever': medrag.retriever_name,
                'total_questions': total_questions,
                'correct_answers': correct_count,
                'accuracy': accuracy,
                'avg_inference_time': avg_inference_time,
                'results': results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        
        # Show some example results
        print("\nExample Results:")
        print("-" * 50)
        for i, result in enumerate(results[:3]):
            print(f"Question {i+1}: {result['question'][:100]}...")
            print(f"Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
            print(f"Status: {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}")
            if result['reasoning']:
                print(f"Reasoning: {result['reasoning'][:200]}...")
            print("-" * 50)
        
        return True
    else:
        print("No questions were successfully processed.")
        return False

def test_simple_medqa_question():
    """
    Test a single MedQA question for debugging
    """
    print("=" * 80)
    print("Testing Single MedQA Question (Debug)")
    print("=" * 80)
    
    # Simple test question
    question = "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. What is the most likely diagnosis?"
    options = {
        "A": "Anterior myocardial infarction",
        "B": "Inferior myocardial infarction", 
        "C": "Lateral myocardial infarction",
        "D": "Posterior myocardial infarction"
    }
    correct_answer = "B"
    
    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Correct Answer: {correct_answer}")
    print()
    
    try:
        # Initialize MedRAG with available corpus
        corpus_options = ["Textbooks", "PubMed"]
        medrag = None
        
        for corpus_name in corpus_options:
            try:
                medrag = MedRAG(
                    llm_name="google/medgemma-4b-it",
                    rag=True,
                    retriever_name="MedCPT",
                    corpus_name=corpus_name,
                    db_dir="./MedRAG/corpus",
                    cache_dir="./MedRAG/cache",
                    corpus_cache=True,
                    use_vllm=False
                )
                print(f"✓ MedRAG initialized with {corpus_name}")
                break
            except Exception as e:
                print(f"✗ Failed with {corpus_name}: {e}")
                continue
        
        if medrag is None:
            print("✗ Failed to initialize MedRAG")
            return False
        
        # Generate answer
        print("Generating answer with RAG...")
        answer, snippets, scores = medrag.answer(question=question, options=options, k=16)
        
        print(f"Generated Answer: {answer}")
        print(f"Number of snippets retrieved: {len(snippets) if snippets else 0}")
        
        if snippets:
            print("\nTop 3 Retrieved Snippets:")
            for i, snippet in enumerate(snippets[:3]):
                # Handle different snippet formats
                if isinstance(snippet, dict):
                    content = snippet.get('content', str(snippet))
                else:
                    content = str(snippet)
                print(f"{i+1}. {content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error in simple test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test MedGemma 4B-IT on MedQA with PubMed RAG')
    parser.add_argument('--simple', action='store_true', help='Run simple single question test')
    parser.add_argument('--full', action='store_true', help='Run full MedQA test')
    
    args = parser.parse_args()
    
    if args.simple:
        success = test_simple_medqa_question()
    elif args.full:
        success = test_medgemma_medqa_pubmed()
    else:
        # Default: run both tests
        print("Running simple test first...")
        success1 = test_simple_medqa_question()
        print("\n" + "="*80 + "\n")
        print("Running full MedQA test...")
        success2 = test_medgemma_medqa_pubmed()
        success = success1 and success2
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
