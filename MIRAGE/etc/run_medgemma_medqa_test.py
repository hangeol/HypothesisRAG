#!/usr/bin/env python3
"""
Complete MedGemma 4B-IT test with MedQA using PubMed RAG
This script runs in the MedRAG directory with proper imports
"""

import os
import sys
import json
import time
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

def test_medgemma_on_medqa():
    """
    Test MedGemma 4B-IT on MedQA dataset with PubMed RAG
    """
    print("=" * 80)
    print("MedGemma 4B-IT + MedQA + PubMed RAG Test")
    print("=" * 80)
    
    # Load MedQA dataset
    print("Loading MedQA dataset...")
    try:
        medqa_dataset = QADataset("medqa")
        print(f"✓ MedQA dataset loaded: {len(medqa_dataset)} questions")
    except Exception as e:
        print(f"✗ Error loading MedQA dataset: {e}")
        return False
    
    # Initialize MedRAG with different corpus options
    corpus_options = ["Textbooks", "PubMed"]  # Only use available corpus
    medrag = None
    
    for corpus_name in corpus_options:
        print(f"\nTrying to initialize MedRAG with {corpus_name} corpus...")
        try:
            medrag = MedRAG(
                llm_name="google/medgemma-4b-it",
                rag=True,
                retriever_name="MedCPT",  # Best for biomedical domain
                corpus_name=corpus_name,
                db_dir="./MedRAG/corpus",
                cache_dir="./MedRAG/cache",
                corpus_cache=True,
                use_vllm=False
            )
            print(f"✓ MedRAG initialized successfully with {corpus_name}")
            break
        except Exception as e:
            print(f"✗ Failed with {corpus_name}: {e}")
            continue
    
    if medrag is None:
        print("✗ Failed to initialize MedRAG with any corpus")
        return False
    
    # Test on subset of questions
    test_size = min(100, len(medqa_dataset))  # Test on first 100 questions
    print(f"\nTesting on {test_size} questions from MedQA...")
    
    results = []
    correct_count = 0
    total_questions = 0
    
    start_time = time.time()
    
    for i in tqdm(range(test_size), desc="Processing questions"):
        try:
            # Get question data
            question_data = medqa_dataset[i]
            question = question_data['question']
            options = question_data['options']
            correct_answer = question_data['answer']
            
            # Generate answer with RAG
            answer_json, snippets, scores = medrag.answer(
                question=question,
                options=options,
                k=32  # Retrieve top 32 snippets
            )
            
            # Parse answer
            try:
                if isinstance(answer_json, str):
                    answer_data = json.loads(answer_json)
                else:
                    answer_data = answer_json
                    
                predicted_answer = answer_data.get('answer_choice', answer_data.get('answer', ''))
                reasoning = answer_data.get('step_by_step_thinking', answer_data.get('reasoning', ''))
                
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, extract answer directly
                predicted_answer = str(answer_json)
                reasoning = ""
            
            # Check correctness
            is_correct = predicted_answer.strip().upper() == correct_answer.strip().upper()
            if is_correct:
                correct_count += 1
            
            total_questions += 1
            
            # Store result
            result = {
                'question_id': i,
                'question': question[:200] + "..." if len(question) > 200 else question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'reasoning': reasoning[:300] + "..." if len(reasoning) > 300 else reasoning,
                'num_snippets': len(snippets) if snippets else 0
            }
            results.append(result)
            
            # Print progress every 20 questions
            if (i + 1) % 20 == 0:
                current_accuracy = correct_count / total_questions * 100
                elapsed_time = time.time() - start_time
                avg_time_per_q = elapsed_time / (i + 1)
                print(f"\nProgress: {i+1}/{test_size}")
                print(f"Accuracy: {current_accuracy:.2f}%")
                print(f"Avg time per question: {avg_time_per_q:.2f}s")
            
        except Exception as e:
            print(f"\nError processing question {i}: {e}")
            continue
    
    # Calculate final results
    if total_questions > 0:
        accuracy = correct_count / total_questions * 100
        total_time = time.time() - start_time
        avg_time = total_time / total_questions
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Model: google/medgemma-4b-it")
        print(f"Dataset: MedQA")
        print(f"Corpus: {medrag.corpus_name}")
        print(f"Retriever: {medrag.retriever_name}")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time per Question: {avg_time:.2f}s")
        print("=" * 80)
        
        # Save results
        results_file = f"medgemma_medqa_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model': 'google/medgemma-4b-it',
                'dataset': 'MedQA',
                'corpus': medrag.corpus_name,
                'retriever': medrag.retriever_name,
                'total_questions': total_questions,
                'correct_answers': correct_count,
                'accuracy': accuracy,
                'total_time': total_time,
                'avg_time_per_question': avg_time,
                'results': results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        
        # Show some examples
        print("\nExample Results:")
        print("-" * 50)
        for i, result in enumerate(results[:5]):
            status = "✓ CORRECT" if result['is_correct'] else "✗ INCORRECT"
            print(f"Q{i+1}: {result['question']}")
            print(f"Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
            print(f"Status: {status}")
            if result['reasoning']:
                print(f"Reasoning: {result['reasoning']}")
            print("-" * 50)
        
        return True
    else:
        print("No questions were successfully processed.")
        return False

if __name__ == "__main__":
    success = test_medgemma_on_medqa()
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed.")
        sys.exit(1)
