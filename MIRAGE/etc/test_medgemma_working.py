#!/usr/bin/env python3
"""
Working MedGemma 4B-IT test with correct imports
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
# We need to be careful about path order
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

def test_medgemma_simple():
    """
    Simple test with MedGemma 4B-IT
    """
    print("=" * 80)
    print("MedGemma 4B-IT Simple Test")
    print("=" * 80)
    
    # Simple medical question
    question = "What is the most common cause of community-acquired pneumonia?"
    options = {
        "A": "Streptococcus pneumoniae",
        "B": "Haemophilus influenzae", 
        "C": "Mycoplasma pneumoniae",
        "D": "Legionella pneumophila"
    }
    
    print(f"Question: {question}")
    print(f"Options: {options}")
    print()
    
    try:
        # Initialize MedRAG without RAG first (simpler)
        print("Initializing MedGemma without RAG...")
        medrag = MedRAG(
            llm_name="google/medgemma-4b-it",
            rag=False,
            cache_dir="./MedRAG/cache",
            use_vllm=False
        )
        print("✓ MedGemma initialized successfully")
        
        # Generate answer
        print("\nGenerating answer...")
        start_time = time.time()
        answer, _, _ = medrag.answer(question=question, options=options)
        inference_time = time.time() - start_time
        
        print(f"✓ Answer generated in {inference_time:.2f}s")
        print(f"Answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_medgemma_with_rag():
    """
    Test MedGemma with RAG
    """
    print("=" * 80)
    print("MedGemma 4B-IT with RAG Test")
    print("=" * 80)
    
    question = "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. What is the most likely diagnosis?"
    options = {
        "A": "Anterior myocardial infarction",
        "B": "Inferior myocardial infarction", 
        "C": "Lateral myocardial infarction",
        "D": "Posterior myocardial infarction"
    }
    
    print(f"Question: {question}")
    print(f"Options: {options}")
    print()
    
    try:
        # Try with available corpus
        corpus_options = ["Textbooks"]  # Start with most likely available
        
        for corpus_name in corpus_options:
            print(f"Trying to initialize MedRAG with {corpus_name}...")
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
                print(f"✓ MedRAG with {corpus_name} initialized successfully")
                break
            except Exception as e:
                print(f"✗ Failed with {corpus_name}: {e}")
                continue
        else:
            print("Could not initialize with RAG, falling back to CoT")
            return False
        
        # Generate answer with RAG
        print("\nGenerating answer with RAG...")
        start_time = time.time()
        answer, snippets, scores = medrag.answer(question=question, options=options, k=16)
        inference_time = time.time() - start_time
        
        print(f"✓ Answer generated in {inference_time:.2f}s")
        print(f"Answer: {answer}")
        print(f"Retrieved {len(snippets) if snippets else 0} snippets")
        
        if snippets and len(snippets) > 0:
            print("\nTop 2 Retrieved Snippets:")
            for i, snippet in enumerate(snippets[:2]):
                # Handle different snippet formats
                if isinstance(snippet, dict):
                    content = snippet.get('content', str(snippet))
                else:
                    content = str(snippet)
                print(f"{i+1}. {content[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_medqa_subset():
    """
    Test on a small subset of MedQA
    """
    print("=" * 80)
    print("MedQA Subset Test")
    print("=" * 80)
    
    try:
        # Load MedQA dataset
        print("Loading MedQA dataset...")
        medqa_dataset = QADataset("medqa")
        print(f"✓ MedQA dataset loaded: {len(medqa_dataset)} questions")
        
        # Initialize MedRAG without RAG for speed
        print("Initializing MedGemma...")
        medrag = MedRAG(
            llm_name="google/medgemma-4b-it",
            rag=False,
            cache_dir="./MedRAG/cache",
            use_vllm=False
        )
        print("✓ MedGemma initialized")
        
        # Test on first 10 questions
        test_size = min(10, len(medqa_dataset))
        print(f"\nTesting on {test_size} questions...")
        
        correct_count = 0
        results = []
        
        for i in tqdm(range(test_size), desc="Processing"):
            try:
                question_data = medqa_dataset[i]
                question = question_data['question']
                options = question_data['options']
                correct_answer = question_data['answer']
                
                # Generate answer
                answer_json, _, _ = medrag.answer(question=question, options=options)
                
                # Parse answer
                try:
                    if isinstance(answer_json, str):
                        answer_data = json.loads(answer_json)
                    else:
                        answer_data = answer_json
                    predicted_answer = answer_data.get('answer_choice', answer_data.get('answer', ''))
                except:
                    predicted_answer = str(answer_json)
                
                is_correct = predicted_answer.strip().upper() == correct_answer.strip().upper()
                if is_correct:
                    correct_count += 1
                
                results.append({
                    'question': question[:100] + "...",
                    'correct': correct_answer,
                    'predicted': predicted_answer,
                    'is_correct': is_correct
                })
                
            except Exception as e:
                print(f"Error on question {i}: {e}")
                continue
        
        accuracy = correct_count / test_size * 100 if test_size > 0 else 0
        
        print(f"\nResults: {correct_count}/{test_size} correct ({accuracy:.1f}%)")
        
        # Show first few results
        print("\nFirst 5 results:")
        for i, result in enumerate(results[:5]):
            status = "✓" if result['is_correct'] else "✗"
            print(f"{i+1}. {status} {result['question']}")
            print(f"   Correct: {result['correct']}, Predicted: {result['predicted']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in MedQA test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting MedGemma 4B-IT tests...")
    print()
    
    # Test 1: Simple question without RAG
    print("Test 1: Simple question without RAG")
    success1 = test_medgemma_simple()
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Question with RAG
    print("Test 2: Question with RAG")
    success2 = test_medgemma_with_rag()
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: MedQA subset
    print("Test 3: MedQA subset")
    success3 = test_medqa_subset()
    
    if success1 or success2 or success3:
        print("\n✓ At least one test completed successfully!")
    else:
        print("\n✗ All tests failed.")
        sys.exit(1)

