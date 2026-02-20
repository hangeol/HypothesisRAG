#!/usr/bin/env python3
"""
Simplified test script for MedGemma 4B-IT with MedQA using PubMed RAG
"""

import os
import sys
import json
import time

# Add paths carefully to avoid conflicts
sys.path.insert(0, '.')
sys.path.insert(0, 'MedRAG')
sys.path.insert(0, 'MedRAG/src')

# Import MedRAG
try:
    from MedRAG.src.medrag import MedRAG
    print("✓ Successfully imported MedRAG from MedRAG.src")
except ImportError as e:
    print(f"✗ Failed to import MedRAG: {e}")
    sys.exit(1)

def test_medgemma_simple():
    """
    Simple test of MedGemma 4B-IT with a single medical question
    """
    print("=" * 80)
    print("Testing MedGemma 4B-IT with Medical Question")
    print("=" * 80)
    
    # Simple medical question
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
        # Initialize MedRAG with MedGemma
        print("Initializing MedRAG with MedGemma 4B-IT...")
        medrag = MedRAG(
            llm_name="google/medgemma-4b-it",
            rag=True,
            retriever_name="MedCPT",
            corpus_name="Textbooks",  # Use available corpus
            db_dir="./MedRAG/corpus",
            cache_dir="./MedRAG/cache",
            corpus_cache=True,
            use_vllm=False
        )
        print("✓ MedRAG initialized successfully")
        
        # Test answer generation
        print("\nGenerating answer with RAG...")
        start_time = time.time()
        answer, snippets, scores = medrag.answer(question=question, options=options, k=16)
        inference_time = time.time() - start_time
        
        print(f"✓ Answer generated in {inference_time:.2f}s")
        print(f"Answer: {answer}")
        print(f"Number of snippets retrieved: {len(snippets) if snippets else 0}")
        
        if snippets and len(snippets) > 0:
            print("\nTop 2 Retrieved Snippets:")
            for i, snippet in enumerate(snippets[:2]):
                # Handle different snippet formats
                if isinstance(snippet, dict):
                    content = snippet.get('content', str(snippet))
                else:
                    content = str(snippet)
                print(f"{i+1}. {content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_rag():
    """
    Test MedGemma without RAG (Chain-of-Thought only)
    """
    print("\n" + "=" * 80)
    print("Testing MedGemma 4B-IT without RAG (CoT only)")
    print("=" * 80)
    
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
        # Initialize MedRAG without RAG
        print("Initializing MedGemma for Chain-of-Thought...")
        medrag = MedRAG(
            llm_name="google/medgemma-4b-it",
            rag=False,  # No RAG, just CoT
            cache_dir="./MedRAG/cache",
            use_vllm=False
        )
        print("✓ MedGemma initialized successfully")
        
        # Generate answer
        print("\nGenerating answer with Chain-of-Thought...")
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

if __name__ == "__main__":
    print("Starting MedGemma 4B-IT tests...")
    print()
    
    # Test without RAG first (simpler)
    success1 = test_without_rag()
    
    print("\n" + "="*50 + "\n")
    
    # Test with RAG
    success2 = test_medgemma_simple()
    
    if success1 or success2:
        print("\n✓ At least one test completed successfully!")
    else:
        print("\n✗ All tests failed.")
        sys.exit(1)
