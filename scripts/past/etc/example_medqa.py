#!/usr/bin/env python3
"""
Example: Multi-Subquery RAG for MedQA

This script demonstrates how to use the Multi-Subquery RAG system
on the MedQA dataset, similar to the MIRAGE test_medqa.py approach.
"""

import os
import sys
import json
import time
import argparse
from typing import Optional

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE', 'src'))

from multi_subquery_rag import MultiSubqueryRAG


def load_medqa_dataset(max_questions: Optional[int] = None):
    """Load MedQA dataset"""
    try:
        from src.utils import QADataset
        dataset = QADataset("medqa")
        print(f"✓ MedQA dataset loaded: {len(dataset)} questions")
        
        if max_questions:
            return dataset, min(max_questions, len(dataset))
        return dataset, len(dataset)
    except ImportError:
        print("Warning: Could not load MedQA dataset. Using example questions.")
        return None, 0


def parse_answer(answer_text: str) -> str:
    """Parse answer choice from model output"""
    import re
    
    # Try JSON parsing
    try:
        if '{' in answer_text and '}' in answer_text:
            start = answer_text.find('{')
            end = answer_text.rfind('}') + 1
            json_str = answer_text[start:end]
            answer_data = json.loads(json_str)
            return answer_data.get('answer_choice', answer_data.get('answer', ''))
    except:
        pass
    
    # Look for patterns
    answer_upper = answer_text.upper()
    for choice in ['A', 'B', 'C', 'D']:
        if f"ANSWER IS {choice}" in answer_upper or f"CHOICE {choice}" in answer_upper:
            return choice
        if f"ANSWER: {choice}" in answer_upper or f"ANSWER_CHOICE\": \"{choice}" in answer_upper:
            return choice
    
    # Find isolated letters
    matches = re.findall(r'\b([ABCD])\b', answer_upper)
    if matches:
        return matches[-1]
    
    return ""


def run_example():
    """Run example with sample medical questions"""
    
    print("\n" + "=" * 70)
    print("Multi-Subquery RAG - Example Medical Questions")
    print("=" * 70)
    
    # Sample questions
    questions = [
        {
            "question": "A 45-year-old man presents with fatigue, increased thirst, and frequent urination. "
                       "His fasting blood glucose is 142 mg/dL. What is the most likely diagnosis?",
            "options": {
                "A": "Type 1 diabetes mellitus",
                "B": "Type 2 diabetes mellitus",
                "C": "Diabetes insipidus",
                "D": "Prediabetes"
            },
            "answer": "B"
        },
        {
            "question": "Which of the following is a common side effect of ACE inhibitors?",
            "options": {
                "A": "Hypokalemia",
                "B": "Dry cough",
                "C": "Weight gain",
                "D": "Bradycardia"
            },
            "answer": "B"
        }
    ]
    
    # Initialize the RAG system
    print("\nInitializing Multi-Subquery RAG system...")
    
    rag = MultiSubqueryRAG(
        llm_provider="openai",
        model_name="gpt-4o-mini",
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        num_subqueries=3,
        k_per_subquery=5,
    )
    
    results = []
    correct = 0
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"Question {i}/{len(questions)}")
        print(f"{'=' * 70}")
        
        result = rag.answer(q["question"], q["options"])
        
        # Parse the answer
        predicted = parse_answer(result["final_answer"])
        is_correct = predicted == q["answer"]
        
        if is_correct:
            correct += 1
        
        print(f"\n✓ Correct Answer: {q['answer']}")
        print(f"✓ Predicted Answer: {predicted}")
        print(f"✓ Status: {'CORRECT ✅' if is_correct else 'INCORRECT ❌'}")
        
        results.append({
            "question": q["question"],
            "correct_answer": q["answer"],
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "subqueries": result["subqueries"],
            "final_answer": result["final_answer"],
        })
    
    # Summary
    accuracy = correct / len(questions) * 100 if questions else 0
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total Questions: {len(questions)}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    return results


def run_medqa_evaluation(max_questions: int = 10):
    """Run evaluation on MedQA dataset"""
    
    print("\n" + "=" * 70)
    print("Multi-Subquery RAG - MedQA Evaluation")
    print("=" * 70)
    
    # Load dataset
    dataset, total = load_medqa_dataset(max_questions)
    
    if dataset is None:
        print("Cannot run evaluation without MedQA dataset.")
        return run_example()
    
    # Initialize the RAG system
    print("\nInitializing Multi-Subquery RAG system...")
    
    rag = MultiSubqueryRAG(
        llm_provider="openai",
        model_name="gpt-4o-mini",
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        num_subqueries=3,
        k_per_subquery=5,
    )
    
    results = []
    correct = 0
    
    start_time = time.time()
    
    for i in range(total):
        q = dataset[i]
        question = q['question']
        options = q['options']
        correct_answer = q['answer']
        
        print(f"\n{'=' * 70}")
        print(f"Question {i + 1}/{total}")
        print(f"{'=' * 70}")
        print(f"Q: {question[:100]}...")
        
        try:
            result = rag.answer(question, options)
            
            # Parse the answer
            predicted = parse_answer(result["final_answer"])
            is_correct = predicted.upper() == correct_answer.upper()
            
            if is_correct:
                correct += 1
            
            print(f"\n✓ Correct: {correct_answer} | Predicted: {predicted} | {'✅' if is_correct else '❌'}")
            
            results.append({
                "question_id": i,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "subqueries": result["subqueries"],
                "final_answer": result["final_answer"],
            })
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue
        
        # Progress update
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            accuracy = correct / (i + 1) * 100
            print(f"\n📊 Progress: {i + 1}/{total} | Accuracy: {accuracy:.1f}% | Time: {elapsed:.1f}s")
    
    # Final summary
    total_time = time.time() - start_time
    accuracy = correct / len(results) * 100 if results else 0
    
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Questions: {len(results)}")
    print(f"Correct Answers: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Avg Time per Question: {total_time / len(results):.1f}s" if results else "N/A")
    
    # Save results
    timestamp = int(time.time())
    output_file = f"multi_subquery_rag_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "total_questions": len(results),
            "correct_answers": correct,
            "total_time": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Subquery RAG for Medical QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--mode', type=str, default='example',
        choices=['example', 'medqa'],
        help='Mode: "example" for sample questions, "medqa" for dataset evaluation'
    )
    parser.add_argument(
        '--max-questions', type=int, default=10,
        help='Maximum number of questions to evaluate (for medqa mode)'
    )
    parser.add_argument(
        '--num-subqueries', type=int, default=3,
        help='Number of subqueries to generate per question'
    )
    parser.add_argument(
        '--k-per-subquery', type=int, default=5,
        help='Number of documents to retrieve per subquery'
    )
    parser.add_argument(
        '--model', type=str, default='gpt-4o-mini',
        help='LLM model to use'
    )
    parser.add_argument(
        '--provider', type=str, default='openai',
        choices=['openai', 'anthropic'],
        help='LLM provider'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'example':
        run_example()
    else:
        run_medqa_evaluation(max_questions=args.max_questions)


if __name__ == "__main__":
    main()

