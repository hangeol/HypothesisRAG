#!/usr/bin/env python3
"""
MedQA Evaluation for RAG Comparison Experiment

This script evaluates the 3-way RAG comparison pipeline on MedQA testset:
- direct_rag: No query rewriting
- baseline_rewrite_rag: LLM query rewriting without planning
- planning_rewrite_rag: Planning-based query rewriting

Purpose: Compare retrieval quality and answer accuracy across different strategies.
"""

import os
import sys
import json
import time
import re
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path to find rag_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_core import create_rag_compare_graph, RAGCompareGraph, MIRAGE_SYSTEM_PROMPT


# ============================================================================
# MedQA Dataset Loader
# ============================================================================
class MedQADataset:
    """
    MedQA Dataset loader compatible with MIRAGE benchmark.json format
    """
    
    def __init__(self, benchmark_path: Optional[str] = None):
        """
        Initialize MedQA dataset.
        
        Args:
            benchmark_path: Path to benchmark.json file
        """
        if benchmark_path is None:
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "MIRAGE", "benchmark.json"),
                os.path.join(os.path.dirname(__file__), "..", "MIRAGE", "benchmark.json"),
                "/mnt/data1/home/hangeol/project/MIRAGE/benchmark.json",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    benchmark_path = path
                    break
            
            if benchmark_path is None:
                raise FileNotFoundError(
                    f"benchmark.json not found. Tried: {possible_paths}"
                )
        
        print(f"Loading MedQA from: {benchmark_path}")
        
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
        
        if "medqa" not in benchmark:
            raise KeyError("'medqa' not found in benchmark.json")
        
        self.dataset = benchmark["medqa"]
        self.index = sorted(self.dataset.keys())
        
        print(f"✓ Loaded {len(self)} MedQA questions")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, key) -> Dict[str, Any]:
        if isinstance(key, int):
            return self.dataset[self.index[key]]
        elif isinstance(key, slice):
            return [self.__getitem__(i) for i in range(len(self))[key]]
        else:
            raise KeyError(f"Key type {type(key)} not supported")


# ============================================================================
# Answer Parsing
# ============================================================================
def parse_answer(answer_text: str) -> str:
    """
    Parse answer choice (A, B, C, D) from model output.
    
    Args:
        answer_text: Raw answer text from LLM
        
    Returns:
        Parsed answer choice (A, B, C, D) or empty string
    """
    if not answer_text:
        return ""
    
    answer_text = str(answer_text)
    
    # Try JSON parsing first
    try:
        if '{' in answer_text and '}' in answer_text:
            start = answer_text.find('{')
            end = answer_text.rfind('}') + 1
            json_str = answer_text[start:end]
            answer_data = json.loads(json_str)
            choice = answer_data.get('answer_choice', answer_data.get('answer', ''))
            if choice and choice.upper() in ['A', 'B', 'C', 'D']:
                return choice.upper()
    except (json.JSONDecodeError, AttributeError):
        pass
    
    answer_upper = answer_text.upper()
    
    # Look for common patterns
    patterns = [
        r"ANSWER IS[:\s]*([ABCD])",
        r"ANSWER[:\s]*([ABCD])",
        r"CHOICE[:\s]*([ABCD])",
        r"OPTION[:\s]*([ABCD])",
        r"\"ANSWER_CHOICE\"[:\s]*\"([ABCD])\"",
        r"CORRECT ANSWER[:\s]*([ABCD])",
        r"THE ANSWER[:\s]*([ABCD])",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer_upper)
        if match:
            return match.group(1)
    
    # Last resort: find isolated A, B, C, D (take the last one)
    matches = re.findall(r'\b([ABCD])\b', answer_upper)
    if matches:
        return matches[-1]
    
    return ""


# ============================================================================
# Answer Generation using RAG Results
# ============================================================================
def generate_answer_from_evidence(
    llm,
    question: str,
    options: Dict[str, str],
    retrieved_docs: List[Dict[str, Any]],
    system_prompt: str = MIRAGE_SYSTEM_PROMPT,
) -> Tuple[str, str]:
    """
    Generate answer from retrieved evidence.
    
    Args:
        llm: LangChain LLM instance
        question: The question text
        options: Answer options dict
        retrieved_docs: List of retrieved documents
        system_prompt: System prompt to use
        
    Returns:
        Tuple of (raw_response, parsed_answer)
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Format context from retrieved docs
    context_parts = []
    for idx, doc in enumerate(retrieved_docs[:10]):  # Use top 10
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        context_parts.append(f"Document [{idx+1}] (Title: {title})\n{content}")
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
    
    # Format options
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    # Build prompt (MIRAGE format)
    user_prompt = f"""Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options_text}

Please think step-by-step and generate your output in json:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        raw_response = response.content
        parsed_answer = parse_answer(raw_response)
        return raw_response, parsed_answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return str(e), ""


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_single_question(
    graph: RAGCompareGraph,
    question_data: Dict[str, Any],
    modes: List[str],
    top_k: int = 5,
    generate_answers: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a single question across multiple modes.
    
    Args:
        graph: RAGCompareGraph instance
        question_data: Question dict with 'question', 'options', 'answer'
        modes: List of modes to evaluate
        top_k: Number of documents to retrieve per query
        generate_answers: Whether to generate answers from evidence
        
    Returns:
        Evaluation result dict
    """
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data.get('answer_idx', question_data.get('answer', ''))
    
    result = {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "modes": {},
    }
    
    for mode in modes:
        try:
            # Run RAG pipeline
            rag_result = graph.run(question, mode=mode, top_k=top_k)
            
            mode_result = {
                "final_queries": rag_result["final_queries"],
                "num_queries": len(rag_result["final_queries"]),
                "num_docs": len(rag_result["retrieved_docs"]),
                "retrieved_docs": [
                    {
                        "id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "content": doc.get("content", "")[:500],  # Truncate for storage
                        "fused_score": doc.get("fused_score", 0),
                    }
                    for doc in rag_result["retrieved_docs"][:5]  # Top 5 only
                ],
                "metrics": rag_result["metrics"],
            }
            
            # Add plan for planning mode
            if mode == "planning" and rag_result.get("plan"):
                mode_result["plan"] = rag_result["plan"]
            
            # Generate answer if requested
            if generate_answers:
                raw_response, predicted_answer = generate_answer_from_evidence(
                    graph.llm,
                    question,
                    options,
                    rag_result["retrieved_docs"],
                )
                mode_result["raw_response"] = raw_response
                mode_result["predicted_answer"] = predicted_answer
                mode_result["is_correct"] = predicted_answer.upper() == correct_answer.upper()
            
            result["modes"][mode] = mode_result
            
        except Exception as e:
            result["modes"][mode] = {
                "error": str(e),
                "is_correct": False,
            }
    
    return result


def run_medqa_evaluation(
    max_questions: int = 100,
    modes: List[str] = ["direct", "baseline", "planning"],
    top_k: int = 5,
    model_name: str = "gpt-4o-mini",
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    generate_answers: bool = True,
    output_dir: str = ".",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run MedQA evaluation across different RAG modes.
    
    Args:
        max_questions: Maximum number of questions to evaluate
        modes: List of modes to evaluate
        top_k: Documents to retrieve per query
        model_name: OpenAI model name
        retriever_name: Retriever name
        corpus_name: Corpus name
        generate_answers: Whether to generate final answers
        output_dir: Output directory for results
        verbose: Print progress
        
    Returns:
        Evaluation summary dict
    """
    print("=" * 80)
    print("MedQA RAG Comparison Evaluation")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Modes: {modes}")
    print(f"Max questions: {max_questions}")
    print(f"Top-K: {top_k}")
    print(f"Generate answers: {generate_answers}")
    print("=" * 80)
    
    # Load dataset
    dataset = MedQADataset()
    total_questions = min(max_questions, len(dataset))
    
    # Initialize graph
    print("\nInitializing RAG Compare Graph...")
    graph = create_rag_compare_graph(
        model_name=model_name,
        retriever_name=retriever_name,
        corpus_name=corpus_name,
    )
    print(f"✓ Graph initialized")
    
    # Run evaluation
    results = []
    mode_stats = {mode: {"correct": 0, "total": 0, "errors": 0} for mode in modes}
    
    start_time = time.time()
    
    for i in tqdm(range(total_questions), desc="Evaluating", disable=not verbose):
        try:
            question_data = dataset[i]
            
            result = evaluate_single_question(
                graph=graph,
                question_data=question_data,
                modes=modes,
                top_k=top_k,
                generate_answers=generate_answers,
            )
            
            result["question_id"] = i
            results.append(result)
            
            # Update stats
            for mode in modes:
                mode_result = result["modes"].get(mode, {})
                if "error" not in mode_result:
                    mode_stats[mode]["total"] += 1
                    if mode_result.get("is_correct", False):
                        mode_stats[mode]["correct"] += 1
                else:
                    mode_stats[mode]["errors"] += 1
            
            # Progress update
            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total_questions - i - 1) / rate if rate > 0 else 0
                
                print(f"\n[{i+1}/{total_questions}] Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s")
                for mode in modes:
                    stats = mode_stats[mode]
                    if stats["total"] > 0:
                        acc = stats["correct"] / stats["total"] * 100
                        print(f"  {mode}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
                    
        except Exception as e:
            print(f"\nError processing question {i}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Compute final summary
    summary = {
        "config": {
            "model_name": model_name,
            "retriever_name": retriever_name,
            "corpus_name": corpus_name,
            "top_k": top_k,
            "generate_answers": generate_answers,
            "max_questions": max_questions,
            "total_evaluated": len(results),
        },
        "timing": {
            "total_seconds": total_time,
            "avg_per_question": total_time / len(results) if results else 0,
        },
        "mode_results": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Compute per-mode metrics
    for mode in modes:
        stats = mode_stats[mode]
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
        else:
            accuracy = 0.0
        
        summary["mode_results"][mode] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "errors": stats["errors"],
            "accuracy": accuracy,
        }
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total questions evaluated: {len(results)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per question: {total_time/len(results):.2f}s" if results else "N/A")
    print()
    
    print("Accuracy by Mode:")
    print("-" * 40)
    for mode in modes:
        stats = summary["mode_results"][mode]
        print(f"  {mode:12s}: {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']:.1f}%)")
    
    # Compare modes
    if len(modes) > 1 and "direct" in modes:
        print()
        print("Improvement over Direct:")
        print("-" * 40)
        direct_acc = summary["mode_results"]["direct"]["accuracy"]
        for mode in modes:
            if mode != "direct":
                mode_acc = summary["mode_results"][mode]["accuracy"]
                diff = mode_acc - direct_acc
                print(f"  {mode:12s}: {diff:+.1f}%")
    
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"medqa_rag_compare_{timestamp}.json")
    
    full_results = {
        "summary": summary,
        "results": results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return summary


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MedQA Evaluation for RAG Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 questions)
  python rag_compare_medqa_eval.py --max-questions 10

  # Full evaluation (all questions)
  python rag_compare_medqa_eval.py --max-questions 1000

  # Specific modes only
  python rag_compare_medqa_eval.py --modes direct planning --max-questions 50

  # Evidence-only (no answer generation)
  python rag_compare_medqa_eval.py --no-answers --max-questions 100
        """
    )
    
    parser.add_argument(
        '--max-questions', '-n',
        type=int,
        default=100,
        help='Maximum number of questions to evaluate (default: 100)'
    )
    parser.add_argument(
        '--modes', '-m',
        nargs='+',
        choices=['direct', 'baseline', 'planning'],
        default=['direct', 'baseline', 'planning'],
        help='Modes to evaluate (default: all three)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Documents to retrieve per query (default: 5)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        default='MedCPT',
        help='Retriever name (default: MedCPT)'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='Textbooks',
        help='Corpus name (default: Textbooks)'
    )
    parser.add_argument(
        '--no-answers',
        action='store_true',
        help='Skip answer generation (evaluate retrieval only)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for results (default: current directory)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    summary = run_medqa_evaluation(
        max_questions=args.max_questions,
        modes=args.modes,
        top_k=args.top_k,
        model_name=args.model,
        retriever_name=args.retriever,
        corpus_name=args.corpus,
        generate_answers=not args.no_answers,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    return summary


if __name__ == "__main__":
    main()
