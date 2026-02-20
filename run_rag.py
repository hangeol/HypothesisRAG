#!/usr/bin/env python3
"""
RAG Comparison Experiment Runner

CLI tool for running 3-way RAG comparison experiments:
- direct_rag: No query rewriting
- baseline_rewrite_rag: LLM query rewriting without planning
- planning_rewrite_rag: Planning-based query rewriting

Usage:
    python -m run_rag --mode direct --input "..."
    python -m run_rag --input_file data.jsonl --batch
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add current directory to path
# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from rag_core import create_rag_compare_graph, RAGCompareGraph


def run_single(
    graph: RAGCompareGraph,
    user_input: str,
    mode: str,
    top_k: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single RAG comparison experiment.
    
    Args:
        graph: RAGCompareGraph instance
        user_input: The medical question
        mode: One of "direct", "baseline", "planning"
        top_k: Number of documents to retrieve per query
        verbose: Print progress
        
    Returns:
        Result dictionary
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Mode: {mode.upper()}")
        print(f"{'='*70}")
        print(f"Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    
    result = graph.run(user_input, mode=mode, top_k=top_k)
    
    if verbose:
        print(f"\n📝 Final Queries ({len(result['final_queries'])}):")
        for i, q in enumerate(result['final_queries'], 1):
            print(f"   {i}. {q[:80]}{'...' if len(q) > 80 else ''}")
        
        print(f"\n📚 Retrieved Documents ({len(result['retrieved_docs'])}):")
        for i, doc in enumerate(result['retrieved_docs'][:5], 1):
            title = doc.get('title', 'N/A')[:50]
            score = doc.get('fused_score', 0)
            print(f"   {i}. [{score:.3f}] {title}")
        
        if result.get('plan'):
            print(f"\n🎯 Plan:")
            print(f"   Features: {result['plan'].get('observed_features', [])}")
            print(f"   Co-occurrence: {result['plan'].get('must_check_cooccurrence', [])}")
        
        print(f"\n📊 Metrics:")
        for k, v in result['metrics'].items():
            if isinstance(v, float):
                print(f"   {k}: {v:.3f}")
            else:
                print(f"   {k}: {v}")
    
    return result


def run_batch(
    graph: RAGCompareGraph,
    input_file: str,
    modes: List[str],
    top_k: int = 5,
    output_dir: str = ".",
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Run batch RAG comparison experiments.
    
    Args:
        graph: RAGCompareGraph instance
        input_file: Path to JSONL file with inputs
        modes: List of modes to run
        top_k: Number of documents to retrieve per query
        output_dir: Directory for output files
        verbose: Print progress
        
    Returns:
        Dict mapping mode to output file path
    """
    # Load input data
    inputs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    inputs.append(data)
                except json.JSONDecodeError:
                    # Treat as plain text
                    inputs.append({"id": len(inputs), "text": line})
    
    if verbose:
        print(f"Loaded {len(inputs)} inputs from {input_file}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = {}
    
    for mode in modes:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running mode: {mode.upper()}")
            print(f"{'='*70}")
        
        results = []
        for i, item in enumerate(inputs):
            input_id = item.get("id", i)
            text = item.get("text", item.get("question", str(item)))
            
            if verbose:
                print(f"[{i+1}/{len(inputs)}] Processing ID: {input_id}")
            
            try:
                result = graph.run(text, mode=mode, top_k=top_k)
                result["input_id"] = input_id
                results.append(result)
            except Exception as e:
                print(f"Error processing {input_id}: {e}")
                results.append({
                    "input_id": input_id,
                    "error": str(e),
                    "mode": mode,
                })
        
        # Save results
        output_file = os.path.join(output_dir, f"results_{mode}_{timestamp}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        output_files[mode] = output_file
        if verbose:
            print(f"✓ Saved {len(results)} results to {output_file}")
    
    return output_files


def run_comparison(
    graph: RAGCompareGraph,
    user_input: str,
    top_k: int = 5,
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all three modes and compare results.
    
    Args:
        graph: RAGCompareGraph instance
        user_input: The medical question
        top_k: Number of documents to retrieve per query
        output_file: Optional output file path
        verbose: Print progress
        
    Returns:
        Comparison results
    """
    modes = ["direct", "baseline", "planning"]
    comparison = {
        "user_input": user_input,
        "top_k": top_k,
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }
    
    for mode in modes:
        result = run_single(graph, user_input, mode, top_k, verbose)
        comparison["results"][mode] = result
    
    # Generate comparison summary
    if verbose:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        for mode in modes:
            r = comparison["results"][mode]
            print(f"\n[{mode.upper()}]")
            print(f"  Queries: {len(r['final_queries'])}")
            print(f"  Docs: {len(r['retrieved_docs'])}")
            
            # Show query examples
            if r['final_queries']:
                print(f"  First query: {r['final_queries'][0][:60]}...")
    
    # Save if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n✓ Results saved to {output_file}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="RAG Comparison Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query, direct mode
  python -m rag_compare_runner --mode direct --input "What are symptoms of diabetes?"

  # Single query, all modes comparison
  python -m rag_compare_runner --input "What are symptoms of diabetes?" --compare

  # Batch mode from file
  python -m rag_compare_runner --input_file data.jsonl --batch

  # Batch mode with specific modes
  python -m rag_compare_runner --input_file data.jsonl --batch --modes direct baseline

  # Save output to file
  python -m rag_compare_runner --input "..." --compare --out results.json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Single input query'
    )
    input_group.add_argument(
        '--input_file', '-f',
        type=str,
        help='Input file (JSONL format, each line: {"id": ..., "text": ...})'
    )
    
    # Mode options
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['direct', 'baseline', 'planning'],
        help='Mode for single query (ignored if --compare or --batch)'
    )
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Run all three modes and compare (single input only)'
    )
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Run batch mode (requires --input_file)'
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['direct', 'baseline', 'planning'],
        default=['direct', 'baseline', 'planning'],
        help='Modes to run in batch mode (default: all three)'
    )
    
    # Configuration
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=5,
        help='Number of documents to retrieve per query (default: 5)'
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
    
    # Output options
    parser.add_argument(
        '--out', '-o',
        type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for batch mode (default: current directory)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and args.batch:
        parser.error("--batch requires --input_file, not --input")
    
    if args.input_file and not args.batch:
        parser.error("--input_file requires --batch flag")
    
    if args.input and not args.compare and not args.mode:
        parser.error("Single input requires --mode or --compare")
    
    # Initialize graph
    verbose = not args.quiet
    if verbose:
        print("Initializing RAG Compare Graph...")
    
    graph = create_rag_compare_graph(
        model_name=args.model,
        retriever_name=args.retriever,
        corpus_name=args.corpus,
    )
    
    if verbose:
        print(f"✓ Graph initialized (model: {args.model}, retriever: {args.retriever})")
    
    # Run experiment
    if args.batch:
        # Batch mode
        output_files = run_batch(
            graph=graph,
            input_file=args.input_file,
            modes=args.modes,
            top_k=args.top_k,
            output_dir=args.output_dir,
            verbose=verbose,
        )
        if verbose:
            print(f"\n✓ Batch processing complete. Output files:")
            for mode, path in output_files.items():
                print(f"   {mode}: {path}")
    
    elif args.compare:
        # Comparison mode (all three modes)
        comparison = run_comparison(
            graph=graph,
            user_input=args.input,
            top_k=args.top_k,
            output_file=args.out,
            verbose=verbose,
        )
    
    else:
        # Single mode
        result = run_single(
            graph=graph,
            user_input=args.input,
            mode=args.mode,
            top_k=args.top_k,
            verbose=verbose,
        )
        
        # Save if output file specified
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\n✓ Result saved to {args.out}")


if __name__ == "__main__":
    main()
