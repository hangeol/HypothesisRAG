#!/usr/bin/env python3
"""
Graph Visualization for Multi-Subquery RAG System

This module provides tools to visualize the LangGraph workflow.
"""

import os
from typing import Optional


def visualize_graph(save_path: Optional[str] = None):
    """
    Create a visualization of the Multi-Subquery RAG graph.
    
    Args:
        save_path: Path to save the visualization image.
                  If None, displays in notebook or saves to default path.
    """
    from multi_subquery_rag import MultiSubqueryRAG
    
    # Create a minimal instance
    rag = MultiSubqueryRAG.__new__(MultiSubqueryRAG)
    rag.num_subqueries = 3
    rag.k_per_subquery = 5
    rag.llm = None
    rag.retrieval_system = None
    
    # Build graph
    from langgraph.graph import StateGraph, START, END
    from multi_subquery_rag import GraphState
    
    workflow = StateGraph(GraphState)
    
    # Add nodes (with dummy functions for visualization)
    workflow.add_node("generate_subqueries", lambda x: x)
    workflow.add_node("process_subqueries", lambda x: x)
    workflow.add_node("synthesize_answer", lambda x: x)
    
    # Define edges
    workflow.add_edge(START, "generate_subqueries")
    workflow.add_edge("generate_subqueries", "process_subqueries")
    workflow.add_edge("process_subqueries", "synthesize_answer")
    workflow.add_edge(END, "synthesize_answer")
    
    graph = workflow.compile()
    
    # Try to generate visualization
    try:
        # Get the graph image
        from IPython.display import Image, display
        
        png_data = graph.get_graph().draw_mermaid_png()
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(png_data)
            print(f"✓ Graph saved to: {save_path}")
        else:
            # Try to display in notebook
            try:
                display(Image(png_data))
            except:
                # Save to default path
                default_path = "multi_subquery_rag_graph.png"
                with open(default_path, 'wb') as f:
                    f.write(png_data)
                print(f"✓ Graph saved to: {default_path}")
                
    except ImportError:
        print("Note: Install graphviz and pygraphviz for PNG export.")
        print("\nMermaid diagram:")
        print(graph.get_graph().draw_mermaid())


def print_graph_structure():
    """Print a text representation of the graph structure"""
    
    structure = """
    Multi-Subquery RAG Graph Structure
    ===================================
    
    ┌─────────────┐
    │    START    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │     generate_subqueries         │
    │  ───────────────────────────    │
    │  Input: original_question       │
    │  Output: List[subqueries]       │
    │                                 │
    │  - Analyzes the question        │
    │  - Generates N focused queries  │
    │  - Each subquery targets        │
    │    a specific information need  │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │      process_subqueries         │
    │  ───────────────────────────    │
    │  For each subquery:             │
    │    1. Retrieve K documents      │
    │    2. Generate focused response │
    │                                 │
    │  Output: List[SubqueryResult]   │
    │    - subquery                   │
    │    - documents                  │
    │    - response                   │
    │    - scores                     │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────┐
    │      synthesize_answer          │
    │  ───────────────────────────    │
    │  Input: All subquery responses  │
    │                                 │
    │  - Combines all information     │
    │  - Resolves conflicts           │
    │  - Generates final answer       │
    │                                 │
    │  Output: final_answer           │
    └──────────────┬──────────────────┘
                   │
                   ▼
    ┌─────────────┐
    │     END     │
    └─────────────┘
    
    
    State Schema
    ============
    
    GraphState:
        original_question: str        # Input question
        options: Dict[str, str]       # Optional answer choices
        subqueries: List[str]         # Generated subqueries
        subquery_results: List[       # Results per subquery
            SubqueryResult:
                subquery: str
                documents: List[Dict]
                response: str
                scores: List[float]
        ]
        final_answer: str             # Synthesized answer
        num_subqueries: int           # Config: number of subqueries
        k_per_subquery: int           # Config: docs per subquery
    """
    
    print(structure)


if __name__ == "__main__":
    print_graph_structure()
    print("\n")
    
    try:
        visualize_graph("multi_subquery_rag_graph.png")
    except Exception as e:
        print(f"Could not generate image: {e}")

