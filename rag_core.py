#!/usr/bin/env python3
"""
RAG Comparison Experiment Graph using LangGraph

This module implements a 3-way RAG comparison:
(A) direct_rag: No query rewriting, direct retrieval
(B) baseline_rewrite_rag: LLM query rewriting without planning
(C) planning_rewrite_rag: Planning-based query rewriting

Purpose: Compare query rewriting strategies and retrieval evidence quality
"""

import os
import sys
import json
import re
from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import existing retriever
sys.path.insert(0, os.path.dirname(__file__))
from retriever import MIRAGERetriever, create_retriever

# Import MIRAGE prompt template (try multiple paths)
MIRAGE_SYSTEM_PROMPT = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE', 'MedRAG', 'src'))
    from template import general_medrag_system
    MIRAGE_SYSTEM_PROMPT = general_medrag_system
    print("✓ Loaded MIRAGE system prompt from template.py")
except ImportError:
    print("⚠ Using built-in MIRAGE system prompt (template.py not found)")


# ============================================================================
# Multi-Query Rewriting Prompt (Required)
# ============================================================================
MULTI_QUERY_PROMPT = """You are an AI language model assistant. Your task
is to generate exactly three different versions of the
given user question to retrieve relevant documents
from a vector database. By generating multiple
perspectives on the user question, your goal is to
help the user overcome some of the limitations of
the distance-based similarity search.
Original question: {query}
Format your response in plain text as:
Sub-query 1:
Sub-query 2:
Sub-query 3:"""


# ============================================================================
# Planning Schema (Pydantic for structured output)
# ============================================================================
class PlanningOutput(BaseModel):
    """Schema for planning step output"""
    observed_features: List[str] = Field(
        description="Key symptoms/findings/conditions from the input (3-7 items)",
        min_length=1,
        max_length=7
    )
    must_check_cooccurrence: List[List[str]] = Field(
        default=[],
        description="Feature pairs whose co-occurrence is diagnostically important"
    )
    need_disambiguation: List[str] = Field(
        default=[],
        description="Confusing concept pairs or disease groups to distinguish"
    )


# ============================================================================
# Graph State Definition
# ============================================================================
class RAGCompareState(TypedDict):
    """State for the RAG comparison graph"""
    # Input
    user_input: str
    mode: Literal["direct", "baseline", "planning"]
    
    # Planning (only for planning mode)
    plan: Optional[Dict[str, Any]]
    
    # Query rewriting
    rewritten_queries: Optional[List[str]]  # baseline/planning only
    final_queries: List[str]                # Queries actually used for retrieval
    
    # Retrieval results
    retrieved_docs: List[Dict[str, Any]]    # id/score/text/metadata/query_trace
    
    # Configuration
    top_k: int
    
    # Logs and debugging
    logs: List[str]


# ============================================================================
# RAG Comparison Graph
# ============================================================================
class RAGCompareGraph:
    """
    LangGraph-based RAG Comparison System
    
    Implements 3-way comparison:
    - direct: No rewriting
    - baseline: LLM rewriting without planning
    - planning: Planning-based rewriting
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the RAG comparison system.
        
        Args:
            model_name: OpenAI model name
            temperature: LLM temperature
            retriever_name: Retriever name (MedCPT, BM25, etc.)
            corpus_name: Corpus name (Textbooks, PubMed, etc.)
            api_key: OpenAI API key (falls back to env var)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        
        # Initialize LLM with structured output for planning
        self.llm_structured = self.llm.with_structured_output(PlanningOutput)
        
        # Initialize retriever
        self.retriever = create_retriever(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(RAGCompareState)
        
        # Add nodes
        workflow.add_node("normalize_input", self._normalize_input)
        workflow.add_node("route_mode", self._route_mode)
        workflow.add_node("make_plan", self._make_plan)
        workflow.add_node("make_queries_direct", self._make_queries_direct)
        workflow.add_node("make_queries_rewrite_baseline", self._make_queries_rewrite_baseline)
        workflow.add_node("make_queries_rewrite_planning", self._make_queries_rewrite_planning)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("summarize_output", self._summarize_output)
        
        # Define edges
        workflow.add_edge(START, "normalize_input")
        workflow.add_edge("normalize_input", "route_mode")
        
        # Conditional edges based on mode
        workflow.add_conditional_edges(
            "route_mode",
            self._decide_route,
            {
                "direct": "make_queries_direct",
                "baseline": "make_queries_rewrite_baseline",
                "planning": "make_plan",
            }
        )
        
        # Planning mode: plan -> rewrite -> retrieve
        workflow.add_edge("make_plan", "make_queries_rewrite_planning")
        
        # All modes converge to retrieve
        workflow.add_edge("make_queries_direct", "retrieve")
        workflow.add_edge("make_queries_rewrite_baseline", "retrieve")
        workflow.add_edge("make_queries_rewrite_planning", "retrieve")
        
        # Retrieve -> summarize -> end
        workflow.add_edge("retrieve", "summarize_output")
        workflow.add_edge("summarize_output", END)
        
        return workflow.compile()
    
    def _decide_route(self, state: RAGCompareState) -> str:
        """Decide which route to take based on mode"""
        return state["mode"]
    
    # ========================================================================
    # Node: normalize_input
    # ========================================================================
    def _normalize_input(self, state: RAGCompareState) -> Dict[str, Any]:
        """Normalize and validate input"""
        user_input = state["user_input"].strip()
        logs = state.get("logs", [])
        logs.append(f"[normalize_input] Input length: {len(user_input)} chars")
        
        return {
            "user_input": user_input,
            "logs": logs,
        }
    
    # ========================================================================
    # Node: route_mode
    # ========================================================================
    def _route_mode(self, state: RAGCompareState) -> Dict[str, Any]:
        """Route to appropriate processing based on mode"""
        mode = state["mode"]
        logs = state.get("logs", [])
        logs.append(f"[route_mode] Mode: {mode}")
        return {"logs": logs}
    
    # ========================================================================
    # Node: make_plan (planning mode only)
    # ========================================================================
    def _make_plan(self, state: RAGCompareState) -> Dict[str, Any]:
        """Generate a simple plan for query rewriting"""
        user_input = state["user_input"]
        logs = state.get("logs", [])
        
        planning_prompt = f"""Analyze the following medical question and extract key information for retrieval planning.

Question: {user_input}

Extract:
1. observed_features: List 3-7 key symptoms, findings, or conditions mentioned
2. must_check_cooccurrence: List pairs of features whose co-occurrence is important
3. need_disambiguation: List any confusing concepts or disease pairs that need distinction

Output JSON only. Do NOT provide diagnosis or conclusions."""

        try:
            messages = [
                SystemMessage(content=MIRAGE_SYSTEM_PROMPT),
                HumanMessage(content=planning_prompt)
            ]
            
            result = self.llm_structured.invoke(messages)
            plan = {
                "observed_features": result.observed_features,
                "must_check_cooccurrence": result.must_check_cooccurrence,
                "need_disambiguation": result.need_disambiguation,
            }
            logs.append(f"[make_plan] Extracted {len(plan['observed_features'])} features")
            
        except Exception as e:
            logs.append(f"[make_plan] Error: {str(e)}, using fallback plan")
            # Fallback: extract simple keywords
            words = re.findall(r'\b[A-Za-z]{4,}\b', user_input)
            plan = {
                "observed_features": words[:5] if words else ["symptom"],
                "must_check_cooccurrence": [],
                "need_disambiguation": [],
            }
        
        return {"plan": plan, "logs": logs}
    
    # ========================================================================
    # Node: make_queries_direct (direct mode)
    # ========================================================================
    def _make_queries_direct(self, state: RAGCompareState) -> Dict[str, Any]:
        """Direct mode: use user input as-is"""
        user_input = state["user_input"]
        logs = state.get("logs", [])
        
        final_queries = [user_input]
        logs.append(f"[make_queries_direct] Using original query")
        
        return {
            "rewritten_queries": None,
            "final_queries": final_queries,
            "logs": logs,
        }
    
    # ========================================================================
    # Node: make_queries_rewrite_baseline (baseline mode)
    # ========================================================================
    def _make_queries_rewrite_baseline(self, state: RAGCompareState) -> Dict[str, Any]:
        """Baseline mode: LLM rewrites query without planning"""
        user_input = state["user_input"]
        logs = state.get("logs", [])
        
        # Use multi-query prompt
        prompt = MULTI_QUERY_PROMPT.format(query=user_input)
        
        rewritten_queries = self._generate_subqueries(prompt, logs, fallback_source=user_input)
        
        logs.append(f"[make_queries_rewrite_baseline] Generated {len(rewritten_queries)} queries")
        
        return {
            "rewritten_queries": rewritten_queries,
            "final_queries": rewritten_queries,
            "logs": logs,
        }
    
    # ========================================================================
    # Node: make_queries_rewrite_planning (planning mode)
    # ========================================================================
    def _make_queries_rewrite_planning(self, state: RAGCompareState) -> Dict[str, Any]:
        """Planning mode: use plan to enhance query rewriting"""
        user_input = state["user_input"]
        plan = state.get("plan", {})
        logs = state.get("logs", [])
        
        # Create plan summary
        features = plan.get("observed_features", [])
        cooccurrence = plan.get("must_check_cooccurrence", [])
        
        features_str = ", ".join(features[:5]) if features else "N/A"
        cooccurrence_str = " & ".join([f"({a}&{b})" for a, b in cooccurrence[:2]]) if cooccurrence else ""
        
        plan_summary = f"Key features: {features_str}"
        if cooccurrence_str:
            plan_summary += f" | Must-check: {cooccurrence_str}"
        
        # Combine user input with plan summary for query generation
        enhanced_query = f"User question: {user_input} | {plan_summary}"
        
        # Use multi-query prompt with enhanced query
        prompt = MULTI_QUERY_PROMPT.format(query=enhanced_query)
        
        rewritten_queries = self._generate_subqueries(
            prompt, logs, 
            fallback_source=features if features else user_input
        )
        
        logs.append(f"[make_queries_rewrite_planning] Generated {len(rewritten_queries)} queries with plan")
        
        return {
            "rewritten_queries": rewritten_queries,
            "final_queries": rewritten_queries,
            "logs": logs,
        }
    
    def _generate_subqueries(
        self, 
        prompt: str, 
        logs: List[str], 
        fallback_source: Any,
        max_retries: int = 1
    ) -> List[str]:
        """Generate sub-queries using LLM with retry and fallback"""
        
        for attempt in range(max_retries + 1):
            try:
                messages = [
                    SystemMessage(content=MIRAGE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ]
                
                response = self.llm.invoke(messages)
                content = response.content
                
                # Parse "Sub-query N:" format
                queries = self._parse_subqueries(content)
                
                if len(queries) >= 2:
                    return queries[:3]  # Return max 3 queries
                
                logs.append(f"[generate_subqueries] Attempt {attempt+1}: parsed {len(queries)} queries, retrying...")
                
            except Exception as e:
                logs.append(f"[generate_subqueries] Attempt {attempt+1} error: {str(e)}")
        
        # Fallback
        logs.append("[generate_subqueries] Using fallback query generation")
        return self._fallback_queries(fallback_source)
    
    def _parse_subqueries(self, content: str) -> List[str]:
        """Parse sub-queries from LLM response"""
        queries = []
        
        # Try "Sub-query N:" pattern
        pattern = r'Sub-query\s*\d+\s*:\s*(.+?)(?=Sub-query\s*\d+\s*:|$)'
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            query = match.strip()
            if query and len(query) > 5:
                queries.append(query)
        
        # Alternative: numbered list
        if len(queries) < 2:
            pattern2 = r'^\d+[.)]\s*(.+)$'
            for line in content.split('\n'):
                match = re.match(pattern2, line.strip())
                if match:
                    query = match.group(1).strip()
                    if query and len(query) > 5:
                        queries.append(query)
        
        return queries
    
    def _fallback_queries(self, source: Any) -> List[str]:
        """Generate fallback queries when LLM fails"""
        if isinstance(source, list):
            # Use features from planning
            features = source[:5]
            queries = [
                f"What is {features[0]}?" if features else "medical condition",
                f"{' '.join(features[:2])} symptoms" if len(features) >= 2 else "clinical findings",
                f"{' '.join(features[:3])} diagnosis treatment" if len(features) >= 3 else "medical diagnosis",
            ]
        else:
            # Use keywords from user input
            text = str(source)
            words = re.findall(r'\b[A-Za-z]{4,}\b', text)
            if len(words) >= 3:
                queries = [
                    f"{words[0]} {words[1]} medical",
                    f"{words[1]} {words[2]} symptoms",
                    f"{words[0]} {words[2]} diagnosis",
                ]
            else:
                queries = [text, f"{text} symptoms", f"{text} diagnosis"]
        
        return queries[:3]
    
    # ========================================================================
    # Node: retrieve
    # ========================================================================
    def _retrieve(self, state: RAGCompareState) -> Dict[str, Any]:
        """Retrieve documents for all queries"""
        final_queries = state["final_queries"]
        top_k = state.get("top_k", 5)
        mode = state["mode"]
        logs = state.get("logs", [])
        
        all_docs = []
        doc_scores = {}  # For fusion
        doc_data = {}
        
        for query in final_queries:
            try:
                docs, scores = self.retriever.retrieve(query, k=top_k)
                
                for doc, score in zip(docs, scores):
                    doc_id = doc.get("id", doc.get("title", str(hash(doc.get("content", "")[:100]))))
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                        doc_data[doc_id] = doc.copy()
                        doc_data[doc_id]["query_trace"] = []
                    
                    doc_scores[doc_id] += score
                    doc_data[doc_id]["query_trace"].append(query)
                    
            except Exception as e:
                logs.append(f"[retrieve] Error for query '{query[:30]}...': {str(e)}")
        
        # Sort by fused score and take top 10
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        merged_top_k = 10
        retrieved_docs = []
        for doc_id in sorted_ids[:merged_top_k]:
            doc = doc_data[doc_id]
            doc["fused_score"] = doc_scores[doc_id]
            retrieved_docs.append(doc)
        
        logs.append(f"[retrieve] Retrieved {len(retrieved_docs)} docs (fused from {len(final_queries)} queries)")
        
        return {
            "retrieved_docs": retrieved_docs,
            "logs": logs,
        }
    
    # ========================================================================
    # Node: summarize_output
    # ========================================================================
    def _summarize_output(self, state: RAGCompareState) -> Dict[str, Any]:
        """Generate summary with metrics"""
        logs = state.get("logs", [])
        logs.append("[summarize_output] Generating summary")
        return {"logs": logs}
    
    # ========================================================================
    # Public API
    # ========================================================================
    def run(
        self,
        user_input: str,
        mode: Literal["direct", "baseline", "planning"],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run the RAG comparison for a single input.
        
        Args:
            user_input: The medical question
            mode: One of "direct", "baseline", "planning"
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dict with mode, plan, final_queries, retrieved_docs, and metrics
        """
        initial_state: RAGCompareState = {
            "user_input": user_input,
            "mode": mode,
            "plan": None,
            "rewritten_queries": None,
            "final_queries": [],
            "retrieved_docs": [],
            "top_k": top_k,
            "logs": [],
        }
        
        result = self.graph.invoke(initial_state)
        
        # Compute metrics
        metrics = self._compute_metrics(result)
        
        return {
            "mode": result["mode"],
            "user_input": result["user_input"],
            "plan": result.get("plan"),
            "rewritten_queries": result.get("rewritten_queries"),
            "final_queries": result["final_queries"],
            "retrieved_docs": result["retrieved_docs"],
            "metrics": metrics,
            "logs": result["logs"],
        }
    
    def _compute_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comparison metrics"""
        final_queries = state.get("final_queries", [])
        retrieved_docs = state.get("retrieved_docs", [])
        plan = state.get("plan")
        mode = state.get("mode")
        
        metrics = {
            "num_queries": len(final_queries),
            "query_lengths": [len(q) for q in final_queries],
            "num_retrieved_docs": len(retrieved_docs),
        }
        
        # Planning-specific metrics
        if mode == "planning" and plan:
            features = plan.get("observed_features", [])
            cooccurrence = plan.get("must_check_cooccurrence", [])
            
            # Check if features appear in queries
            feature_in_queries = {}
            for feature in features:
                feature_lower = feature.lower()
                found = any(feature_lower in q.lower() for q in final_queries)
                feature_in_queries[feature] = found
            
            metrics["features_in_queries"] = feature_in_queries
            metrics["feature_coverage"] = sum(feature_in_queries.values()) / len(features) if features else 0
            
            # Check co-occurrence in docs
            if cooccurrence and retrieved_docs:
                cooccurrence_docs = []
                for pair in cooccurrence:
                    if len(pair) >= 2:
                        a, b = pair[0].lower(), pair[1].lower()
                        count = sum(
                            1 for doc in retrieved_docs
                            if a in doc.get("content", "").lower() and b in doc.get("content", "").lower()
                        )
                        cooccurrence_docs.append({
                            "pair": pair,
                            "docs_with_both": count,
                            "ratio": count / len(retrieved_docs) if retrieved_docs else 0,
                        })
                metrics["cooccurrence_coverage"] = cooccurrence_docs
        
        return metrics


def create_rag_compare_graph(**kwargs) -> RAGCompareGraph:
    """Factory function to create RAG comparison graph"""
    return RAGCompareGraph(**kwargs)


# ============================================================================
# Main (for testing)
# ============================================================================
if __name__ == "__main__":
    # Simple test
    graph = create_rag_compare_graph()
    
    test_input = "A 45-year-old man presents with fatigue, increased thirst, and frequent urination. His fasting blood glucose is 142 mg/dL. What could be the diagnosis?"
    
    print("=" * 70)
    print("Testing RAG Compare Graph")
    print("=" * 70)
    
    for mode in ["direct", "baseline", "planning"]:
        print(f"\n--- Mode: {mode} ---")
        result = graph.run(test_input, mode=mode, top_k=3)
        
        print(f"Final queries: {result['final_queries']}")
        print(f"Retrieved docs: {len(result['retrieved_docs'])}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
