"""
Multi-Subquery RAG System using LangGraph

A LangGraph-based system for answering complex questions using:
1. Multi-subquery generation
2. Parallel document retrieval
3. Per-subquery response generation
4. Final answer synthesis
"""

from .multi_subquery_rag import MultiSubqueryRAG, GraphState, SubqueryResult
from .config import (
    MultiSubqueryRAGConfig,
    LLMConfig,
    RetrieverConfig,
    SubqueryConfig,
    get_config,
    PRESETS,
)
from .retriever import (
    BaseRetriever,
    MIRAGERetriever,
    HybridRetriever,
    create_retriever,
)

__all__ = [
    # Main classes
    "MultiSubqueryRAG",
    "GraphState",
    "SubqueryResult",
    # Config
    "MultiSubqueryRAGConfig",
    "LLMConfig",
    "RetrieverConfig",
    "SubqueryConfig",
    "get_config",
    "PRESETS",
    # Retrievers
    "BaseRetriever",
    "MIRAGERetriever",
    "HybridRetriever",
    "create_retriever",
]

__version__ = "0.1.0"

