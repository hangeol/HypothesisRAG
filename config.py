#!/usr/bin/env python3
"""
Configuration settings for the Multi-Subquery RAG System
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """Configuration for the LLM"""
    provider: str = "openai"  # "openai", "anthropic", "vllm"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    
    def __post_init__(self):
        # Try to load API key from environment if not provided
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class RetrieverConfig:
    """Configuration for the retrieval system"""
    retriever_name: str = "MedCPT"  # "MedCPT", "BM25", "Contriever", "SPECTER", "RRF-2", "RRF-4"
    corpus_name: str = "Textbooks"  # "Textbooks", "PubMed", "StatPearls", "Wikipedia", "MedText", "MedCorp"
    db_dir: Optional[str] = None
    cache: bool = True
    HNSW: bool = False
    
    def __post_init__(self):
        if self.db_dir is None:
            # Default to MIRAGE's corpus directory
            self.db_dir = os.path.join(
                os.path.dirname(__file__),
                '..', 'MIRAGE', 'MedRAG', 'corpus'
            )


@dataclass
class SubqueryConfig:
    """Configuration for subquery generation and processing"""
    num_subqueries: int = 3  # Number of subqueries to generate
    k_per_subquery: int = 5  # Documents to retrieve per subquery
    rrf_k: int = 100  # Parameter for Reciprocal Rank Fusion
    max_context_length: int = 8192  # Maximum context length for synthesis


@dataclass
class MultiSubqueryRAGConfig:
    """Main configuration for the Multi-Subquery RAG system"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    subquery: SubqueryConfig = field(default_factory=SubqueryConfig)
    
    # Logging and debugging
    verbose: bool = True
    save_intermediate_results: bool = False
    output_dir: str = "./outputs"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MultiSubqueryRAGConfig":
        """Create config from a dictionary"""
        llm_config = LLMConfig(**config_dict.get("llm", {}))
        retriever_config = RetrieverConfig(**config_dict.get("retriever", {}))
        subquery_config = SubqueryConfig(**config_dict.get("subquery", {}))
        
        return cls(
            llm=llm_config,
            retriever=retriever_config,
            subquery=subquery_config,
            verbose=config_dict.get("verbose", True),
            save_intermediate_results=config_dict.get("save_intermediate_results", False),
            output_dir=config_dict.get("output_dir", "./outputs"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "retriever": {
                "retriever_name": self.retriever.retriever_name,
                "corpus_name": self.retriever.corpus_name,
                "db_dir": self.retriever.db_dir,
                "cache": self.retriever.cache,
                "HNSW": self.retriever.HNSW,
            },
            "subquery": {
                "num_subqueries": self.subquery.num_subqueries,
                "k_per_subquery": self.subquery.k_per_subquery,
                "rrf_k": self.subquery.rrf_k,
                "max_context_length": self.subquery.max_context_length,
            },
            "verbose": self.verbose,
            "save_intermediate_results": self.save_intermediate_results,
            "output_dir": self.output_dir,
        }


# Preset configurations for common use cases
PRESETS = {
    "default": MultiSubqueryRAGConfig(),
    
    "medical_openai": MultiSubqueryRAGConfig(
        llm=LLMConfig(provider="openai", model_name="gpt-4o"),
        retriever=RetrieverConfig(retriever_name="MedCPT", corpus_name="Textbooks"),
        subquery=SubqueryConfig(num_subqueries=3, k_per_subquery=5),
    ),
    
    "medical_fast": MultiSubqueryRAGConfig(
        llm=LLMConfig(provider="openai", model_name="gpt-4o-mini"),
        retriever=RetrieverConfig(retriever_name="MedCPT", corpus_name="Textbooks"),
        subquery=SubqueryConfig(num_subqueries=2, k_per_subquery=3),
    ),
    
    "comprehensive": MultiSubqueryRAGConfig(
        llm=LLMConfig(provider="openai", model_name="gpt-4o"),
        retriever=RetrieverConfig(retriever_name="RRF-2", corpus_name="MedCorp"),
        subquery=SubqueryConfig(num_subqueries=5, k_per_subquery=10),
    ),
    
    "local_vllm": MultiSubqueryRAGConfig(
        llm=LLMConfig(provider="vllm", model_name="google/medgemma-4b-it"),
        retriever=RetrieverConfig(retriever_name="MedCPT", corpus_name="Textbooks"),
        subquery=SubqueryConfig(num_subqueries=3, k_per_subquery=5),
    ),
}


def get_config(preset: str = "default") -> MultiSubqueryRAGConfig:
    """Get a preset configuration"""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]

