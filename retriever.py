#!/usr/bin/env python3
"""
Retriever wrapper for the Multi-Subquery RAG System

This module provides a unified interface for document retrieval,
wrapping the MIRAGE retrieval system.
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Add MIRAGE paths for retrieval
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, 'MIRAGE'))
sys.path.insert(0, os.path.join(_BASE_DIR, 'MIRAGE', 'MedRAG'))
sys.path.insert(0, os.path.join(_BASE_DIR, 'MIRAGE', 'MedRAG', 'src'))


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (documents, scores)
            - documents: List of dicts with 'title' and 'content' keys
            - scores: List of relevance scores
        """
        pass
    
    @abstractmethod
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
    ) -> List[Tuple[List[Dict[str, Any]], List[float]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of documents to retrieve per query
            
        Returns:
            List of (documents, scores) tuples, one per query
        """
        pass


class MIRAGERetriever(BaseRetriever):
    """
    Retriever using MIRAGE's retrieval system
    
    Supports multiple retriever models (MedCPT, BM25, Contriever, SPECTER)
    and multiple corpora (Textbooks, PubMed, StatPearls, Wikipedia).
    """
    
    def __init__(
        self,
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        db_dir: Optional[str] = None,
        cache: bool = True,
        HNSW: bool = False,
    ):
        """
        Initialize the MIRAGE retriever.
        
        Args:
            retriever_name: Name of retriever ("MedCPT", "BM25", "Contriever", "SPECTER", "RRF-2", "RRF-4")
            corpus_name: Name of corpus ("Textbooks", "PubMed", "StatPearls", "Wikipedia", "MedText", "MedCorp")
            db_dir: Directory containing corpus data
            cache: Whether to cache documents in memory
            HNSW: Whether to use HNSW index for faster search
        """
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir or os.path.join(_BASE_DIR, 'MIRAGE', 'MedRAG', 'corpus')
        self.cache = cache
        self.HNSW = HNSW
        
        self._retrieval_system = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization of the retrieval system"""
        if self._initialized:
            return
        
        try:
            from MedRAG.src.utils import RetrievalSystem
            
            self._retrieval_system = RetrievalSystem(
                retriever_name=self.retriever_name,
                corpus_name=self.corpus_name,
                db_dir=self.db_dir,
                cache=self.cache,
                HNSW=self.HNSW,
            )
            self._initialized = True
            print(f"✓ MIRAGE retriever initialized: {self.retriever_name} on {self.corpus_name}")
        except ImportError as e:
            print(f"✗ Failed to import MIRAGE retrieval system: {e}")
            print("  Make sure MedRAG is accessible in your PYTHONPATH.")
            raise RuntimeError(f"MedRAG import failed: {e}")
        except Exception as e:
            print(f"✗ Failed to initialize retrieval system: {e}")
            raise RuntimeError(f"Initialization failed: {e}")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        rrf_k: int = 100,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            rrf_k: RRF parameter for multi-retriever fusion
            
        Returns:
            Tuple of (documents, scores)
        """
        self._lazy_init()
        
        if self._retrieval_system is None:
            raise RuntimeError("Retrieval system is not initialized.")
        
        try:
            documents, scores = self._retrieval_system.retrieve(
                query, k=k, rrf_k=rrf_k
            )
            return documents, scores
        except Exception as e:
            print(f"Warning: Retrieval failed for query '{query[:50]}...': {e}")
            return [], []
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        rrf_k: int = 100,
    ) -> List[Tuple[List[Dict[str, Any]], List[float]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of documents to retrieve per query
            rrf_k: RRF parameter for multi-retriever fusion
            
        Returns:
            List of (documents, scores) tuples
        """
        results = []
        for query in queries:
            docs, scores = self.retrieve(query, k=k, rrf_k=rrf_k)
            results.append((docs, scores))
        return results
    



class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining multiple retrieval methods
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from
    multiple retrievers.
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        rrf_k: int = 60,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            retrievers: List of retriever instances to combine
            rrf_k: RRF parameter (default 60 as in the original paper)
        """
        self.retrievers = retrievers
        self.rrf_k = rrf_k
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve and fuse documents from multiple retrievers.
        """
        all_results = []
        
        # Get results from all retrievers
        for retriever in self.retrievers:
            docs, scores = retriever.retrieve(query, k=k * 2)  # Get more for fusion
            all_results.append((docs, scores))
        
        # Fuse results using RRF
        return self._rrf_fusion(all_results, k)
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
    ) -> List[Tuple[List[Dict[str, Any]], List[float]]]:
        """
        Retrieve documents for multiple queries.
        """
        return [self.retrieve(query, k) for query in queries]
    
    def _rrf_fusion(
        self,
        results: List[Tuple[List[Dict[str, Any]], List[float]]],
        k: int,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (rrf_k + rank_i)) across all retrievers
        """
        doc_scores = {}
        doc_data = {}
        
        for docs, _ in results:
            for rank, doc in enumerate(docs):
                doc_id = doc.get("id", doc.get("title", str(hash(doc.get("content", "")))))
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_data[doc_id] = doc
                
                # RRF formula
                doc_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        top_ids = sorted_ids[:k]
        top_docs = [doc_data[doc_id] for doc_id in top_ids]
        top_scores = [doc_scores[doc_id] for doc_id in top_ids]
        
        return top_docs, top_scores


def create_retriever(
    retriever_type: str = "mirage",
    **kwargs,
) -> BaseRetriever:
    """
    Factory function to create a retriever.
    
    Args:
        retriever_type: Type of retriever ("mirage", "hybrid")
        **kwargs: Additional arguments for the retriever
        
    Returns:
        A retriever instance
    """
    if retriever_type == "mirage":
        return MIRAGERetriever(**kwargs)
    elif retriever_type == "hybrid":
        # Create multiple MIRAGE retrievers for hybrid
        retrievers = [
            MIRAGERetriever(retriever_name="MedCPT", **kwargs),
            MIRAGERetriever(retriever_name="BM25", **kwargs),
        ]
        return HybridRetriever(retrievers, rrf_k=kwargs.get("rrf_k", 60))
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


if __name__ == "__main__":
    # Test the retriever
    retriever = create_retriever(
        retriever_type="mirage",
        retriever_name="MedCPT",
        corpus_name="Textbooks",
    )
    
    query = "What are the symptoms of diabetes?"
    docs, scores = retriever.retrieve(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(docs)} documents:")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"\n[{i+1}] Score: {score:.4f}")
        print(f"    Title: {doc.get('title', 'N/A')}")
        print(f"    Content: {doc.get('content', 'N/A')[:200]}...")

