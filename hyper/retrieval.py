#!/usr/bin/env python3
"""
MedRAG retrieval interface for HYPER

This module provides a unified interface for document retrieval,
wrapping the MIRAGE retrieval system.
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Add MIRAGE paths for retrieval
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "MIRAGE"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "MIRAGE", "MedRAG"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "MIRAGE", "MedRAG", "src"))


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
        self.db_dir = db_dir or os.path.join(
            _PROJECT_ROOT, "MIRAGE", "MedRAG", "corpus"
        )
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
            print(
                f"✓ MIRAGE retriever initialized: {self.retriever_name} on {self.corpus_name}"
            )
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
            documents, scores = self._retrieval_system.retrieve(query, k=k, rrf_k=rrf_k)
            return documents, scores
        except Exception as e:
            raise RuntimeError("Document retrieval failed.") from e

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


def create_retriever(retriever_type="mirage", **kwargs):
    """Create the MedRAG retriever used by HYPER."""
    if retriever_type != "mirage":
        raise ValueError("HYPER uses the MedRAG retrieval backend.")
    return MIRAGERetriever(**kwargs)
