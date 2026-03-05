import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DocumentReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda"):
        """
        Initialize the reranker model.
        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda' or 'cpu'. Can also specify 'cuda:1', etc.
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        print(f"Loading reranker model {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the model with sequence classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Reranker {self.model_name} loaded successfully.")

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 15, batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents for a given query.
        
        Args:
            query: The search query.
            docs: A list of document dictionaries containing 'content'.
            top_k: The number of documents to return after reranking.
            batch_size: Batch size for model inference to avoid OOM.
            
        Returns:
            The top_k reranked documents, with 'rerank_score' added to each dict.
        """
        if not docs:
            return []

        # Some rerankers (like MedCPT-Cross-Encoder) take (query, doc) pairs
        # BGE and Qwen3 also generally take (query, doc) pairs
        pairs = [[query, doc.get("content", "")] for doc in docs]
        
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, # Max length for most cross-encoders
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Extract scores (usually logit 0 for binary sequence classification, or a single raw score)
                scores = outputs.logits.view(-1,).float().cpu().numpy()
                all_scores.extend(scores.tolist())

        # Combine docs with their new scores
        scored_docs = []
        for doc, score in zip(docs, all_scores):
            new_doc = doc.copy()
            new_doc["rerank_score"] = float(score)
            scored_docs.append(new_doc)

        # Sort by rerank score descending
        sorted_docs = sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)
        
        return sorted_docs[:top_k]
