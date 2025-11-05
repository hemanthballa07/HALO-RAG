"""
Cross-Encoder Reranking Module using DeBERTa-v3-base on MS MARCO
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional
import torch


class CrossEncoderReranker:
    """
    Cross-encoder reranker for reordering retrieved documents.
    Uses DeBERTa-v3-base fine-tuned on MS MARCO.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
        max_length: int = 512
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to run model on
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents for a given query.
        
        Args:
            query: Query string
            documents: List of document strings
            top_k: Number of top documents to return (None = all)
        
        Returns:
            List of (original_index, document, score) tuples sorted by score
        """
        if len(documents) == 0:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Create results with original indices
        results = [
            (idx, doc, float(score))
            for idx, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def rerank_with_ids(
        self,
        query: str,
        document_ids: List[int],
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents with original document IDs preserved.
        
        Args:
            query: Query string
            document_ids: List of original document IDs
            documents: List of document strings
            top_k: Number of top documents to return
        
        Returns:
            List of (doc_id, document, score) tuples sorted by score
        """
        if len(documents) != len(document_ids):
            raise ValueError("document_ids and documents must have same length")
        
        # Rerank
        reranked = self.rerank(query, documents, top_k=None)
        
        # Map back to original IDs
        results = [
            (document_ids[orig_idx], doc, score)
            for orig_idx, doc, score in reranked
        ]
        
        if top_k is not None:
            results = results[:top_k]
        
        return results

