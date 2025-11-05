"""
Hybrid Retrieval Module: FAISS (dense) + BM25 (sparse) fusion
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import torch


class HybridRetriever:
    """
    Hybrid retrieval combining dense (FAISS) and sparse (BM25) methods.
    Uses 0.6/0.4 fusion weights by default.
    """
    
    def __init__(
        self,
        dense_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        device: str = "cuda",
        index_type: str = "faiss"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_model_name: Sentence transformer model for dense retrieval
            dense_weight: Weight for dense retrieval scores (0.6)
            sparse_weight: Weight for sparse retrieval scores (0.4)
            device: Device to run models on
            index_type: Type of dense index ("faiss")
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize dense retriever
        self.dense_model = SentenceTransformer(dense_model_name, device=device)
        self.embedding_dim = self.dense_model.get_sentence_embedding_dimension()
        self.device = device
        
        # Initialize sparse retriever (BM25)
        self.bm25 = None
        self.corpus_tokenized = None
        
        # FAISS index
        self.faiss_index = None
        self.corpus = None
        self.corpus_embeddings = None
        
    def build_index(self, corpus: List[str], tokenize_corpus: bool = True):
        """
        Build both dense and sparse indices.
        
        Args:
            corpus: List of documents to index
            tokenize_corpus: Whether to tokenize for BM25
        """
        self.corpus = corpus
        
        # Build dense index (FAISS)
        corpus_embeddings = self.dense_model.encode(
            corpus,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        self.corpus_embeddings = corpus_embeddings
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(corpus_embeddings)
        self.faiss_index.add(corpus_embeddings.astype('float32'))
        
        # Build sparse index (BM25)
        if tokenize_corpus:
            # Simple tokenization for BM25
            self.corpus_tokenized = [doc.lower().split() for doc in corpus]
        else:
            self.corpus_tokenized = corpus
        
        self.bm25 = BM25Okapi(self.corpus_tokenized)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        return_scores: bool = False
    ) -> List[Tuple[int, str, Optional[float]]]:
        """
        Retrieve documents using hybrid retrieval.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            return_scores: Whether to return fusion scores
        
        Returns:
            List of (doc_id, document, score) tuples
        """
        if self.faiss_index is None or self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Dense retrieval
        query_embedding = self.dense_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        dense_scores, dense_indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
        
        # Normalize dense scores to [0, 1]
        dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        
        # Sparse retrieval (BM25)
        query_tokenized = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokenized)
        
        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores_norm = bm25_scores
        
        # Fusion: combine scores
        fusion_scores = (
            self.dense_weight * dense_scores_norm +
            self.sparse_weight * bm25_scores_norm
        )
        
        # Get top-k by fusion score
        top_indices = np.argsort(fusion_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = int(idx)
            doc = self.corpus[doc_id]
            score = fusion_scores[idx] if return_scores else None
            results.append((doc_id, doc, score))
        
        return results
    
    def retrieve_dense_only(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[int, str]]:
        """Retrieve using only dense retrieval."""
        query_embedding = self.dense_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        results = []
        for idx in indices[0]:
            doc_id = int(idx)
            doc = self.corpus[doc_id]
            results.append((doc_id, doc))
        
        return results
    
    def retrieve_sparse_only(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[int, str]]:
        """Retrieve using only sparse retrieval."""
        query_tokenized = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokenized)
        
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = int(idx)
            doc = self.corpus[doc_id]
            results.append((doc_id, doc))
        
        return results

