"""
Retrieval service for hybrid retrieval and reranking.
"""

from typing import List, Tuple
import time

from ..core.models import ModelLoader
from ..core.logging import get_logger
from ..schemas.responses import PassageResult

logger = get_logger(__name__)


class RetrievalService:
    """Service for document retrieval and reranking."""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize retrieval service.
        
        Args:
            model_loader: Model loader instance
        """
        self.model_loader = model_loader
    
    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 20,
        rerank_k: int = 5
    ) -> Tuple[List[PassageResult], float, float]:
        """
        Retrieve and rerank documents.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            rerank_k: Number of documents to keep after reranking
        
        Returns:
            Tuple of (passages, retrieval_time_ms, reranking_time_ms)
        """
        try:
            retriever = self.model_loader.get_retriever()
            reranker = self.model_loader.get_reranker()
            
            # Retrieval
            logger.info(f"Retrieving top-{top_k} documents for query: {query[:50]}...")
            retrieval_start = time.time()
            
            retrieved = retriever.retrieve(query, top_k=top_k, return_scores=True)
            
            retrieval_time = (time.time() - retrieval_start) * 1000
            logger.info(
                f"Retrieved {len(retrieved)} documents in {retrieval_time:.2f}ms",
                extra={
                    "retrieval_time_ms": retrieval_time,
                    "num_retrieved": len(retrieved)
                }
            )
            
            # Reranking
            logger.info(f"Reranking to top-{rerank_k} documents")
            reranking_start = time.time()
            
            # Extract passages and IDs
            passages = [doc[1] for doc in retrieved]
            doc_ids = [doc[0] for doc in retrieved]
            
            # Rerank
            reranked_results = reranker.rerank(query, passages, top_k=rerank_k)
            
            reranking_time = (time.time() - reranking_start) * 1000
            logger.info(
                f"Reranked to {len(reranked_results)} documents in {reranking_time:.2f}ms",
                extra={
                    "reranking_time_ms": reranking_time,
                    "num_reranked": len(reranked_results)
                }
            )
            
            # Build response
            passage_results = []
            for passage, score in reranked_results:
                # Find original doc_id
                doc_id = doc_ids[passages.index(passage)]
                passage_results.append(
                    PassageResult(
                        doc_id=doc_id,
                        text=passage,
                        score=float(score)
                    )
                )
            
            # Log score distribution
            scores = [p.score for p in passage_results]
            if scores:
                logger.info(
                    f"Reranking scores - min: {min(scores):.3f}, max: {max(scores):.3f}, avg: {sum(scores)/len(scores):.3f}",
                    extra={
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "avg_score": sum(scores) / len(scores)
                    }
                )
            
            return passage_results, retrieval_time, reranking_time
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
            raise
