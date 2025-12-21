"""
RAG service for full pipeline execution.
"""

from typing import List, Optional, Dict, Any
import time

from ..core.models import ModelLoader
from ..core.logging import get_logger
from ..schemas.responses import PassageResult, VerificationDetails, ClaimDetail

logger = get_logger(__name__)


class RAGService:
    """Service for full RAG pipeline with verification."""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize RAG service.
        
        Args:
            model_loader: Model loader instance
        """
        self.model_loader = model_loader
    
    def generate(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        enable_verification: bool = True,
        enable_revision: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer using full RAG pipeline.
        
        Args:
            query: Query string
            top_k_retrieve: Number of documents to retrieve
            top_k_rerank: Number of documents to rerank
            enable_verification: Enable verification
            enable_revision: Enable revision
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dictionary with answer, sources, verification, and timing
        """
        try:
            pipeline = self.model_loader.get_rag_pipeline()
            
            logger.info(f"Starting RAG pipeline for query: {query[:50]}...")
            pipeline_start = time.time()
            
            # Track individual stage times
            retrieval_time = 0
            generation_time = 0
            verification_time = 0
            
            # Run pipeline
            result = pipeline.generate(
                query=query,
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                **generation_kwargs
            )
            
            total_time = (time.time() - pipeline_start) * 1000
            
            # Extract timing from result if available
            if 'retrieval_time_ms' in result:
                retrieval_time = result['retrieval_time_ms']
            if 'generation_time_ms' in result:
                generation_time = result['generation_time_ms']
            if 'verification_time_ms' in result:
                verification_time = result['verification_time_ms']
            
            # Build sources
            sources = []
            if 'retrieved_docs' in result:
                for i, (doc_id, passage, score) in enumerate(result['retrieved_docs']):
                    sources.append(
                        PassageResult(
                            doc_id=doc_id,
                            text=passage,
                            score=float(score) if score is not None else 0.0
                        )
                    )
            
            # Build verification details
            verification_details = None
            if enable_verification and 'verification' in result:
                ver = result['verification']
                
                # Extract claims
                claims = []
                if 'claims' in ver:
                    for claim in ver['claims']:
                        claims.append(
                            ClaimDetail(
                                claim_id=claim.claim_id,
                                text=claim.text,
                                entailment=claim.entailment,
                                confidence=claim.confidence,
                                evidence_ids=claim.evidence_ids
                            )
                        )
                
                verification_details = VerificationDetails(
                    verified=ver.get('verified', False),
                    total_claims=ver.get('total_claims', 0),
                    verified_claims=ver.get('verified_claims', 0),
                    hallucinated_claims=ver.get('hallucinated_claims', 0),
                    avg_confidence=ver.get('avg_confidence', 0.0),
                    revision_cycles=ver.get('revision_cycles', 0),
                    reason=ver.get('reason', ''),
                    claims=claims
                )
                
                # Log verification metrics
                logger.info(
                    f"Verification: {ver.get('verified_claims', 0)}/{ver.get('total_claims', 0)} claims verified",
                    extra={
                        "verified": ver.get('verified', False),
                        "total_claims": ver.get('total_claims', 0),
                        "verified_claims": ver.get('verified_claims', 0),
                        "hallucinated_claims": ver.get('hallucinated_claims', 0),
                        "revision_cycles": ver.get('revision_cycles', 0)
                    }
                )
            
            # Determine status
            status = "VERIFIED"
            answer = result.get('answer', '')
            
            if enable_verification and verification_details:
                if not verification_details.verified:
                    status = "ABSTAINED"
                    answer = "I cannot verify this answer with the available evidence."
            
            # Log pipeline completion
            logger.info(
                f"Pipeline completed in {total_time:.2f}ms (status: {status})",
                extra={
                    "total_time_ms": total_time,
                    "retrieval_time_ms": retrieval_time,
                    "generation_time_ms": generation_time,
                    "verification_time_ms": verification_time,
                    "status": status
                }
            )
            
            return {
                "query": query,
                "answer": answer,
                "status": status,
                "sources": sources,
                "verification": verification_details,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "verification_time_ms": verification_time,
                "total_time_ms": total_time
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}", exc_info=True)
            raise
