"""
Pydantic response schemas for API responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PassageResult(BaseModel):
    """Schema for a retrieved passage."""
    
    doc_id: int = Field(..., description="Document ID in corpus")
    text: str = Field(..., description="Passage text")
    score: float = Field(..., description="Retrieval/reranking score")


class ClaimDetail(BaseModel):
    """Schema for a verification claim."""
    
    claim_id: str = Field(..., description="Claim identifier")
    text: str = Field(..., description="Claim text")
    entailment: str = Field(..., description="Entailment label (ENTAILMENT/CONTRADICTION/UNVERIFIED)")
    confidence: float = Field(..., description="Confidence score")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence passage IDs")


class VerificationDetails(BaseModel):
    """Schema for verification results."""
    
    verified: bool = Field(..., description="Whether answer is verified")
    total_claims: int = Field(..., description="Total number of claims")
    verified_claims: int = Field(..., description="Number of verified claims")
    hallucinated_claims: int = Field(..., description="Number of hallucinated claims")
    avg_confidence: float = Field(..., description="Average confidence score")
    revision_cycles: int = Field(..., description="Number of revision cycles")
    reason: str = Field(..., description="Verification reason/explanation")
    claims: List[ClaimDetail] = Field(default_factory=list, description="Detailed claim information")


class EmbedResponse(BaseModel):
    """Response schema for embedding generation."""
    
    embeddings: List[List[float]] = Field(..., description="Embedding vectors")
    dimension: int = Field(..., description="Embedding dimension")
    num_texts: int = Field(..., description="Number of texts embedded")
    
    class Config:
        schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3]],
                "dimension": 768,
                "num_texts": 1
            }
        }


class RetrieveResponse(BaseModel):
    """Response schema for retrieval."""
    
    query: str = Field(..., description="Original query")
    passages: List[PassageResult] = Field(..., description="Retrieved passages")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    reranking_time_ms: Optional[float] = Field(None, description="Reranking time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "passages": [
                    {
                        "doc_id": 0,
                        "text": "Paris is the capital of France.",
                        "score": 0.95
                    }
                ],
                "retrieval_time_ms": 123.45,
                "reranking_time_ms": 45.67
            }
        }


class GenerateResponse(BaseModel):
    """Response schema for RAG generation."""
    
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    status: str = Field(..., description="Response status (VERIFIED/ABSTAINED)")
    sources: List[PassageResult] = Field(..., description="Source passages")
    verification: Optional[VerificationDetails] = Field(None, description="Verification details")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    verification_time_ms: Optional[float] = Field(None, description="Verification time in milliseconds")
    total_time_ms: float = Field(..., description="Total pipeline time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "When was the Eiffel Tower built?",
                "answer": "The Eiffel Tower was built from 1887 to 1889.",
                "status": "VERIFIED",
                "sources": [
                    {
                        "doc_id": 1,
                        "text": "The Eiffel Tower was constructed from 1887 to 1889.",
                        "score": 0.95
                    }
                ],
                "verification": {
                    "verified": True,
                    "total_claims": 1,
                    "verified_claims": 1,
                    "hallucinated_claims": 0,
                    "avg_confidence": 0.92,
                    "revision_cycles": 0,
                    "reason": "All claims verified",
                    "claims": []
                },
                "retrieval_time_ms": 123.45,
                "generation_time_ms": 234.56,
                "verification_time_ms": 45.67,
                "total_time_ms": 403.68
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    corpus_size: Optional[int] = Field(None, description="Number of documents in corpus")
    device: str = Field(..., description="Device being used (cuda/cpu)")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": True,
                "corpus_size": 1000,
                "device": "cuda"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "detail": "top_k must be greater than 0"
            }
        }
