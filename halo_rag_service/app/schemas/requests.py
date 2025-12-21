"""
Pydantic request schemas for API validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class EmbedRequest(BaseModel):
    """Request schema for embedding generation."""
    
    texts: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_items=1,
        max_items=100
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate that texts are not empty."""
        for text in v:
            if not text.strip():
                raise ValueError("Texts cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "texts": ["What is retrieval-augmented generation?"]
            }
        }


class RetrieveRequest(BaseModel):
    """Request schema for retrieval."""
    
    query: str = Field(
        ...,
        description="Query string",
        min_length=1,
        max_length=512
    )
    top_k: int = Field(
        default=20,
        description="Number of documents to retrieve",
        gt=0,
        le=100
    )
    rerank_k: int = Field(
        default=5,
        description="Number of documents to keep after reranking",
        gt=0,
        le=50
    )
    
    @validator('rerank_k')
    def validate_rerank_k(cls, v, values):
        """Ensure rerank_k <= top_k."""
        if 'top_k' in values and v > values['top_k']:
            raise ValueError("rerank_k must be <= top_k")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 20,
                "rerank_k": 5
            }
        }


class GenerateRequest(BaseModel):
    """Request schema for RAG generation."""
    
    query: str = Field(
        ...,
        description="Query string",
        min_length=1,
        max_length=512
    )
    top_k_retrieve: int = Field(
        default=20,
        description="Number of documents to retrieve",
        gt=0,
        le=100
    )
    top_k_rerank: int = Field(
        default=5,
        description="Number of documents to keep after reranking",
        gt=0,
        le=50
    )
    enable_verification: bool = Field(
        default=True,
        description="Enable entailment-based verification"
    )
    enable_revision: bool = Field(
        default=True,
        description="Enable adaptive revision if verification fails"
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate (overrides config)",
        gt=0,
        le=512
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (overrides config)",
        ge=0.0,
        le=2.0
    )
    do_sample: Optional[bool] = Field(
        default=None,
        description="Whether to use sampling (overrides config)"
    )
    num_beams: Optional[int] = Field(
        default=None,
        description="Number of beams for beam search (overrides config)",
        gt=0,
        le=10
    )
    
    @validator('top_k_rerank')
    def validate_rerank_k(cls, v, values):
        """Ensure top_k_rerank <= top_k_retrieve."""
        if 'top_k_retrieve' in values and v > values['top_k_retrieve']:
            raise ValueError("top_k_rerank must be <= top_k_retrieve")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "When was the Eiffel Tower built?",
                "top_k_retrieve": 20,
                "top_k_rerank": 5,
                "enable_verification": True,
                "enable_revision": True,
                "temperature": 0.7
            }
        }
