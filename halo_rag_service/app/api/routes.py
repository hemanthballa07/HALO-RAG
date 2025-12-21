"""
API route definitions for HALO-RAG service.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from ..schemas import (
    EmbedRequest, EmbedResponse,
    RetrieveRequest, RetrieveResponse,
    GenerateRequest, GenerateResponse,
    HealthResponse, ErrorResponse
)
from ..services import EmbeddingService, RetrievalService, RAGService
from ..core.config import settings
from ..core.logging import get_logger
from .dependencies import ModelLoaderDep, RequestIDDep

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check service health and model loading status"
)
async def health_check(
    model_loader: ModelLoaderDep,
    request_id: RequestIDDep
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status and model loading information.
    """
    logger.info("Health check requested")
    
    try:
        is_loaded = model_loader.is_loaded()
        corpus_size = len(model_loader.get_corpus()) if is_loaded else None
        
        return HealthResponse(
            status="healthy" if is_loaded else "initializing",
            models_loaded=is_loaded,
            corpus_size=corpus_size,
            device=settings.device
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            corpus_size=None,
            device=settings.device
        )


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Generate Embeddings",
    description="Generate dense embeddings for input texts",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def embed(
    request: EmbedRequest,
    model_loader: ModelLoaderDep,
    request_id: RequestIDDep
) -> EmbedResponse:
    """
    Generate embeddings for texts.
    
    Args:
        request: Embedding request with texts
        model_loader: Injected model loader
        request_id: Request ID for tracking
    
    Returns:
        Embedding vectors
    """
    logger.info(f"Embedding request for {len(request.texts)} texts")
    
    try:
        service = EmbeddingService(model_loader)
        embeddings, dimension = service.generate_embeddings(request.texts)
        
        return EmbedResponse(
            embeddings=embeddings,
            dimension=dimension,
            num_texts=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve and Rerank",
    description="Retrieve documents using hybrid retrieval and rerank with cross-encoder",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def retrieve(
    request: RetrieveRequest,
    model_loader: ModelLoaderDep,
    request_id: RequestIDDep
) -> RetrieveResponse:
    """
    Retrieve and rerank documents.
    
    Args:
        request: Retrieval request with query and parameters
        model_loader: Injected model loader
        request_id: Request ID for tracking
    
    Returns:
        Retrieved and reranked passages
    """
    logger.info(f"Retrieval request: {request.query[:50]}...")
    
    try:
        service = RetrievalService(model_loader)
        passages, retrieval_time, reranking_time = service.retrieve_and_rerank(
            query=request.query,
            top_k=request.top_k,
            rerank_k=request.rerank_k
        )
        
        return RetrieveResponse(
            query=request.query,
            passages=passages,
            retrieval_time_ms=retrieval_time,
            reranking_time_ms=reranking_time
        )
        
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="RAG Generation",
    description="Generate answer using full RAG pipeline with verification and revision",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def generate(
    request: GenerateRequest,
    model_loader: ModelLoaderDep,
    request_id: RequestIDDep
) -> GenerateResponse:
    """
    Generate answer using RAG pipeline.
    
    Args:
        request: Generation request with query and parameters
        model_loader: Injected model loader
        request_id: Request ID for tracking
    
    Returns:
        Generated answer with verification and sources
    """
    logger.info(f"Generation request: {request.query[:50]}...")
    
    try:
        service = RAGService(model_loader)
        
        # Build generation kwargs
        generation_kwargs = {}
        if request.max_new_tokens is not None:
            generation_kwargs['max_new_tokens'] = request.max_new_tokens
        if request.temperature is not None:
            generation_kwargs['temperature'] = request.temperature
        if request.do_sample is not None:
            generation_kwargs['do_sample'] = request.do_sample
        if request.num_beams is not None:
            generation_kwargs['num_beams'] = request.num_beams
        
        result = service.generate(
            query=request.query,
            top_k_retrieve=request.top_k_retrieve,
            top_k_rerank=request.top_k_rerank,
            enable_verification=request.enable_verification,
            enable_revision=request.enable_revision,
            **generation_kwargs
        )
        
        return GenerateResponse(**result)
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )
