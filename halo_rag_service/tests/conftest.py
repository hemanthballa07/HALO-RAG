"""
Pytest configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock
from typing import List

from app.main import app
from app.core.models import ModelLoader


@pytest.fixture
def mock_model_loader():
    """Mock model loader for testing without loading actual models."""
    mock_loader = Mock(spec=ModelLoader)
    mock_loader.is_loaded.return_value = True
    
    # Mock corpus
    mock_loader.get_corpus.return_value = [
        "Paris is the capital of France.",
        "The Eiffel Tower was built in 1889.",
        "The Seine flows through Paris."
    ]
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.embedding_dim = 768
    mock_retriever.dense_model.encode.return_value = [[0.1] * 768]
    mock_loader.get_retriever.return_value = mock_retriever
    
    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        ("Paris is the capital of France.", 0.95)
    ]
    mock_loader.get_reranker.return_value = mock_reranker
    
    # Mock generator
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Paris is the capital of France."
    mock_loader.get_generator.return_value = mock_generator
    
    # Mock verification controller
    mock_verification = MagicMock()
    mock_loader.get_verification_controller.return_value = mock_verification
    
    # Mock RAG pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.generate.return_value = {
        "answer": "Paris is the capital of France.",
        "retrieved_docs": [
            (0, "Paris is the capital of France.", 0.95)
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
        "retrieval_time_ms": 100.0,
        "generation_time_ms": 200.0,
        "verification_time_ms": 50.0
    }
    mock_loader.get_rag_pipeline.return_value = mock_pipeline
    
    return mock_loader


@pytest.fixture
def test_client(mock_model_loader, monkeypatch):
    """Test client with mocked dependencies."""
    # Mock the get_model_loader function
    def mock_get_model_loader():
        return mock_model_loader
    
    monkeypatch.setattr("app.api.dependencies.get_model_loader", mock_get_model_loader)
    monkeypatch.setattr("app.core.models.get_model_loader", mock_get_model_loader)
    
    # Create test client
    client = TestClient(app)
    return client


@pytest.fixture
def sample_embed_request():
    """Sample embedding request."""
    return {
        "texts": ["What is RAG?", "How does it work?"]
    }


@pytest.fixture
def sample_retrieve_request():
    """Sample retrieval request."""
    return {
        "query": "What is the capital of France?",
        "top_k": 20,
        "rerank_k": 5
    }


@pytest.fixture
def sample_generate_request():
    """Sample generation request."""
    return {
        "query": "What is the capital of France?",
        "top_k_retrieve": 20,
        "top_k_rerank": 5,
        "enable_verification": True,
        "enable_revision": True
    }
