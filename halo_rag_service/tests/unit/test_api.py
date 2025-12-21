"""
Unit tests for API endpoints.
Tests request validation, response schemas, and error handling.
"""

import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_success(self, test_client):
        """Test successful health check."""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "device" in data
        assert data["models_loaded"] is True
    
    def test_health_check_response_schema(self, test_client):
        """Test health check response schema."""
        response = test_client.get("/health")
        data = response.json()
        
        required_fields = ["status", "models_loaded", "device"]
        for field in required_fields:
            assert field in data


class TestEmbedEndpoint:
    """Tests for /embed endpoint."""
    
    def test_embed_success(self, test_client, sample_embed_request):
        """Test successful embedding generation."""
        response = test_client.post("/embed", json=sample_embed_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "embeddings" in data
        assert "dimension" in data
        assert "num_texts" in data
        assert data["num_texts"] == len(sample_embed_request["texts"])
    
    def test_embed_empty_texts(self, test_client):
        """Test embedding with empty texts list."""
        response = test_client.post("/embed", json={"texts": []})
        
        # Should fail validation (min_items=1)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_embed_invalid_request(self, test_client):
        """Test embedding with invalid request."""
        response = test_client.post("/embed", json={"invalid": "field"})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestRetrieveEndpoint:
    """Tests for /retrieve endpoint."""
    
    def test_retrieve_success(self, test_client, sample_retrieve_request):
        """Test successful retrieval."""
        response = test_client.post("/retrieve", json=sample_retrieve_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "query" in data
        assert "passages" in data
        assert "retrieval_time_ms" in data
        assert "reranking_time_ms" in data
        assert data["query"] == sample_retrieve_request["query"]
    
    def test_retrieve_invalid_top_k(self, test_client):
        """Test retrieval with invalid top_k."""
        request = {
            "query": "test query",
            "top_k": 0,  # Invalid: must be > 0
            "rerank_k": 5
        }
        response = test_client.post("/retrieve", json=request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrieve_rerank_k_greater_than_top_k(self, test_client):
        """Test retrieval with rerank_k > top_k."""
        request = {
            "query": "test query",
            "top_k": 5,
            "rerank_k": 10  # Invalid: rerank_k > top_k
        }
        response = test_client.post("/retrieve", json=request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_retrieve_empty_query(self, test_client):
        """Test retrieval with empty query."""
        request = {
            "query": "",  # Invalid: min_length=1
            "top_k": 20,
            "rerank_k": 5
        }
        response = test_client.post("/retrieve", json=request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""
    
    def test_generate_success(self, test_client, sample_generate_request):
        """Test successful generation."""
        response = test_client.post("/generate", json=sample_generate_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "query" in data
        assert "answer" in data
        assert "status" in data
        assert "sources" in data
        assert "verification" in data
        assert "total_time_ms" in data
        assert data["query"] == sample_generate_request["query"]
    
    def test_generate_with_custom_params(self, test_client):
        """Test generation with custom parameters."""
        request = {
            "query": "What is RAG?",
            "top_k_retrieve": 10,
            "top_k_rerank": 3,
            "enable_verification": False,
            "temperature": 0.5,
            "max_new_tokens": 128
        }
        response = test_client.post("/generate", json=request)
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_generate_invalid_temperature(self, test_client):
        """Test generation with invalid temperature."""
        request = {
            "query": "test query",
            "temperature": 3.0  # Invalid: must be <= 2.0
        }
        response = test_client.post("/generate", json=request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_generate_verification_details(self, test_client, sample_generate_request):
        """Test that verification details are included."""
        response = test_client.post("/generate", json=sample_generate_request)
        data = response.json()
        
        if data.get("verification"):
            verification = data["verification"]
            assert "verified" in verification
            assert "total_claims" in verification
            assert "verified_claims" in verification
            assert "hallucinated_claims" in verification


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "status" in data
