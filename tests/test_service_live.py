"""Optional smoke test with the real models and a small local corpus."""

import os

import pytest
from fastapi.testclient import TestClient

from halo_rag_service.app import app


@pytest.mark.skipif(os.environ.get("HALO_RAG_RUN_LIVE") != "1", reason="requires local models")
def test_service_with_real_pipeline():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["corpus_size"] > 0

        response = client.post("/generate", json={
            "query": "Where is the Eiffel Tower?",
            "top_k_retrieve": 3,
            "top_k_rerank": 2,
            "max_new_tokens": 32,
            "max_revision_iterations": 0,
            "do_sample": False,
        })
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["answer"]
        assert result["sources"]
        assert result["status"] in {"verified", "unverified", "abstained"}

        focused = client.post("/generate", json={
            "query": "Where is the Eiffel Tower?",
            "top_k_retrieve": 3,
            "top_k_rerank": 2,
            "evidence_limit": 1,
            "abstain_if_unanswered": True,
            "max_new_tokens": 32,
        })
        assert focused.status_code == 200, focused.text
        focused_result = focused.json()
        assert len(focused_result["sources"]) == 1
        assert focused_result["status"] in {"verified", "unverified", "abstained"}
