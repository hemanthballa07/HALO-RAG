"""HTTP service tests use a fake pipeline, so no model download is needed."""

import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from halo_rag_service.app import create_app, load_default_pipeline


class FakePipeline:
    def __init__(self):
        self.device = "cpu"
        self.corpus = ["Paris is in France.", "London is in England."]
        self.calls = []
        self.failure = None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return {
            "query": kwargs["query"],
            "generated_text": "Paris is in France.",
            "reranked_docs": [0],
            "reranked_texts": [self.corpus[0]],
            "verification_results": {
                "verified": True,
                "num_entailed": 1,
                "num_total": 1,
                "entailment_rate": 1.0,
            },
            "verified": True,
            "abstained": False,
        }


def test_service_loads_pipeline_once_and_maps_current_result_format():
    pipeline = FakePipeline()
    loads = []

    def factory():
        loads.append(True)
        return pipeline

    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json() == {
            "status": "ready", "device": "cpu", "corpus_size": 2,
        }
        for _ in range(2):
            response = client.post("/generate", json={
                "query": "  Where is Paris?  ", "top_k_retrieve": 8,
                "top_k_rerank": 3, "max_new_tokens": 64,
            })
            assert response.status_code == 200
            assert response.json() == {
                "query": "Where is Paris?",
                "answer": "Paris is in France.",
                "status": "verified",
                "verified": True,
                "abstained": False,
                "sources": [{"text": "Paris is in France."}],
                "verification": {
                    "verified": True, "num_entailed": 1,
                    "num_total": 1, "entailment_rate": 1.0,
                },
            }
    assert loads == [True]
    assert len(pipeline.calls) == 2
    assert pipeline.calls[0]["top_k_retrieve"] == 8
    assert pipeline.calls[0]["max_new_tokens"] == 64


@pytest.mark.parametrize("payload", [
    {"query": "   "},
    {"query": "question", "top_k_retrieve": 0},
    {"query": "question", "top_k_retrieve": 3, "top_k_rerank": 4},
    {"query": "question", "max_new_tokens": 0},
])
def test_service_rejects_invalid_requests(payload):
    with TestClient(create_app(FakePipeline)) as client:
        assert client.post("/generate", json=payload).status_code == 422


def test_service_distinguishes_unverified_and_abstained_answers():
    pipeline = FakePipeline()
    original_generate = pipeline.generate

    def unverified(**kwargs):
        result = original_generate(**kwargs)
        result["verified"] = False
        result["verification_results"]["verified"] = False
        return result

    pipeline.generate = unverified
    with TestClient(create_app(lambda: pipeline)) as client:
        response = client.post("/generate", json={"query": "Where is Paris?"})
        assert response.json()["status"] == "unverified"

        def abstained(**kwargs):
            result = unverified(**kwargs)
            result["abstained"] = True
            return result

        pipeline.generate = abstained
        response = client.post("/generate", json={"query": "Where is Paris?"})
        assert response.json()["status"] == "abstained"


def test_service_hides_internal_errors():
    pipeline = FakePipeline()
    pipeline.failure = RuntimeError("private model path")
    with TestClient(create_app(lambda: pipeline)) as client:
        response = client.post("/generate", json={"query": "Where is Paris?"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Generation failed"}


def test_default_service_requires_a_real_corpus(monkeypatch):
    monkeypatch.delenv("HALO_RAG_CORPUS_PATH", raising=False)
    with pytest.raises(RuntimeError, match="HALO_RAG_CORPUS_PATH"):
        load_default_pipeline()


def test_default_service_uses_the_config_without_qlora(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("Paris is in France.\n\nLondon is in England.\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiments:\n  device: cpu\nrevision:\n  max_iterations: 2\n",
                           encoding="utf-8")
    monkeypatch.setenv("HALO_RAG_CORPUS_PATH", str(corpus_path))
    monkeypatch.setenv("HALO_RAG_CONFIG_PATH", str(config_path))
    calls = []

    def make_pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "src.pipeline", SimpleNamespace(
        SelfVerificationRAGPipeline=make_pipeline
    ))
    load_default_pipeline()

    assert calls[0]["corpus"] == ["Paris is in France.", "London is in England."]
    assert calls[0]["device"] == "cpu"
    assert calls[0]["use_qlora"] is False
    assert calls[0]["max_revision_iterations"] == 2
