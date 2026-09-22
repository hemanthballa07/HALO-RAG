"""Keep retrieval audit fields stable when revision changes the evidence."""

from types import SimpleNamespace

from src.pipeline.rag_pipeline import SelfVerificationRAGPipeline


def test_initial_document_ids_survive_retrieval_revision():
    pipeline = SelfVerificationRAGPipeline.__new__(SelfVerificationRAGPipeline)
    pipeline.max_revision_iterations = 1
    pipeline.enable_revision = True
    pipeline.retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [(101, "initial passage"), (201, "distractor")]
    )
    pipeline.reranker = SimpleNamespace(
        rerank=lambda query, documents, top_k: [(0, documents[0], 0.9)]
    )
    pipeline.generator = SimpleNamespace(generate=lambda query, context, **kwargs: "initial answer")
    pipeline.claim_extractor = SimpleNamespace(extract_claims=lambda answer: [answer])
    pipeline.verifier = SimpleNamespace(
        verify_generation=lambda answer, contexts, claims, query: {
            "verified": False,
            "verification_results": [],
            "entailment_rate": 0.0,
        }
    )
    pipeline.revision_strategy = SimpleNamespace(
        revise=lambda **kwargs: (
            "revised answer",
            {"verified": True, "verification_results": [], "entailment_rate": 1.0},
            {
                "strategy_name": "re_retrieval",
                "evidence_contexts": ["revised passage"],
                "document_ids": [999],
            },
        )
    )

    result = pipeline.generate("question", top_k_retrieve=2, top_k_rerank=1)

    assert result["initial_retrieved_docs"] == [101, 201]
    assert result["initial_reranked_docs"] == [101]
    assert result["retrieved_docs"] == [999]
    assert result["reranked_docs"] == [999]
