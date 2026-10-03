"""Self-consistency retrieval scores must use the selected sample's documents."""

from types import SimpleNamespace

from experiments import exp5_self_consistency as exp5


def test_selected_answer_uses_its_own_retrieval(monkeypatch):
    class FakePipeline:
        def __init__(self, **_kwargs):
            self.calls = 0
            self.claim_extractor = SimpleNamespace(extract_claims=lambda text: [text])
            self.verifier = SimpleNamespace(
                verify_generation=lambda *_args: {
                    "verification_results": [{"is_entailed": True}]
                }
            )

        def generate(self, _query, **_kwargs):
            doc_id = self.calls
            self.calls += 1
            answer = "Paris" if doc_id == 3 else "London"
            return {
                "retrieved_docs": [doc_id],
                "reranked_texts": [f"passage {doc_id}"],
                "generated_text": answer,
                "verification_results": {
                    "verification_results": [{"label": "ENTAILED", "is_entailed": True}]
                },
            }

    monkeypatch.setattr(exp5, "SelfVerificationRAGPipeline", FakePipeline)
    monkeypatch.setattr(exp5, "resolve_device", lambda _device: "cpu")

    results = exp5.run_self_consistency_experiment(
        queries=["Which city?"],
        ground_truths=["Paris"],
        relevant_docs=[[3]],
        corpus=["passage 0", "passage 1", "passage 2", "passage 3"],
        config={"experiments": {"device": "cpu"}},
        k=2,
    )["individual_results"]

    assert results["greedy"][0]["metrics"]["recall@5"] == 0.0
    assert results["self_consistency"][0]["generated"] == "Paris"
    assert results["self_consistency"][0]["metrics"]["recall@5"] == 1.0
