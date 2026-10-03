"""Verified training data must not hide failed queries."""

import pytest

from src.data.verified_collector import collect_verified_data


def test_collector_fails_instead_of_returning_partial_data():
    class FailingPipeline:
        def generate(self, query, **_kwargs):
            if query == "second":
                raise ValueError("model failed")
            return {
                "generated_text": "answer",
                "verification_results": {
                    "verification_results": [{"label": "ENTAILED"}]
                },
                "reranked_texts": ["passage"],
            }

    with pytest.raises(RuntimeError, match="query 1") as error:
        collect_verified_data(
            pipeline=FailingPipeline(),
            queries=["first", "second"],
            ground_truths=["answer", "answer"],
            relevant_docs=[[0], [0]],
            corpus=["passage"],
        )

    assert isinstance(error.value.__cause__, ValueError)


def test_collector_still_filters_low_precision_examples():
    class Pipeline:
        def generate(self, query, **_kwargs):
            label = "ENTAILED" if query == "supported" else "NO_EVIDENCE"
            return {
                "generated_text": "answer",
                "verification_results": {
                    "verification_results": [{"label": label}]
                },
                "reranked_texts": ["passage"],
            }

    result = collect_verified_data(
        pipeline=Pipeline(),
        queries=["supported", "unsupported"],
        ground_truths=["answer", "answer"],
        relevant_docs=[[0], [0]],
        corpus=["passage"],
    )

    assert [item["question"] for item in result] == ["supported"]


def test_collector_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must have equal lengths"):
        collect_verified_data(
            pipeline=None,
            queries=["question"],
            ground_truths=[],
            relevant_docs=[[0]],
            corpus=["passage"],
        )
