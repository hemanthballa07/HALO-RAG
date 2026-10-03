"""Validation for retrieval comparison inputs."""

import pytest

from experiments.exp2_retrieval_comparison import run_retrieval_comparison


@pytest.mark.parametrize(
    "queries,ground_truths,relevant_docs",
    [
        (["question"], [], [[0]]),
        (["question"], ["answer"], []),
    ],
)
def test_retrieval_comparison_rejects_misaligned_inputs(
    queries, ground_truths, relevant_docs
):
    with pytest.raises(ValueError, match="must have equal lengths"):
        run_retrieval_comparison(
            queries=queries,
            ground_truths=ground_truths,
            relevant_docs=relevant_docs,
            corpus=["answer"],
            config={},
        )


def test_retrieval_comparison_rejects_empty_queries():
    with pytest.raises(ValueError, match="at least one query is required"):
        run_retrieval_comparison(
            queries=[],
            ground_truths=[],
            relevant_docs=[],
            corpus=[],
            config={},
        )
