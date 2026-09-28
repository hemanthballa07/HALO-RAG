"""The verifier challenge runner is testable without loading model weights."""

from pathlib import Path

import pytest

from experiments.evaluate_verifier_challenges import evaluate_cases, load_challenges


FIXTURE = Path(__file__).resolve().parents[1] / "experiments/fixtures/verifier_challenges.json"


def test_challenge_fixture_has_paired_supported_and_unsupported_cases():
    cases = load_challenges(FIXTURE)

    assert len(cases) == 10
    assert sum(case["supported"] for case in cases) == 5
    assert len({case["id"] for case in cases}) == len(cases)


def test_evaluator_counts_false_accepts_by_scoring_method():
    class FakeVerifier:
        threshold = 0.75

        def verify_claim(self, claim, context, query=None):
            return {
                "entailment": 1.0 if claim == "accepted" else 0.1,
                "method": "question_sentence_match" if claim == "accepted" else "nli",
            }

    cases = [
        {"id": "supported", "question": "Q1?", "answer": "accepted",
         "context": "Evidence one.", "supported": True},
        {"id": "unsupported", "question": "Q2?", "answer": "accepted",
         "context": "Evidence two.", "supported": False},
        {"id": "rejected", "question": "Q3?", "answer": "rejected",
         "context": "Evidence three.", "supported": False},
    ]

    result = evaluate_cases(cases, FakeVerifier())

    assert result["summary"] == {
        "count": 3,
        "true_accepts": 1,
        "false_rejects": 0,
        "false_accepts": 1,
        "true_rejects": 1,
        "false_accept_methods": {"question_sentence_match": 1},
    }
    assert result["cases"][1]["accepted"] is True
    assert result["cases"][1]["verification_method"] == "question_sentence_match"


def test_challenge_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "challenges.json"
    path.write_text('''[
      {"id":"same","question":"Q?","answer":"A","context":"C","supported":true},
      {"id":"same","question":"Q?","answer":"A","context":"C","supported":false}
    ]''', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate challenge id"):
        load_challenges(path)
