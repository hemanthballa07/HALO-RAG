"""The verifier challenge runner is testable without loading model weights."""

from pathlib import Path
from types import ModuleType

import pytest

from experiments.evaluate_verifier_challenges import evaluate_cases, load_challenges
from tests.test_regressions import load_module


torch_stub = ModuleType("torch")
torch_stub.Tensor = type("Tensor", (), {})
transformers_stub = ModuleType("transformers")
transformers_stub.AutoTokenizer = object
transformers_stub.AutoModelForSequenceClassification = object
EntailmentVerifier = load_module(
    "halo_challenge_verifier",
    "src/verification/entailment_verifier.py",
    {"torch": torch_stub, "transformers": transformers_stub},
).EntailmentVerifier


FIXTURE = Path(__file__).resolve().parents[1] / "experiments/fixtures/verifier_challenges.json"


def test_challenge_fixture_has_paired_supported_and_unsupported_cases():
    cases = load_challenges(FIXTURE)

    assert len(cases) == 10
    assert sum(case["supported"] for case in cases) == 5
    assert len({case["id"] for case in cases}) == len(cases)


@pytest.mark.parametrize("case", load_challenges(FIXTURE), ids=lambda case: case["id"])
def test_shortcut_respects_question_constraints(case):
    assert EntailmentVerifier._answer_sentence_matches_query(
        case["answer"], case["context"], case["question"]
    ) is case["supported"]


def test_action_guard_does_not_treat_a_following_noun_as_the_action():
    assert EntailmentVerifier._answer_sentence_matches_query(
        "West High School",
        "She attended West High School while pursuing an undergraduate degree.",
        "What school did she attend for college?",
    )


def test_shorter_object_question_keeps_relation_direction():
    assert not EntailmentVerifier._answer_sentence_matches_query(
        "The IPCC",
        "The IPCC supports UNFCCC.",
        "Which organization does UNFCCC support?",
    )


def test_object_question_accepts_matching_relation():
    assert EntailmentVerifier._answer_sentence_matches_query(
        "UNFCCC",
        "The IPCC supports UNFCCC.",
        "Which organization does the IPCC support?",
    )


def test_formula_anchor_allows_spacing_before_subscript():
    assert EntailmentVerifier._answer_sentence_matches_query(
        "triplet oxygen",
        "The O 2 molecule ground state is called triplet oxygen.",
        "What is the O2 molecule ground state called?",
    )


def test_shortcut_requires_named_location_in_evidence():
    context = "Schools in Malaysia became National Type after independence."

    assert not EntailmentVerifier._answer_sentence_matches_query(
        "National Type",
        context,
        "What type did schools in China become after independence?",
    )
    assert EntailmentVerifier._answer_sentence_matches_query(
        "National Type",
        context,
        "What type did schools in Malaysia become after independence?",
    )


def test_shortcut_requires_the_queried_condition():
    context = "Decompression sickness occurs in divers."

    assert not EntailmentVerifier._answer_sentence_matches_query(
        "divers", context, "Who does decompression oxygen sickness occur in?"
    )
    assert EntailmentVerifier._answer_sentence_matches_query(
        "divers", context, "Who does decompression sickness occur in?"
    )


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
