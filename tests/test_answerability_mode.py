"""Checks for the optional single-passage abstention mode."""

from types import SimpleNamespace

import pytest
import torch

from experiments.evaluate_focused_answers import summarize
from experiments.run_representative_benchmark import evaluate_cases
from src.evaluation.benchmark import BenchmarkCase, BenchmarkSet
from src.generator.flan_t5_generator import FLANT5Generator
from src.pipeline.rag_pipeline import SelfVerificationRAGPipeline


def make_pipeline(answer):
    pipeline = SelfVerificationRAGPipeline.__new__(SelfVerificationRAGPipeline)
    pipeline.max_revision_iterations = 1
    pipeline.enable_revision = True
    pipeline.retriever = SimpleNamespace(
        retrieve=lambda query, top_k: [(101, "first passage"), (201, "second passage")]
    )
    pipeline.reranker = SimpleNamespace(
        rerank=lambda query, documents, top_k: [
            (0, documents[0], 0.9),
            (1, documents[1], 0.8),
        ]
    )
    seen = {}

    def generate(query, context, **kwargs):
        seen["context"] = context
        seen["kwargs"] = kwargs
        return answer

    pipeline.generator = SimpleNamespace(generate=generate)
    pipeline.claim_extractor = SimpleNamespace(extract_claims=lambda text: [text])
    pipeline.verifier = SimpleNamespace(
        verify_generation=lambda text, contexts, claims, query: {
            "verified": True,
            "verification_results": [],
            "entailment_rate": 1.0,
        }
    )
    pipeline.revision_strategy = SimpleNamespace(
        revise=lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected revision"))
    )
    return pipeline, seen


def test_evidence_limit_is_applied_before_generation_and_verification():
    pipeline, seen = make_pipeline("Paris")

    result = pipeline.generate("Where?", evidence_limit=1, top_k_rerank=2)

    assert seen["context"] == "first passage"
    assert result["initial_reranked_docs"] == [101]
    assert result["reranked_docs"] == [101]


def test_unanswerable_marker_abstains_without_verification_or_revision():
    pipeline, seen = make_pipeline("UNANSWERABLE")
    pipeline.verifier.verify_generation = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected verification")
    )

    result = pipeline.generate("Where?", evidence_limit=1, abstain_if_unanswered=True)

    assert seen["kwargs"]["abstain_if_unanswered"] is True
    assert result["abstained"] is True
    assert result["verified"] is False
    assert result["revision_iterations"] == 0


def test_answerable_response_still_uses_verification():
    pipeline, seen = make_pipeline("Paris")

    result = pipeline.generate("Where?", evidence_limit=1, abstain_if_unanswered=True)

    assert seen["kwargs"]["abstain_if_unanswered"] is True
    assert result["verified"] is True
    assert result["abstained"] is False


def test_evidence_limit_must_be_positive():
    pipeline, _ = make_pipeline("Paris")

    with pytest.raises(ValueError, match="evidence_limit"):
        pipeline.generate("Where?", evidence_limit=0)


def test_abstention_prompt_keeps_the_question_after_one_passage():
    prompt = FLANT5Generator.build_prompt(
        "Where are the campuses?",
        "The school was founded in Paris.",
        abstain_if_unanswered=True,
    )

    assert prompt.index("Passage:") < prompt.index("Question:")
    assert "UNANSWERABLE" in prompt


def test_abstention_marker_accepts_case_and_terminal_punctuation():
    assert FLANT5Generator.is_unanswerable_response("UNANSWERABLE")
    assert FLANT5Generator.is_unanswerable_response("unanswerable.")
    assert not FLANT5Generator.is_unanswerable_response("The answer is unanswerable")


def test_long_passage_keeps_the_question_inside_the_input_limit():
    class Tokenizer:
        pad_token_id = 0

        def __init__(self):
            self.last_prompt = ""

        def __call__(self, text, return_tensors=None, **kwargs):
            if return_tensors:
                self.last_prompt = text
                return {"input_ids": torch.tensor([[1]])}
            return {"input_ids": [1] * len(text.split())}

        def decode(self, tokens, **kwargs):
            if isinstance(tokens, torch.Tensor):
                return "answer"
            return " ".join("word" for _ in tokens)

    generator = FLANT5Generator.__new__(FLANT5Generator)
    generator.device = "cpu"
    generator.tokenizer = Tokenizer()
    generator.model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[2]]))

    generator.generate(
        "Where are the campuses?",
        "word " * 1000,
        do_sample=False,
        abstain_if_unanswered=True,
    )

    prompt = generator.tokenizer.last_prompt
    assert len(prompt.split()) <= 512
    assert prompt.endswith("Question: Where are the campuses?\nAnswer:")


def test_focused_summary_keeps_answerability_groups_separate():
    rows = [
        {"answerable": True, "exact_match": 1.0, "f1": 1.0,
         "abstained": False, "evidence_hit": 1.0},
        {"answerable": False, "exact_match": 1.0, "f1": 1.0,
         "abstained": True, "evidence_hit": 0.0},
    ]

    result = summarize(rows)

    assert result["overall"]["exact_match"] == 1.0
    assert result["answerable"]["abstained"] == 0.0
    assert result["unanswerable"]["abstained"] == 1.0


def test_paired_benchmark_can_run_focused_mode_on_the_same_case():
    case = BenchmarkCase(
        example_id="sample", question="Where?", context="Paris", references=("Paris",),
        relevant_doc_id=0, answerable=True,
    )
    benchmark = BenchmarkSet(
        corpus=("Paris", "London"), document_hashes=("a", "b"), cases=(case,), seed=42,
    )
    calls = []

    class Pipeline:
        enable_revision = False

        def generate(self, query, **kwargs):
            calls.append((self.enable_revision, kwargs))
            return {
                "generated_text": "Paris",
                "initial_retrieved_docs": [0, 1],
                "initial_reranked_docs": [0, 1],
                "retrieved_docs": [0, 1],
                "reranked_docs": [0],
                "verified": True,
                "abstained": False,
                "revision_iterations": 0,
            }

    rows = evaluate_cases(Pipeline(), benchmark, 42, 20, 5, include_focused=True)

    assert list(rows) == ["baseline", "revision", "focused"]
    assert [enabled for enabled, _ in calls] == [False, True, False]
    assert calls[2][1]["evidence_limit"] == 1
    assert calls[2][1]["abstain_if_unanswered"] is True
    assert all(row["exact_match"] == 1.0 for variant in rows.values() for row in variant)
