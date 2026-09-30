"""Checks for the optional single-passage abstention mode."""

import json
from types import SimpleNamespace

import pytest
import torch

from experiments.evaluate_focused_answers import summarize
from experiments.export_benchmark_review import (
    load_review_sources, review_rows, spreadsheet_safe,
)
from experiments.generate_human_eval_samples import (
    generate_human_eval_samples, save_human_eval_samples,
)
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


def test_no_answer_mode_does_not_expand_evidence_during_revision():
    pipeline, _ = make_pipeline("Paris")
    pipeline.verifier.verify_generation = lambda *args, **kwargs: {
        "verified": False,
        "verification_results": [],
        "entailment_rate": 0.0,
    }

    result = pipeline.generate("Where?", evidence_limit=1, abstain_if_unanswered=True)

    assert result["reranked_texts"] == ["first passage"]
    assert result["revision_iterations"] == 0
    assert result["verified"] is False
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


def test_human_eval_export_does_not_overwrite_existing_annotations(tmp_path):
    output = tmp_path / "review.csv"
    save_human_eval_samples([{"id": "one", "human_label": "SUPPORTED",
                              "generated_answer": "=HYPERLINK(\"bad\")"}], str(output))
    original = output.read_bytes()
    assert b"'=HYPERLINK" in original

    with pytest.raises(FileExistsError):
        save_human_eval_samples([{"id": "two", "human_label": ""}], str(output))

    assert output.read_bytes() == original


def test_human_eval_requires_requested_sample_count_before_loading_models():
    with pytest.raises(ValueError, match="requested 2 samples"):
        generate_human_eval_samples(["question"], ["answer"], [[0]], ["passage"], {}, 2)


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
                "reranked_texts": ["Paris"],
                "claims": ["Paris"],
                "verification_results": {"verification_results": [{"claim": "Paris"}]},
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


def test_review_export_rejects_mixed_benchmark_protocols(tmp_path):
    metadata = {
        "dataset": "squad_v2", "split": "validation", "corpus_size": 500,
        "top_k_retrieve": 20, "top_k_rerank": 5, "max_revisions": 3,
        "config_sha256": "config", "source_sha256": "first",
        "models": {}, "package_versions": {}, "seed": 42,
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"metadata": metadata, "cases": {"focused": []}}))
    second_metadata = {**metadata, "seed": 123, "source_sha256": "second"}
    second.write_text(json.dumps({
        "metadata": second_metadata, "cases": {"focused": []},
    }))

    with pytest.raises(ValueError, match="incompatible benchmark protocol.*source_sha256"):
        load_review_sources([first, second], "focused", "config")

    second_metadata["source_sha256"] = "first"
    second.write_text(json.dumps({
        "metadata": second_metadata, "cases": {"focused": []},
    }))
    assert len(load_review_sources([first, second], "focused", "config")) == 2

    with pytest.raises(ValueError, match="duplicate seed 42"):
        load_review_sources([first, first], "focused", "config")


def test_review_export_keeps_source_and_retrieved_evidence_separate():
    case = BenchmarkCase(
        example_id="unanswerable", question="Where?", context="The source passage.",
        references=(), relevant_doc_id=0, answerable=False,
    )
    benchmark = BenchmarkSet(
        corpus=("The source passage.", "A retrieved passage."),
        document_hashes=("a", "b"), cases=(case,), seed=3,
    )
    source = {
        "metadata": {"seed": 3},
        "cases": {"focused": [{
            "example_id": "unanswerable", "exact_match": 0.0,
            "reranked_doc_ids": [1], "generated": "Paris", "abstained": False,
            "verified": True, "final_evidence_hit": 0.0,
        }]},
    }

    rows = review_rows(benchmark, source, "focused")

    assert len(rows) == 1
    assert rows[0]["source_passage"] == "The source passage."
    assert rows[0]["evidence_passage"] == "A retrieved passage."
    assert rows[0]["answers_question"] == ""
    assert rows[0]["supported_by_evidence"] == ""


def test_review_export_can_select_verified_unanswerable_answers():
    cases = tuple(
        BenchmarkCase(
            example_id=example_id, question="Where?", context="Source passage.",
            references=("Paris",) if answerable else (), relevant_doc_id=0,
            answerable=answerable,
        )
        for example_id, answerable in (
            ("false_accept", False), ("abstained", False), ("answerable", True)
        )
    )
    benchmark = BenchmarkSet(
        corpus=("Source passage.", "Retrieved passage."),
        document_hashes=("a", "b"), cases=cases, seed=3,
    )
    rows = [
        {
            "example_id": case.example_id, "exact_match": 0.0,
            "reranked_doc_ids": [1], "generated": "Paris",
            "abstained": case.example_id == "abstained",
            "verified": case.example_id != "abstained", "final_evidence_hit": 0.0,
        }
        for case in cases
    ]
    source = {"metadata": {"seed": 3}, "cases": {"focused": rows}}

    selected = review_rows(benchmark, source, "focused", scope="false-accepts")

    assert [row["example_id"] for row in selected] == ["false_accept"]
    assert selected[0]["source_passage"] == "Source passage."
    assert selected[0]["evidence_passage"] == "Retrieved passage."


def test_review_export_escapes_spreadsheet_formulas():
    assert spreadsheet_safe("=2+2") == "'=2+2"
    assert spreadsheet_safe("  @command") == "'  @command"
    assert spreadsheet_safe("A normal answer") == "A normal answer"
