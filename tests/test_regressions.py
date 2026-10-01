"""Regression tests for core HALO-RAG behavior.

These tests avoid downloading models or datasets. External ML dependencies are
replaced with small test doubles so the suite can run in a lightweight CI job.
"""

from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str, stubs: dict[str, types.ModuleType] | None = None):
    """Load a source module directly while temporarily installing dependency stubs."""
    original_modules: dict[str, types.ModuleType | None] = {}
    for module_name, module in (stubs or {}).items():
        original_modules[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module

    try:
        spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {relative_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, original in original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original


class ClaimExtractorTests(unittest.TestCase):
    @staticmethod
    def extractor(sentences, structured=None):
        spacy = types.ModuleType("spacy")
        module = load_module(
            "halo_test_claim_extractor",
            "src/verification/claim_extractor.py",
            {"spacy": spacy},
        )
        extractor = module.ClaimExtractor.__new__(module.ClaimExtractor)
        extractor.nlp = lambda text: types.SimpleNamespace(
            sents=[types.SimpleNamespace(text=sentence) for sentence in sentences]
        )
        extractor._extract_svo_from_sentence = lambda sent: [
            {"claim": claim} for claim in (structured or {}).get(sent.text, [])
        ]
        return extractor

    def test_preserves_fallback_for_each_sentence(self):
        extractor = self.extractor(
            ["Paris is in France.", "Ada wrote a book."],
            {"Ada wrote a book.": ["Ada wrote book"]},
        )

        self.assertEqual(
            extractor.extract_claims("Paris is in France. Ada wrote a book."),
            ["Paris is in France.", "Ada wrote book"],
        )

    def test_ignores_questions_and_repeated_claims(self):
        extractor = self.extractor(
            ["Paris is in France.", "Where is Paris?", "Paris is in France."]
        )

        self.assertEqual(
            extractor.extract_claims("Paris is in France. Where is Paris? Paris is in France."),
            ["Paris is in France."],
        )

    def test_keeps_short_answer_without_svo(self):
        extractor = self.extractor(["late 1990s"])

        self.assertEqual(extractor.extract_claims("late 1990s"), ["late 1990s"])

    def test_svo_claim_keeps_auxiliary_and_negation(self):
        extractor = self.extractor([])
        extractor._extract_svo_from_sentence = types.MethodType(
            type(extractor)._extract_svo_from_sentence, extractor
        )

        def token(text, index, dependency, children=(), part_of_speech=""):
            return types.SimpleNamespace(
                text=text, i=index, dep_=dependency, pos_=part_of_speech,
                children=children,
            )

        subject = token("Alice", 0, "nsubj")
        auxiliary = token("did", 1, "aux")
        negation = token("not", 2, "neg")
        determiner = token("the", 4, "det")
        obj = token("report", 5, "dobj", (determiner,))
        root = token("write", 3, "ROOT", (subject, auxiliary, negation, obj), "VERB")

        self.assertEqual(extractor._extract_svo_from_sentence([root]), [{
            "subject": "Alice",
            "verb": "did not write",
            "object": "the report",
            "claim": "Alice did not write the report",
        }])


class DataLoaderTests(unittest.TestCase):
    @staticmethod
    def loaders_module():
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: []
        datasets.Dataset = object
        return load_module("halo_test_loaders", "src/data/loaders.py", {"datasets": datasets})

    def test_normalize_text_converts_curly_quotes(self):
        loaders = self.loaders_module()
        self.assertEqual(loaders.normalize_text("  “quoted” and ‘single’  "), '"quoted" and \'single\'')

    def test_hotpot_context_dictionary_is_supported(self):
        loaders = self.loaders_module()
        loaders.hf_load_dataset = lambda *args, **kwargs: [
            {
                "id": "hp-1",
                "question": "Which city?",
                "answer": "Gainesville",
                "context": {
                    "title": ["Florida", "University of Florida"],
                    "sentences": [["Florida is a state."], ["UF is in Gainesville."]],
                },
            }
        ]

        examples = loaders.load_hotpotqa(limit=1)

        self.assertEqual(len(examples), 1)
        self.assertIn("Florida: Florida is a state.", examples[0]["context"])
        self.assertIn("University of Florida: UF is in Gainesville.", examples[0]["context"])

    def test_natural_questions_end_token_is_exclusive(self):
        loaders = self.loaders_module()
        loaders.hf_load_dataset = lambda *args, **kwargs: [
            {
                "id": "nq-1",
                "question": {"text": "What is the answer?", "tokens": ["What", "is", "the", "answer"]},
                "document": {
                    "title": "Example",
                    "tokens": [
                        {"token": "Alpha", "is_html": False},
                        {"token": "Beta", "is_html": False},
                        {"token": "Gamma", "is_html": False},
                    ],
                },
                "annotations": [
                    {"short_answers": [{"start_token": 1, "end_token": 2, "text": "Beta"}]}
                ],
            }
        ]

        examples = loaders.load_natural_questions(limit=1)

        self.assertEqual(examples[0]["answers"], ["Beta"])


class RetrievalTests(unittest.TestCase):
    @staticmethod
    def retrieval_module():
        faiss = types.ModuleType("faiss")
        faiss.normalize_L2 = lambda value: None
        faiss.IndexFlatIP = object
        rank_bm25 = types.ModuleType("rank_bm25")
        rank_bm25.BM25Okapi = object
        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.SentenceTransformer = object
        torch = types.ModuleType("torch")
        return load_module(
            "halo_test_retrieval",
            "src/retrieval/hybrid_retrieval.py",
            {
                "faiss": faiss,
                "rank_bm25": rank_bm25,
                "sentence_transformers": sentence_transformers,
                "torch": torch,
            },
        )

    def test_embedding_dimension_uses_current_method_with_legacy_fallback(self):
        retrieval = self.retrieval_module()
        retrieval.resolve_device = lambda _preferred: "mps"

        class CurrentModel:
            def get_embedding_dimension(self):
                return 768

            def get_sentence_embedding_dimension(self):
                raise AssertionError("deprecated method should not be called")

        retrieval.SentenceTransformer = lambda *_args, **_kwargs: CurrentModel()
        current = retrieval.HybridRetriever()
        self.assertEqual(current.embedding_dim, 768)

        class LegacyModel:
            def get_sentence_embedding_dimension(self):
                return 384

        retrieval.SentenceTransformer = lambda *_args, **_kwargs: LegacyModel()
        legacy = retrieval.HybridRetriever()
        self.assertEqual(legacy.embedding_dim, 384)

    def test_dense_retrieval_clamps_top_k_to_corpus_size(self):
        retrieval = self.retrieval_module()

        class DenseModel:
            def encode(self, *args, **kwargs):
                return np.array([1.0, 0.0], dtype=np.float32)

        class Index:
            def search(self, query, top_k):
                scores = np.array([[0.9, 0.8] + [-3.4028235e38] * (top_k - 2)], dtype=np.float32)
                indices = np.array([[0, 1] + [-1] * (top_k - 2)], dtype=np.int64)
                return scores, indices

        retriever = retrieval.HybridRetriever.__new__(retrieval.HybridRetriever)
        retriever.dense_model = DenseModel()
        retriever.faiss_index = Index()
        retriever.corpus = ["doc0", "doc1"]

        self.assertEqual(retriever.retrieve_dense_only("query", top_k=5), [(0, "doc0"), (1, "doc1")])

    def test_non_candidates_receive_no_dense_score(self):
        retrieval = self.retrieval_module()

        class DenseModel:
            def encode(self, *args, **kwargs):
                return np.array([1.0, 0.0], dtype=np.float32)

        class Index:
            def search(self, query, top_k):
                return (
                    np.array([[-0.1, -0.2]], dtype=np.float32),
                    np.array([[0, 1]], dtype=np.int64),
                )

        class BM25:
            def get_scores(self, query):
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        retriever = retrieval.HybridRetriever.__new__(retrieval.HybridRetriever)
        retriever.dense_model = DenseModel()
        retriever.faiss_index = Index()
        retriever.bm25 = BM25()
        retriever.corpus = ["dense match", "weak match", "sparse match"]
        retriever.dense_weight = 0.6
        retriever.sparse_weight = 0.4

        results = retriever.retrieve("query", top_k=2, return_scores=True)

        self.assertEqual([result[0] for result in results], [0, 2])
        self.assertAlmostEqual(results[1][2], 0.4, places=6)


class GeneratorTrainingTests(unittest.TestCase):
    def test_padding_tokens_are_masked_from_training_loss(self):
        torch = types.ModuleType("torch")
        transformers = types.ModuleType("transformers")
        transformers.TrainingArguments = object
        transformers.Trainer = object
        transformers.DataCollatorForSeq2Seq = object
        peft = types.ModuleType("peft")
        peft.LoraConfig = object
        peft.get_peft_model = lambda model, config: model
        peft.TaskType = types.SimpleNamespace(SEQ_2_SEQ_LM="seq2seq")
        datasets = types.ModuleType("datasets")

        class Dataset:
            @staticmethod
            def from_dict(payload):
                return payload

        datasets.Dataset = Dataset
        trainer_module = load_module(
            "halo_test_qlora_trainer",
            "src/generator/qlora_trainer.py",
            {
                "torch": torch,
                "transformers": transformers,
                "peft": peft,
                "datasets": datasets,
            },
        )

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, values=None, **kwargs):
                if "text_target" in kwargs:
                    return {"input_ids": np.array([[7, 0], [8, 9]])}
                return {
                    "input_ids": np.array([[1, 2], [3, 0]]),
                    "attention_mask": np.array([[1, 1], [1, 0]]),
                }

        trainer = trainer_module.QLoRATrainer.__new__(trainer_module.QLoRATrainer)
        trainer.tokenizer = Tokenizer()
        dataset = trainer.prepare_dataset(["q1", "q2"], ["c1", "c2"], ["a1", "a2"])

        self.assertEqual(dataset["labels"].tolist(), [[7, -100], [8, 9]])


class EvaluationMetricTests(unittest.TestCase):
    @staticmethod
    def metrics():
        module = load_module("halo_test_metrics", "src/evaluation/metrics.py")
        return module.EvaluationMetrics()

    def test_exact_match_normalizes_articles_and_punctuation(self):
        metrics = self.metrics()
        self.assertEqual(metrics.exact_match("The Eiffel Tower.", "eiffel tower"), 1.0)

    def test_f1_counts_repeated_tokens(self):
        metrics = self.metrics()
        self.assertAlmostEqual(metrics.f1_score("red red blue", "red blue blue"), 2 / 3)

    def test_coverage_normalizes_punctuation(self):
        metrics = self.metrics()
        self.assertEqual(metrics.coverage("Paris, France", ["Paris is in France."]), 1.0)


class EntailmentVerifierTests(unittest.TestCase):
    @staticmethod
    def verifier_module():
        torch = types.ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})
        torch.no_grad = lambda: None
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace()
        transformers.AutoModelForSequenceClassification = types.SimpleNamespace()
        return load_module(
            "halo_test_verifier",
            "src/verification/entailment_verifier.py",
            {"torch": torch, "transformers": transformers},
        )

    def test_verifier_uses_model_label_mapping(self):
        verifier_module = self.verifier_module()

        class Model:
            config = types.SimpleNamespace(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            )

            def to(self, device):
                return self

            def eval(self):
                return self

        verifier_module.AutoModelForSequenceClassification.from_pretrained = lambda name: Model()
        verifier_module.AutoTokenizer.from_pretrained = lambda *args, **kwargs: object()

        verifier = verifier_module.EntailmentVerifier(device="cpu")

        self.assertEqual(verifier.label_map[1], "entailment")
        self.assertEqual(verifier.entailment_index, 1)
        self.assertEqual(verifier.neutral_index, 2)

    def test_matching_answer_in_unrelated_sentence_is_not_full_support(self):
        verifier_module = self.verifier_module()

        class Tensor:
            def to(self, device):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([[0.01, 0.01, 0.98]])

        class Tokenizer:
            hypothesis = None

            def __call__(self, context, hypothesis, **kwargs):
                self.hypothesis = hypothesis
                return {"input_ids": Tensor()}

        verifier_module.torch.no_grad = contextlib.nullcontext
        verifier_module.torch.softmax = lambda logits, dim: logits
        verifier = verifier_module.EntailmentVerifier.__new__(
            verifier_module.EntailmentVerifier
        )
        verifier.device = "cpu"
        verifier.max_length = 512
        verifier.contradiction_index = 0
        verifier.entailment_index = 1
        verifier.neutral_index = 2
        verifier.tokenizer = Tokenizer()
        verifier.model = lambda **kwargs: types.SimpleNamespace(logits=Tensor())

        result = verifier.verify_claim(
            "Paris", "The school was founded in Paris.",
            "Where are the school's campuses?",
        )

        self.assertLess(result["entailment"], 0.75)
        self.assertEqual(result["method"], "nli")
        self.assertIn("campuses", verifier.tokenizer.hypothesis)

    def test_question_match_method_is_visible_in_generation_audit(self):
        verifier_module = self.verifier_module()
        verifier = verifier_module.EntailmentVerifier.__new__(
            verifier_module.EntailmentVerifier
        )
        verifier.threshold = 0.75

        result = verifier.verify_generation(
            "Fort Beauséjour",
            ["In 1755, the British captured Fort Beauséjour."],
            ["Fort Beauséjour"],
            "In 1755, what fort did the British capture?",
        )

        claim_result = result["verification_results"][0]
        self.assertEqual(claim_result["verification_method"], "question_sentence_match")
        self.assertTrue(claim_result["is_entailed"])

    def test_dated_question_does_not_match_a_different_action(self):
        verifier_module = self.verifier_module()
        matches = verifier_module.EntailmentVerifier._answer_sentence_matches_query
        context = "In 1755, the British captured Fort Beauséjour."

        self.assertFalse(matches(
            "Fort Beauséjour", context,
            "In 1755, what fort did the British surrender?",
        ))
        self.assertTrue(matches(
            "Fort Beauséjour", context,
            "In 1755, what fort did the British capture?",
        ))

    def test_answer_sentence_keeps_common_abbreviations_together(self):
        verifier_module = self.verifier_module()
        matches = verifier_module.EntailmentVerifier._answer_sentence_matches_query

        self.assertTrue(matches(
            "Paris", "The school was founded in Paris.",
            "Where was the school founded?",
        ))
        self.assertTrue(matches(
            "St. Bartholomew's Day massacre",
            "The height of this Huguenot persecution was the St. Bartholomew's Day massacre.",
            "What event was the worst example of Huguenot persecution?",
        ))
        self.assertTrue(matches(
            "23 June 2005",
            "On 23 June 2005, Rep. Joe Barton and Ed Whitfield demanded climate research records.",
            "When did Barton and Whitfield demand climate research records?",
        ))

    def test_generation_verifies_each_claim_once(self):
        verifier_module = self.verifier_module()
        verifier = verifier_module.EntailmentVerifier.__new__(verifier_module.EntailmentVerifier)
        verifier.threshold = 0.75
        calls = []

        def verify_claim(claim, context, query=None):
            calls.append((claim, context, query))
            return {
                "contradiction": 0.1,
                "neutral": 0.1,
                "entailment": 0.8,
                "method": "nli",
            }

        verifier.verify_claim = verify_claim

        result = verifier.verify_generation("A claim", ["context"], ["A claim"])

        self.assertEqual(len(calls), 1)
        self.assertTrue(result["verified"])
        self.assertEqual(result["verification_results"][0]["verification_method"], "nli")

    def test_generation_without_claims_is_not_verified(self):
        verifier_module = self.verifier_module()
        verifier = verifier_module.EntailmentVerifier.__new__(verifier_module.EntailmentVerifier)
        verifier.verify_claim = lambda *args, **kwargs: self.fail("No claim should be verified")

        result = verifier.verify_generation("What is the population?", ["context"], [])

        self.assertEqual(result["num_total"], 0)
        self.assertFalse(result["verified"])

    def test_invalid_entailment_threshold_is_rejected(self):
        verifier_module = self.verifier_module()
        with self.assertRaises(ValueError):
            verifier_module.EntailmentVerifier(device="cpu", threshold=1.1)


class PipelineRevisionTests(unittest.TestCase):
    @staticmethod
    def pipeline():
        retrieval = types.ModuleType("src.retrieval")
        retrieval.HybridRetriever = object
        retrieval.CrossEncoderReranker = object
        generator = types.ModuleType("src.generator")
        generator.FLANT5Generator = object
        verification = types.ModuleType("src.verification")
        verification.EntailmentVerifier = object
        verification.ClaimExtractor = object
        revision = types.ModuleType("src.revision")
        revision.AdaptiveRevisionStrategy = object
        evaluation = types.ModuleType("src.evaluation")
        evaluation.EvaluationMetrics = object
        device = types.ModuleType("src.utils.device")
        device.resolve_device = lambda value: value
        pipeline_module = load_module(
            "halo_test_pipeline_revision",
            "src/pipeline/rag_pipeline.py",
            {
                "src.retrieval": retrieval,
                "src.generator": generator,
                "src.verification": verification,
                "src.revision": revision,
                "src.evaluation": evaluation,
                "src.utils.device": device,
            },
        )

        pipeline = pipeline_module.SelfVerificationRAGPipeline.__new__(
            pipeline_module.SelfVerificationRAGPipeline
        )
        pipeline.max_revision_iterations = 3
        pipeline.enable_revision = True
        pipeline.retriever = types.SimpleNamespace(retrieve=lambda query, top_k: [(1, "evidence")])
        pipeline.reranker = types.SimpleNamespace(
            rerank=lambda query, documents, top_k: [(0, documents[0], 0.9)]
        )
        pipeline.generator = types.SimpleNamespace(
            generate=lambda query, context, **kwargs: "Unverified answer"
        )
        pipeline.claim_extractor = types.SimpleNamespace(
            extract_claims=lambda text: [text]
        )
        pipeline.verifier = types.SimpleNamespace(
            verify_generation=lambda *args, **kwargs: {"verified": False}
        )
        return pipeline

    def test_zero_revision_limit_preserves_unverified_answer(self):
        pipeline = self.pipeline()
        pipeline.revision_strategy = types.SimpleNamespace(
            revise=lambda **kwargs: self.fail("Revision should not run")
        )

        result = pipeline.generate("question", max_revision_iterations=0)

        self.assertEqual(result["generated_text"], "Unverified answer")
        self.assertEqual(result["revision_iterations"], 0)
        self.assertFalse(result["verified"])
        self.assertFalse(result["abstained"])

    def test_request_revision_limit_reaches_strategy(self):
        pipeline = self.pipeline()
        pipeline.max_revision_iterations = 1
        calls = []

        def revise(**kwargs):
            calls.append(kwargs)
            return "Revised answer", {"verified": True}, {"strategy_name": "none"}

        pipeline.revision_strategy = types.SimpleNamespace(revise=revise)

        result = pipeline.generate("question", max_revision_iterations=2)

        self.assertEqual(calls[0]["max_iterations"], 2)
        self.assertEqual(result["revision_iterations"], 1)
        self.assertTrue(result["verified"])


class AdaptiveRevisionLimitTests(unittest.TestCase):
    def test_request_limit_overrides_configured_limit(self):
        module = load_module("halo_test_revision_limits", "src/revision/adaptive_strategies.py")
        strategy = module.AdaptiveRevisionStrategy(
            max_iterations=1, strategy_selection_mode="fixed", fixed_strategy="re_retrieval"
        )
        strategy._re_retrieval_strategy = lambda *args: ("revised", {"verified": True}, {})

        result = strategy.revise(
            query="question",
            initial_generation="original",
            verification_results={"verified": False},
            retrieval_fn=lambda *args: [],
            generation_fn=lambda *args: "revised",
            verification_fn=lambda *args: {"verified": True},
            claim_extractor_fn=lambda text: [text],
            iteration=1,
            max_iterations=2,
        )

        self.assertEqual(result[0], "revised")
        self.assertEqual(strategy.max_iterations, 1)


class ExperimentRunnerTests(unittest.TestCase):
    def test_json_result_metrics_are_flattened(self):
        runner = load_module(
            "halo_test_runner",
            "experiments/run_final_experiments.py",
        )

        metrics = runner.extract_numeric_metrics(
            {
                "aggregated_metrics": {
                    "f1_score": {"mean": 0.75, "std": 0.1},
                    "coverage": {"mean": 0.9},
                }
            }
        )

        self.assertEqual(metrics, {"f1_score": 0.75, "coverage": 0.9})


class RepositoryHealthTests(unittest.TestCase):
    def test_all_tracked_python_files_parse(self):
        files = subprocess.check_output(
            ["git", "ls-files", "*.py"], cwd=PROJECT_ROOT, text=True
        ).splitlines()
        failures = []
        for relative_path in files:
            path = PROJECT_ROOT / relative_path
            if not path.exists():
                continue
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except SyntaxError as exc:
                failures.append(f"{relative_path}:{exc.lineno}: {exc.msg}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
