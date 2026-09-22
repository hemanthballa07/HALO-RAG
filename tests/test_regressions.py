"""Regression tests for core HALO-RAG behavior.

These tests avoid downloading models or datasets. External ML dependencies are
replaced with small test doubles so the suite can run in a lightweight CI job.
"""

from __future__ import annotations

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

    def test_generation_verifies_each_claim_once(self):
        verifier_module = self.verifier_module()
        verifier = verifier_module.EntailmentVerifier.__new__(verifier_module.EntailmentVerifier)
        verifier.threshold = 0.75
        calls = []

        def verify_claim(claim, context, query=None):
            calls.append((claim, context, query))
            return {"contradiction": 0.1, "neutral": 0.1, "entailment": 0.8}

        verifier.verify_claim = verify_claim

        result = verifier.verify_generation("A claim", ["context"], ["A claim"])

        self.assertEqual(len(calls), 1)
        self.assertTrue(result["verified"])

    def test_invalid_entailment_threshold_is_rejected(self):
        verifier_module = self.verifier_module()
        with self.assertRaises(ValueError):
            verifier_module.EntailmentVerifier(device="cpu", threshold=1.1)


class ExperimentRunnerTests(unittest.TestCase):
    def test_json_result_metrics_are_flattened(self):
        yaml = types.ModuleType("yaml")
        src = types.ModuleType("src")
        src_utils = types.ModuleType("src.utils")
        src_utils.get_commit_hash = lambda: "test"
        src_utils.get_timestamp = lambda: "test"
        runner = load_module(
            "halo_test_runner",
            "experiments/run_final_experiments.py",
            {"yaml": yaml, "src": src, "src.utils": src_utils},
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
