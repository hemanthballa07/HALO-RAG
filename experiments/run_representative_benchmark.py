"""Compare baseline and revision on a seeded SQuAD v2 sample with distractors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.data import load_dataset_from_config
from src.evaluation.benchmark import build_benchmark, score_answer, summarize_results
from src.pipeline import SelfVerificationRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--corpus-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-retrieve", type=int, default=20)
    parser.add_argument("--top-k-rerank", type=int, default=5)
    parser.add_argument("--max-revisions", type=int, default=1)
    parser.add_argument(
        "--output", default="results/metrics/representative_benchmark.json"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_cases(pipeline, benchmark, seed: int, top_k_retrieve: int, top_k_rerank: int):
    """Run both variants on identical questions and identical initial RNG states."""
    rows = {"baseline": [], "revision": []}
    for index, case in enumerate(benchmark.cases):
        for variant in rows:
            pipeline.enable_revision = variant == "revision"
            set_seed(seed + index)
            start = time.perf_counter()
            result = pipeline.generate(
                case.question,
                top_k_retrieve=top_k_retrieve,
                top_k_rerank=top_k_rerank,
                do_sample=False,
            )
            elapsed = time.perf_counter() - start
            scores = score_answer(
                result["generated_text"], case.references, result.get("abstained", False)
            )
            rows[variant].append({
                "example_id": case.example_id,
                "question": case.question,
                "references": list(case.references),
                "answerable": case.answerable,
                "relevant_doc_id": case.relevant_doc_id,
                "initial_retrieved_doc_ids": result["initial_retrieved_docs"],
                "initial_reranked_doc_ids": result["initial_reranked_docs"],
                "retrieved_doc_ids": result["retrieved_docs"],
                "reranked_doc_ids": result["reranked_docs"],
                "retrieval_hit": float(case.relevant_doc_id in result["initial_retrieved_docs"]),
                "evidence_hit": float(case.relevant_doc_id in result["initial_reranked_docs"]),
                "final_evidence_hit": float(case.relevant_doc_id in result["reranked_docs"]),
                "generated": result["generated_text"],
                "exact_match": scores["exact_match"],
                "f1": scores["f1"],
                "verified": bool(result.get("verified", False)),
                "abstained": bool(result.get("abstained", False)),
                "revision_iterations": result.get("revision_iterations", 0),
                "latency_seconds": round(elapsed, 3),
            })
        print(f"Completed {index + 1}/{len(benchmark.cases)} questions", flush=True)
    return rows


def main() -> int:
    args = parse_args()
    if args.top_k_retrieve <= 0 or args.top_k_rerank <= 0:
        raise ValueError("retrieval and reranking depth must be positive")
    if args.max_revisions < 0:
        raise ValueError("max-revisions cannot be negative")

    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["datasets"]["active"] != "squad_v2":
        raise ValueError("this benchmark currently supports only SQuAD v2")

    full_data_config = copy.deepcopy(config)
    full_data_config["datasets"]["sample_limit"] = None
    examples = load_dataset_from_config(full_data_config, split="validation")
    benchmark = build_benchmark(examples, args.questions, args.corpus_size, args.seed)
    print(
        f"Selected {len(benchmark.cases)} questions across distinct passages "
        f"from a {len(benchmark.corpus)}-document index",
        flush=True,
    )

    revision_config = config.get("revision", {})
    set_seed(args.seed)
    pipeline = SelfVerificationRAGPipeline(
        corpus=list(benchmark.corpus),
        retrieval_model=config["retrieval"]["dense"]["model_name"],
        reranker_model=config["retrieval"]["reranker"]["model_name"],
        generator_model=config["generation"]["model_name"],
        verifier_model=config["verification"]["entailment_model"],
        entailment_threshold=config["verification"]["threshold"],
        dense_weight=config["retrieval"]["fusion"]["dense_weight"],
        sparse_weight=config["retrieval"]["fusion"]["sparse_weight"],
        device=config["experiments"].get("device", "auto"),
        use_qlora=False,
        enable_revision=True,
        max_revision_iterations=args.max_revisions,
        revision_config=revision_config,
    )
    rows = evaluate_cases(
        pipeline, benchmark, args.seed, args.top_k_retrieve, args.top_k_rerank
    )
    summaries = {variant: summarize_results(values) for variant, values in rows.items()}
    comparison = {
        "exact_match_delta": (
            summaries["revision"]["overall"]["exact_match"]
            - summaries["baseline"]["overall"]["exact_match"]
        ),
        "f1_delta": (
            summaries["revision"]["overall"]["f1"]
            - summaries["baseline"]["overall"]["f1"]
        ),
        "exact_match_improved": sum(
            revised["exact_match"] > baseline["exact_match"]
            for baseline, revised in zip(rows["baseline"], rows["revision"])
        ),
        "exact_match_harmed": sum(
            revised["exact_match"] < baseline["exact_match"]
            for baseline, revised in zip(rows["baseline"], rows["revision"])
        ),
    }
    payload = {
        "metadata": {
            "dataset": "squad_v2",
            "split": "validation",
            "seed": args.seed,
            "question_count": args.questions,
            "corpus_size": args.corpus_size,
            "top_k_retrieve": args.top_k_retrieve,
            "top_k_rerank": args.top_k_rerank,
            "max_revisions": args.max_revisions,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "document_sha256": list(benchmark.document_hashes),
            "question_ids": [case.example_id for case in benchmark.cases],
            "models": {
                "retriever": config["retrieval"]["dense"]["model_name"],
                "reranker": config["retrieval"]["reranker"]["model_name"],
                "generator": config["generation"]["model_name"],
                "verifier": config["verification"]["entailment_model"],
            },
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "faiss-cpu")
            },
        },
        "summary": summaries,
        "comparison": comparison,
        "cases": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")
    for variant, summary in summaries.items():
        overall = summary["overall"]
        print(
            f"{variant}: EM={overall['exact_match']:.3f}, F1={overall['f1']:.3f}, "
            f"retrieval hit={overall['retrieval_hit']:.3f}, "
            f"evidence hit={overall['evidence_hit']:.3f}, "
            f"unanswerable false accept={summary['unanswerable']['false_accept_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
