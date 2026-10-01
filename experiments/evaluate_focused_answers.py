"""Compare single-passage generation prompts on a saved benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data import load_dataset_from_config
from src.evaluation.benchmark import build_benchmark, score_answer
from src.generator import FLANT5Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Override the configured generator model")
    parser.add_argument(
        "--prompt", choices=("abstain", "standard"), default="abstain",
        help="Compare the abstention prompt with the existing generator prompt",
    )
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    def group_summary(group: list[dict]) -> dict:
        return {
            "count": len(group),
            "exact_match": fmean(row["exact_match"] for row in group) if group else None,
            "f1": fmean(row["f1"] for row in group) if group else None,
            "abstained": fmean(row["abstained"] for row in group) if group else None,
            "evidence_hit": fmean(row["evidence_hit"] for row in group) if group else None,
        }

    return {
        "overall": group_summary(rows),
        "answerable": group_summary([row for row in rows if row["answerable"]]),
        "unanswerable": group_summary([row for row in rows if not row["answerable"]]),
    }


def main() -> int:
    args = parse_args()
    source_bytes = args.benchmark.read_bytes()
    source = json.loads(source_bytes)
    metadata = source["metadata"]
    if metadata["dataset"] != "squad_v2" or metadata["split"] != "validation":
        raise ValueError("the source must be a SQuAD v2 validation benchmark")

    config_bytes = args.config.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != metadata["config_sha256"]:
        raise ValueError("the configuration does not match the saved benchmark")
    config = yaml.safe_load(config_bytes)
    data_config = copy.deepcopy(config)
    data_config["datasets"]["sample_limit"] = None
    examples = load_dataset_from_config(data_config, split="validation")
    benchmark = build_benchmark(
        examples,
        metadata["question_count"],
        metadata["corpus_size"],
        metadata["seed"],
    )
    if list(benchmark.document_hashes) != metadata["document_sha256"]:
        raise ValueError("the saved corpus does not match the cached dataset")
    if [case.example_id for case in benchmark.cases] != metadata["question_ids"]:
        raise ValueError("the saved questions do not match the cached dataset")

    baseline_rows = source["cases"]["baseline"]
    if len(baseline_rows) != len(benchmark.cases):
        raise ValueError("the saved benchmark has an incomplete baseline")
    for case, baseline in zip(benchmark.cases, baseline_rows):
        if baseline["example_id"] != case.example_id:
            raise ValueError("the saved baseline order does not match the sample")
        evidence_ids = baseline["initial_reranked_doc_ids"]
        if not evidence_ids or not 0 <= evidence_ids[0] < len(benchmark.corpus):
            raise ValueError("the saved baseline has no valid top passage")

    model_name = args.model or config["generation"]["model_name"]
    if not args.model and model_name != metadata["models"]["generator"]:
        raise ValueError("the generator model does not match the saved benchmark")
    generator = FLANT5Generator(model_name=model_name, use_qlora=False)
    abstain_if_unanswered = args.prompt == "abstain"
    rows = []
    for index, (case, baseline) in enumerate(zip(benchmark.cases, baseline_rows), start=1):
        evidence_id = baseline["initial_reranked_doc_ids"][0]
        raw_answer = generator.generate(
            case.question,
            benchmark.corpus[evidence_id],
            do_sample=False,
            max_new_tokens=32,
            abstain_if_unanswered=abstain_if_unanswered,
        )
        abstained = (
            abstain_if_unanswered
            and FLANT5Generator.is_unanswerable_response(raw_answer)
        )
        scores = score_answer(raw_answer, case.references, abstained=abstained)
        rows.append({
            "example_id": case.example_id,
            "question": case.question,
            "answerable": case.answerable,
            "references": list(case.references),
            "evidence_doc_id": evidence_id,
            "evidence_hit": float(evidence_id == case.relevant_doc_id),
            "generated": raw_answer,
            "abstained": abstained,
            **scores,
        })
        print(f"Completed {index}/{len(benchmark.cases)} questions", flush=True)

    output = args.output or args.benchmark.with_name(
        args.benchmark.stem + f"_top1_{args.prompt}.json"
    )
    payload = {
        "metadata": {
            "source_benchmark": str(args.benchmark),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "dataset": "squad_v2",
            "split": "validation",
            "seed": metadata["seed"],
            "prompt": f"single_passage_{args.prompt}",
            "generator_model": model_name,
        },
        "summary": summarize(rows),
        "baseline_summary": source["summary"]["baseline"],
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    overall = payload["summary"]["overall"]
    print(f"Saved {output}")
    print(f"Top passage ({args.prompt}): EM={overall['exact_match']:.3f}, F1={overall['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
