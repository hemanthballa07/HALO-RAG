"""Combine compatible paired benchmark runs without hiding split results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, median


PROTOCOL_FIELDS = (
    "dataset", "split", "corpus_size", "top_k_retrieve", "top_k_rerank",
    "max_revisions", "config_sha256", "source_sha256", "models", "package_versions",
)


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    result = {"count": len(rows)}
    for field in ("exact_match", "f1", "retrieval_hit", "evidence_hit", "verified", "abstained"):
        result[field] = fmean(float(row[field]) for row in rows)
    latencies = [float(row["latency_seconds"]) for row in rows]
    result["latency_p50_seconds"] = median(latencies)
    result["latency_p95_seconds"] = _percentile_nearest_rank(latencies, 0.95)
    claims = [
        claim for row in rows for claim in row.get("claim_verification", [])
    ]
    result["claim_method_counts"] = dict(sorted(Counter(
        claim.get("verification_method") or "unknown" for claim in claims
    ).items()))
    result["accepted_claim_method_counts"] = dict(sorted(Counter(
        claim.get("verification_method") or "unknown"
        for claim in claims if claim.get("is_entailed", False)
    ).items()))
    if all(not row["answerable"] for row in rows):
        result["false_accept_rate"] = fmean(
            float(row["verified"] and not row["abstained"]) for row in rows
        )
        result["unanswerable_answer_rate"] = fmean(
            float(not row["abstained"] and bool(row["generated"].strip()))
            for row in rows
        )
        result["false_accept_claim_method_counts"] = dict(sorted(Counter(
            claim.get("verification_method") or "unknown"
            for row in rows if row["verified"] and not row["abstained"]
            for claim in row.get("claim_verification", [])
            if claim.get("is_entailed", False)
        ).items()))
    return result


def combine_runs(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one benchmark file is required")

    sources = []
    combined: dict[str, list[dict]] = {}
    protocol = None
    variants = None
    seeds = set()
    all_question_ids = []
    per_seed = []
    for path in paths:
        source_bytes = path.read_bytes()
        payload = json.loads(source_bytes)
        metadata = payload["metadata"]
        missing = [field for field in PROTOCOL_FIELDS if field not in metadata]
        if missing:
            raise ValueError(f"benchmark metadata missing {', '.join(missing)} in {path}; rerun it")
        current_protocol = {field: metadata[field] for field in PROTOCOL_FIELDS}
        current_variants = list(payload["cases"])
        if protocol is None:
            protocol = current_protocol
            variants = current_variants
            combined = {variant: [] for variant in variants}
        elif current_protocol != protocol or current_variants != variants:
            changed = [
                field for field in PROTOCOL_FIELDS
                if current_protocol[field] != protocol[field]
            ]
            if current_variants != variants:
                changed.append("variants")
            raise ValueError(
                f"incompatible benchmark protocol in {path}: {', '.join(changed)}"
            )
        if metadata["seed"] in seeds:
            raise ValueError(f"duplicate seed {metadata['seed']}")
        seeds.add(metadata["seed"])

        question_ids = metadata["question_ids"]
        if len(question_ids) != metadata["question_count"]:
            raise ValueError(f"question count mismatch in {path}")
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(f"repeated question ID within {path}")
        for variant in variants:
            rows = payload["cases"][variant]
            if [row["example_id"] for row in rows] != question_ids:
                raise ValueError(f"case order mismatch for {variant} in {path}")
            combined[variant].extend(rows)
        all_question_ids.extend(question_ids)
        per_seed.append({
            "seed": metadata["seed"],
            "question_count": len(question_ids),
            "summary": payload["summary"],
        })
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        })

    summary = {}
    for variant, rows in combined.items():
        summary[variant] = {
            "overall": _summarize_rows(rows),
            "answerable": _summarize_rows([row for row in rows if row["answerable"]]),
            "unanswerable": _summarize_rows([row for row in rows if not row["answerable"]]),
        }
    paired = {}
    if "baseline" not in combined:
        raise ValueError("benchmark files must include a baseline variant")
    baseline = combined["baseline"]
    for variant in variants:
        if variant == "baseline":
            continue
        other = combined[variant]
        deltas = [row["exact_match"] - base["exact_match"] for base, row in zip(baseline, other)]
        paired[variant] = {
            "exact_match_delta": fmean(deltas),
            "improved": sum(delta > 0 for delta in deltas),
            "harmed": sum(delta < 0 for delta in deltas),
        }

    duplicates = Counter(all_question_ids)
    return {
        "protocol": protocol,
        "variants": variants,
        "sources": sources,
        "per_seed": per_seed,
        "duplicate_question_ids_across_runs": sorted(
            question_id for question_id, count in duplicates.items() if count > 1
        ),
        "summary": summary,
        "paired_vs_baseline": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = combine_runs(args.benchmarks)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
