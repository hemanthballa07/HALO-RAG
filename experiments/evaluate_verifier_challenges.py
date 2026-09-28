"""Run a small, labeled challenge set against the claim verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = PROJECT_ROOT / "experiments/fixtures/verifier_challenges.json"


def load_challenges(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("challenge fixture must be a nonempty list")

    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each challenge must be an object")
        for field in ("id", "question", "answer", "context"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"challenge {field} must be a nonempty string")
        if not isinstance(case.get("supported"), bool):
            raise ValueError("challenge supported must be a boolean")
        if case["id"] in seen:
            raise ValueError(f"duplicate challenge id: {case['id']}")
        seen.add(case["id"])
    return cases


def evaluate_cases(cases: list[dict], verifier) -> dict:
    rows = []
    counts = Counter()
    false_accept_methods = Counter()
    for case in cases:
        score = verifier.verify_claim(
            case["answer"], case["context"], query=case["question"]
        )
        accepted = score["entailment"] >= verifier.threshold
        supported = case["supported"]
        outcome = (
            "true_accepts" if accepted and supported else
            "false_accepts" if accepted else
            "false_rejects" if supported else
            "true_rejects"
        )
        counts[outcome] += 1
        method = score.get("method", "unknown")
        if outcome == "false_accepts":
            false_accept_methods[method] += 1
        rows.append({
            "id": case["id"],
            "supported": supported,
            "accepted": accepted,
            "entailment_score": score["entailment"],
            "verification_method": method,
        })

    return {
        "summary": {
            "count": len(rows),
            "true_accepts": counts["true_accepts"],
            "false_rejects": counts["false_rejects"],
            "false_accepts": counts["false_accepts"],
            "true_rejects": counts["true_rejects"],
            "false_accept_methods": dict(sorted(false_accept_methods.items())),
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-false-accepts", type=int)
    args = parser.parse_args()
    if args.max_false_accepts is not None and args.max_false_accepts < 0:
        parser.error("--max-false-accepts must be nonnegative")

    cases = load_challenges(args.fixture)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.verification.entailment_verifier import EntailmentVerifier

    verifier = EntailmentVerifier(model_name=args.model, threshold=args.threshold)
    result = evaluate_cases(cases, verifier)
    result["metadata"] = {
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "model": args.model,
        "threshold": args.threshold,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if (args.max_false_accepts is not None
            and result["summary"]["false_accepts"] > args.max_false_accepts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
