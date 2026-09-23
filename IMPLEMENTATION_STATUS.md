# Implementation status

## Available components

- Hybrid FAISS and BM25 retrieval with cross-encoder reranking
- FLAN-T5 generation with optional CUDA QLoRA adapters
- Claim extraction and checkpoint-aware NLI label mapping
- Entailment-based verification and three adaptive revision strategies
- SQuAD v2, Natural Questions, and HotpotQA loaders with a shared schema
- Retrieval, answer-quality, factuality, coverage, FEVER-style, and abstention metrics
- Experiments 1 through 9 plus a human-evaluation workflow
- Dependency-free regression tests, repository validation, and GitHub Actions CI

## Operational status

The lightweight quality gate passes without downloading models:

```bash
python -m pip install -r requirements-dev.txt
make check
```

End-to-end experiments require the runtime dependency set, model downloads, dataset
downloads, and substantially more local storage. Experiment 6 additionally requires a
CUDA host with `bitsandbytes`.

## Known limits

- Short-answer verification now checks whether an answer-bearing sentence
  shares content terms with the question before using a direct-match shortcut.
  This catches obvious unrelated matches, but lexical co-occurrence is not a
  complete answer-support test. The NLI fallback also makes mistakes.
- The current FAISS implementation uses an exact inner-product index. It is appropriate
  for the sampled experiment corpora but should be replaced with a trained approximate
  index before indexing millions of passages.
- Automatic verification reduces unsupported claims but cannot guarantee factuality.
- Metric targets in `config/config.yaml` are research goals and need to be established by
  fresh runs on the chosen dataset and hardware.
- The root `final_run_results.zip` is a historical archive produced before the current
  loader, retrieval, and verifier corrections. It is retained for provenance and should
  not be treated as a current benchmark result.

## Exploratory CPU check

On September 22, 2026, three seeded SQuAD v2 validation samples used 20 questions
and 500 passages each, with 10 answerable and 10 unanswerable questions per seed.
The same cached FLAN-T5 large model generated the answers in all comparisons.
The baseline and revision runs below preceded the verifier shortcut guard.

| Seed | Baseline exact match | Revision exact match | Top-passage no-answer exact match |
| --- | ---: | ---: | ---: |
| 42 | 35% | 45% | 75% |
| 123 | 30% | 40% | 55% |
| 456 | 30% | 30% | 75% |

The top-passage result reuses the saved reranking order and reruns generation only.
It does not include verification or revision. Across these 60 sampled questions,
its exact match was 68.3%, including 56.7% on the 30 unanswerable questions.
The samples were used while developing the prompt, so they are not a held-out
estimate of production accuracy. Local per-question results are under
`results/metrics/benchmark_20q_500docs_seed*_top1_abstain.json` and are ignored
by Git. Broader untouched samples are still required.

A verifier-only replay of the saved baseline answers on those same three samples
reduced verified answers for unanswerable questions from 24 to 18 out of 30.
It kept all 19 previously verified exact answers verified. This replay held
retrieval and generation fixed, and it is not a new end-to-end score. False
acceptance remains substantial, so the verifier needs further work.

A separate seed 789 run tested all three variants end to end on the same
20 questions and 500 passages. This seed was not used to select the prompt,
but one small sample is not a release estimate.

| Variant | Overall exact match | Answerable | Unanswerable | Unanswerable false accept | Mean CPU latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 45% | 90% | 0% | 30% | 2.41 s |
| Revision | 65% | 70% | 60% | 40% | 2.62 s |
| Focused, no revision | 95% | 100% | 90% | 0% | 1.06 s |

The focused variant abstained on 9 of 10 unanswerable questions. Its one
non-abstaining answer was wrong and was not verified. These figures are from
`results/metrics/benchmark_20q_500docs_seed789_with_focused.json`, which is
ignored by Git. More seeds, other datasets, and the target CUDA environment
must be checked before choosing a default.

## Larger paired CPU check

Two further runs used seeds 2026 and 2027, each with 40 SQuAD v2 validation
questions and a separate 1,000-passage corpus. The configurations and model
versions matched, and the selected question IDs did not overlap across runs.
The pooled figures below cover 80 questions, split evenly by answerability.

| Variant | Overall exact match | Answerable | Unanswerable | Label-based false accept | CPU latency p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 35% | 70% | 0% | 60% | 1.63 / 3.81 s |
| Revision | 45% | 60% | 30% | 70% | 1.57 / 4.75 s |
| Focused, no revision | 70% | 80% | 60% | 25% | 0.97 / 1.74 s |

Focused improved exact match on 31 paired questions and harmed it on 3,
relative to baseline. Latency covers per-question retrieval through output on
this CPU host; it excludes model loading and index construction. The 95% result
on seed 789 did not persist in these larger samples. Do not choose a default
from these results alone.

The false-accept column counts verified non-abstaining answers on questions
labeled unanswerable by SQuAD v2. Several reviewed passages contain plausible
answers despite that label, so this number is not a direct hallucination rate.
The 24 nonexact focused cases were exported to
`results/human_eval/focused_seed2026_2027_review.csv` for separate relevance
and evidence-support judgments. The individual runs and combined summary are
under `results/metrics/benchmark_40q_1000docs_*`; these local artifacts are
ignored by Git. Human review, other datasets, and a target-hardware run remain
open before release.

## Validation required for a release

Run the full experiment matrix on the target CUDA environment, inspect the generated
plots, complete human annotation, and record the exact commit, dependency lock, dataset
revision, seeds, and hardware used for the published results.
