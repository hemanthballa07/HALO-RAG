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

- The verifier currently treats a short answer found anywhere in a passage as
  fully supported, even when the passage does not answer the question. The
  optional single-passage no-answer prompt avoids some of these cases, but it
  does not fix that verification error.
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
by Git. A broader untouched sample and an end-to-end pipeline comparison are
still required.

## Validation required for a release

Run the full experiment matrix on the target CUDA environment, inspect the generated
plots, complete human annotation, and record the exact commit, dependency lock, dataset
revision, seeds, and hardware used for the published results.
