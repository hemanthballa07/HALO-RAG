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

- The current FAISS implementation uses an exact inner-product index. It is appropriate
  for the sampled experiment corpora but should be replaced with a trained approximate
  index before indexing millions of passages.
- Automatic verification reduces unsupported claims but cannot guarantee factuality.
- Metric targets in `config/config.yaml` are research goals and need to be established by
  fresh runs on the chosen dataset and hardware.
- The root `final_run_results.zip` is a historical archive produced before the current
  loader, retrieval, and verifier corrections. It is retained for provenance and should
  not be treated as a current benchmark result.

## Validation required for a release

Run the full experiment matrix on the target CUDA environment, inspect the generated
plots, complete human annotation, and record the exact commit, dependency lock, dataset
revision, seeds, and hardware used for the published results.
