# HALO-RAG Finalization Guide

This guide walks through the finalization process for HALO-RAG submission.

## Overview

The finalization process involves:
1. Re-running Exp1-8 with seeds {42, 123, 456} and optimal τ from Exp8
2. Creating final summary metrics (mean ± sd)
3. Copying key plots to `results/figures/final/`
4. Creating RESULTS_LOCK.md with reproducibility info
5. Building Wikipedia FAISS index probe
6. Generating presentation and report

## Step 1: Run Final Experiments

Run all experiments with multiple seeds and aggregate results:

```bash
# Run all experiments with seeds 42, 123, 456
python experiments/run_final_experiments.py \
    --seeds 42 123 456 \
    --split validation \
    --copy-plots

# Or run with dry-run for testing
python experiments/run_final_experiments.py \
    --seeds 42 123 456 \
    --split validation \
    --dry-run \
    --copy-plots
```

This will:
- Run Exp1-8 with each seed
- Aggregate results across seeds (mean ± sd)
- Create `results/metrics/final_summary.csv`
- Copy 6 key plots to `results/figures/final/`
- Save aggregated results to `results/metrics/final_aggregated_results.json`

## Step 2: Create RESULTS_LOCK.md

Generate reproducibility document:

```bash
python scripts/create_results_lock.py \
    --tau 0.75 \
    --seeds 42 123 456 \
    --dataset squad_v2 \
    --split validation
```

This creates `RESULTS_LOCK.md` with:
- Git commit hash and branch
- Dataset configuration
- Experimental configuration
- FAISS index metadata
- Verified data snapshots
- Experiment run timestamps
- Reproducibility instructions

## Step 3: Build Wikipedia FAISS Index Probe

Build a probe Wikipedia FAISS index (200k-500k passages):

```bash
# Build index with 300k passages
python scripts/build_wiki_index_probe.py \
    --num-passages 300000 \
    --index-path data/wiki_index_probe.bin \
    --metadata-path data/INDEX_METADATA.json \
    --probe-output results/metrics/wiki_index_probe.json

# Or skip build and only probe existing index
python scripts/build_wiki_index_probe.py \
    --skip-build \
    --index-path data/wiki_index_probe.bin \
    --probe-output results/metrics/wiki_index_probe.json
```

This will:
- Load Wikipedia passages (200k-500k)
- Build FAISS index with sentence-transformer embeddings
- Save index to `data/wiki_index_probe.bin`
- Generate `data/INDEX_METADATA.json` with index metadata
- Probe index with sample queries
- Save probe metrics to `results/metrics/wiki_index_probe.json`

## Step 4: Generate Presentation

Generate presentation content:

```bash
python scripts/generate_presentation.py \
    --output report/presentation_content.md \
    --metrics results/metrics/final_summary.csv
```

This creates `report/presentation_content.md` with:
- 12 slides covering all experiments
- Quiz slide with 5 questions
- Key metrics and figures
- Structured content ready for PPTX conversion

**Note**: Convert markdown to PPTX using:
- Pandoc: `pandoc presentation_content.md -o final_presentation.pptx`
- Or manually create slides using the content

## Step 5: Generate Final Report

Generate 9-page NeurIPS-style report:

```bash
python scripts/generate_final_report.py \
    --output report/final_report.md \
    --metrics results/metrics/final_summary.csv
```

This creates `report/final_report.md` with:
- Abstract
- Introduction
- Method overview
- Experiments 1-8 results
- Results & Analysis
- Limitations & Future Work
- Conclusion
- References
- Appendix

**Note**: Convert markdown to PDF using:
- Pandoc: `pandoc final_report.md -o final_report.pdf --template=neurips`
- Or manually format using NeurIPS LaTeX template

## Step 6: Verify Outputs

Verify all required outputs are present:

### Metrics Files
- [ ] `results/metrics/final_summary.csv` (mean ± sd for all metrics)
- [ ] `results/metrics/final_aggregated_results.json` (detailed results)
- [ ] `results/metrics/wiki_index_probe.json` (index probe metrics)

### Figures
- [ ] `results/figures/final/retrieval_bars.png` (Exp2)
- [ ] `results/figures/final/tau_sweep.png` (Exp3)
- [ ] `results/figures/final/decoding_comparison.png` (Exp5)
- [ ] `results/figures/final/iteration_curves.png` (Exp6)
- [ ] `results/figures/final/pareto_frontier.png` (Exp8)
- [ ] `results/figures/final/ablation_bars.png` (Exp7)

### Documentation
- [ ] `RESULTS_LOCK.md` (reproducibility document)
- [ ] `data/INDEX_METADATA.json` (FAISS index metadata)
- [ ] `report/presentation_content.md` (presentation content)
- [ ] `report/final_report.md` (final report)

### Index Files
- [ ] `data/wiki_index_probe.bin` (FAISS index, if built)
- [ ] `data/INDEX_METADATA.json` (index metadata)

## Step 7: Acceptance Criteria Checklist

Verify all acceptance criteria are met:

### Final Summary CSV
- [ ] Contains mean ± sd for EM, F1, BLEU-4, ROUGE-L
- [ ] Contains mean ± sd for Factual Precision, Hallucination Rate, Verified F1
- [ ] Contains mean ± sd for Abstention Rate, Recall@20, Coverage
- [ ] Aggregated across seeds {42, 123, 456}

### Key Plots
- [ ] `retrieval_bars.png` (retrieval comparison)
- [ ] `tau_sweep.png` (threshold sweep)
- [ ] `decoding_comparison.png` (decoding strategies)
- [ ] `iteration_curves.png` (iterative training)
- [ ] `pareto_frontier.png` (Pareto frontier)
- [ ] `ablation_bars.png` (ablation study)

### RESULTS_LOCK.md
- [ ] Documents dataset/split, τ, seeds, commit hash
- [ ] Documents FAISS index metadata
- [ ] Documents verified data snapshot paths
- [ ] Documents run timestamps
- [ ] Includes reproducibility instructions

### Wikipedia Index Probe
- [ ] `INDEX_METADATA.json` generated
- [ ] `wiki_index_probe.json` generated
- [ ] Index contains 200k-500k passages
- [ ] Probe metrics reported

### Presentation & Report
- [ ] Presentation content generated (12 slides + quiz)
- [ ] Final report generated (9 pages)
- [ ] Both exported to `report/` directory

## Troubleshooting

### Experiments Fail to Run

If experiments fail:
1. Check dataset is loaded: `python experiments/check_dataset_loading.py`
2. Check config file: `config/config.yaml`
3. Check dependencies: `pip install -r requirements.txt`
4. Check GPU availability (if using CUDA)

### Plots Not Found

If plots are not found:
1. Run experiments first to generate plots
2. Check `results/figures/` directory
3. Verify experiment scripts completed successfully

### Index Build Fails

If index build fails:
1. Check Wikipedia dataset is available
2. Check disk space (index can be large)
3. Check GPU memory (if using CUDA)
4. Try with fewer passages (--num-passages 100000)

### Metrics Not Aggregated

If metrics are not aggregated:
1. Check experiment results files exist
2. Check CSV/JSON format is correct
3. Check seeds were used correctly
4. Run with --skip-runs to test aggregation only

## Next Steps

After finalization:
1. Review all outputs
2. Convert presentation to PPTX
3. Convert report to PDF (NeurIPS template)
4. Create final submission package
5. Tag release: `git tag -a v1.0.0 -m "HALO-RAG final release"`
6. Push tag: `git push origin v1.0.0`

## Contact

For questions or issues, refer to:
- Main README.md
- Experiments README.md
- RESULTS_LOCK.md
- GitHub Issues

---
*Last updated: {get_timestamp()}*

