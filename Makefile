PYTHON ?= python3
RUFF ?= ruff

.PHONY: check lint syntax test setup-check

check: lint syntax setup-check test

lint:
	$(RUFF) check .

syntax:
	$(PYTHON) -m compileall -q src experiments scripts tests test_revision_strategies.py

setup-check:
	$(PYTHON) scripts/check_setup.py --skip-dependencies

test:
	$(PYTHON) -m unittest tests.test_regressions -v
	$(PYTHON) -m pytest -q tests/test_benchmark.py tests/test_review_scoring.py tests/test_human_eval_agreement.py tests/test_final_result_reader.py tests/test_results_lock.py
