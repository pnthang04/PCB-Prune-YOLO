PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps
prepare-data:
	$(PYTHON) scripts/prepare_deeppcb.py
validate-data:
	$(PYTHON) scripts/validate_dataset.py
preview-data:
	$(PYTHON) scripts/visualize_annotations.py
train-smoke:
	$(PYTHON) scripts/train_baseline.py --smoke --batch 128 --fraction 0.2
train:
	$(PYTHON) scripts/train_baseline.py
evaluate:
	$(PYTHON) scripts/evaluate_model.py --checkpoint $(MODEL)
benchmark:
	$(PYTHON) scripts/benchmark_model.py --model $(MODEL)
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
