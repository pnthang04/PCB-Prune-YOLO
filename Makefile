PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"
prepare-data:
	$(PYTHON) scripts/prepare_deeppcb.py --images $(IMAGES) --labels $(LABELS) --output data/processed
validate-data:
	$(PYTHON) scripts/validate_dataset.py --root data/processed
preview-data:
	$(PYTHON) scripts/visualize_annotations.py --root data/processed
train-smoke:
	$(PYTHON) scripts/train_baseline.py --epochs 1 --imgsz 320 --batch 2 --name smoke
train:
	$(PYTHON) scripts/train_baseline.py
evaluate:
	$(PYTHON) scripts/evaluate_model.py --checkpoint $(MODEL)
prune-dry-run:
	$(PYTHON) scripts/prune_model.py --checkpoint $(MODEL) --dry-run
prune:
	$(PYTHON) scripts/prune_model.py --checkpoint $(MODEL) --no-dry-run
finetune:
	$(PYTHON) scripts/finetune_pruned.py --model $(MODEL)
benchmark:
	$(PYTHON) scripts/benchmark_model.py --model $(MODEL)
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

