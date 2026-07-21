# Contributing

This repository includes both canonical runtime paths and many historical experiments. Keep changes focused and low-risk.

## Canonical entrypoints

- `workflow.py` — main orchestration entrypoint (`health-check`, `inference`, `evaluate`, `serve`)
- `train_arc.py` — ARC training entrypoint
- `eval_harness/` — evaluation CLI (`python -m eval_harness`)

## Script categories

- **Training:** `train_*.py`, `training/`
- **Evaluation/benchmarks:** `eval_*.py`, `benchmarks/`, `eval_harness/`
- **Submission pipelines:** `arc_agi2_submission/`, `run_rearc_*.py`
- **Serving/demo:** `serve.py`, `octo_server.py`, `inference_service.py`
- **Tests:** `tests/` (prefer this directory over root-level ad-hoc test scripts)

## Local setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Lightweight validation before PR

```bash
ruff check config.py cognition.py inference.py workflow.py health_check.py --ignore E501
python -m pytest -q tests/test_eval_harness.py tests/test_workflow.py tests/test_unified.py
python -m py_compile train_arc.py workflow.py inference.py model.py config.py cognition.py
```
