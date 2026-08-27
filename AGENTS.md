# Repository Guidelines

## Project Structure & Module Organization

`train.py` is the experiment entry point; `tune.py` runs hyperparameter grids, and `rebuild_tune_html.py` reconstructs reports. Models live under `model/{mlp,cnn,gnn,rnn}/`; keep shared math in the nearest `utils.py` and re-export public classes through package `__init__.py` files. Dataset loading and preprocessing belong in `datasets/`. Ignored downloads go in `data/`; generated checkpoints, logs, and reports go in `output/` and `tune_results/`. Theory and references live in `fgib_theory.tex` and `paper/`.

## Build, Test, and Development Commands

- `python train.py --task mnist --model mlp --epochs 1 --runs 1 --no-save` performs a quick end-to-end smoke run.
- `python train.py --task cora --backbone gnn --model fgib --beta 0.001` runs a specific experiment. Use `python train.py --help` for the complete matrix.
- `python tune.py --model vib ceb fgib --beta 1e-4 1e-3 --anchor-scale 1 4 --parallel 2` runs a small tuning grid and generates HTML results.
- `python rebuild_tune_html.py --model vib ceb fgib --beta 1e-4 1e-3` rescans existing logs; add `--rerun` to fill gaps.
- `python -m compileall train.py tune.py rebuild_tune_html.py datasets model` checks syntax without training.

`bash run.sh` launches the full multi-dataset sweep and is GPU- and time-intensive. The documented environment is Python 3.12 with PyTorch, torchvision, and scikit-learn. There is no lockfile or build step.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python layout. Name functions, variables, and modules with `snake_case`; use `PascalCase` for model classes and `UPPER_CASE` for constants. Match the existing concise Chinese comments/docstrings, but keep identifiers in English. Preserve the common model return contract and numerical conventions such as clamped log-variance and zero-initialized VIB/CEB heads. No formatter or linter is configured, so keep imports grouped and avoid unrelated formatting churn.

## Testing Guidelines

No automated test suite or coverage target exists. Before submitting, run `compileall` plus a one-epoch, one-run smoke experiment for every affected task/backbone. Use a fixed `--seed`, prefer `--no-save`, and report validation/test metrics. Put new focused tests in `tests/test_<feature>.py` if adding independently testable preprocessing or mathematical helpers.

## Commit & Pull Request Guidelines

Recent commits use short Chinese, outcome-focused subjects without Conventional Commit prefixes. Follow that style and keep each commit to one concern. Pull requests should explain the model/dataset behavior changed, list exact verification commands, and include representative metrics. Attach screenshots only when HTML reports change. Avoid committing downloaded data, checkpoints, cache files, or large regenerated result trees unless they are the intended deliverable; never place credentials in scripts or logs.
