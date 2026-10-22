# Repository Guidelines

## Project Structure & Module Organization

`src/preprocessing/` and `src/anomaly/` contain the portfolio owner's direct contributions. `src/ml/`, `notebooks/ml/`, `notebooks/dl/`, and `src/final_model/` show the rest of the team project: supervised models, deep learning, and combined inference. `configs/` contains aggregate model settings. The repository has no datasets, model weights, reports, or committed test suite. Keep README attribution aligned with these boundaries.

## Build, Test, and Development Commands

Create a Python 3 environment, then install dependencies with `python -m pip install -r requirements-train.txt`. For preprocessing, run `python src/preprocessing/preprocess.py --input /path/to/input.parquet --output /path/to/processed.parquet`. `requirements.txt` contains inference dependencies; the training file adds notebook packages. There is no build step. The anomaly script expects train, validation, and test Parquet files at `/content/` paths near its top. Update paths for an authorized local dataset before running it. Final inference also needs external model artifacts.

## Coding Style & Naming Conventions

Use four spaces for Python indentation. Name modules, functions, and variables in `snake_case`; name constants in `UPPER_SNAKE_CASE`. Preserve existing dataset column names and prediction output fields because downstream modules depend on them. Keep notebook cells valid JSON and clear outputs and execution counts before committing. No formatter or linter configuration is currently included; follow nearby code style.

## Testing Guidelines

There is no automated framework or coverage threshold yet. For preprocessing changes, verify output row counts, columns, dtypes, and missing value handling on permitted local data. For model changes, fit only on training data, tune on validation data, and reserve test data for final evaluation. If you introduce tests, put them in `tests/` and use `test_*.py` filenames. Record the command and dataset version used for manual checks.

## Commit & Pull Request Guidelines

The curated history has no established message convention. Use short imperative subjects such as `fix: preserve preprocessing schema` or `docs: clarify model inputs`. Pull requests should state the changed component, behavior, validation performed, and any schema impact. Link an issue when one exists. Explain whether the work belongs to preprocessing, anomaly detection, or another project part; do not imply ownership of teammates' code.

## Security & Configuration

Commit source and safe aggregate configuration only. Keep raw and processed records, row-level scores, credentials, trained models, and reports out of Git. Check staged files even when `.gitignore` is present; it does not remove already tracked files.
