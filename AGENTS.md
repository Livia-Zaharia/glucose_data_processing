# Agent Guidelines

This document outlines the coding standards and practices for this project.

## Repository layout (uv workspace)

This repo is a **uv workspace**

### Running the App

The recommended way to start the application is:

- `glucose-process <input_folder>`
- Or using uv: `uv run glucose-process <input_folder>`

## Coding Standards

- **Avoid nested try-catch**: try catch often just hide errors, put them only when errors is what we consider unavoidable in the use-case
- **Type hints**: Mandatory for all Python code.
- **Pathlib**: Always use for all file paths.
- **No relative imports**: Always use absolute imports.
- **Polars**: Prefer over Pandas. Consider using polars expressions to speed up code where relevant. 
- **Be memory efficient**: Use lazyframes (`scan_parquet`) and streaming (`sink_parquet`) for efficiency if applicable, use engine="streaming". Also, remember that you can load multiple files with one scan and can avoid python loops in many cases. Avoid redundant collect-s
- **Memory efficient joins**:  if you make joins make sure largest dataframe goes first. Joins often materializes whole second lazyframe. Pre-filter dataframes before joining to avoid materialization.
- **Typer CLI**: Mandatory for all CLI tools.
- **No placeholders**: Never use `/my/custom/path/` in code.
- **No legacy support**: Refactor aggressively; do not keep old API functions.
- **Dependency Management**: Use `uv sync` and `uv add`. NEVER use `uv pip install`.
- **Versions**: Do not hardcode versions in `__init__.py`; use `project.toml`.
- **Literals**: Prefer declaring constants than using hardcoded strings. Constants easyer to reuse and update.
- **Logging**: For logging use loguru library. It gave additional flexibility when choosing loging channel.

## Testing & Docs

- **Tests**: After major code updates always run tests in tests folder with pytest. If introducing new feature or business logic update look through tests and also update them to reflect changes.
- **Regression tests**: Before code update make checkpoint result with CLI script for data/uom_small dataset. After code update make one more checkpoint and compare them with compare_checkpoints.py. Result checkpoints should match. Exception is only when there are changes in business logic that modify data or fields. In this case compare differences with expected outcomes. Save checkpoints in to checkpoints folder.
- **Docs**: Put all new markdown files (except README/AGENTS) in `docs/`.
