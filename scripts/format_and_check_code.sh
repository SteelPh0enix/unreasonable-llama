#!/bin/sh

python_files=$(ls ./unreasonable_llama/*.py)
echo "Checked files: $python_files"

set -xeo pipefail

uv run ruff format $python_files
uv run mypy $python_files
uv run ruff check --fix $python_files
