#!/bin/sh

python_files=$(ls ./*.py)
echo "Checked files: $python_files"

set -xeo pipefail

poetry run ruff format $python_files
poetry run mypy --strict --pretty $python_files
poetry run ruff check --fix $python_files
