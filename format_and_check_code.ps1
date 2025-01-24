$python_files = Get-ChildItem -Path ./**/*.py
echo "Checked files: ${python_files}"
uv run ruff format $python_files
uv run mypy $python_files
uv run ruff check --fix $python_files
