$python_files = Get-ChildItem -Path ./unreasonable_llama/*.py
echo "Checked files: ${python_files}"
poetry run ruff format $python_files
poetry run mypy $python_files
poetry run ruff check --fix $python_files
