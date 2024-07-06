# unreasonable-llama

(Yet another) Python API for [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

For now, i'm targeting minimal support necessary for `/completion` and `/health` endpoint.
Maybe i'll extend this lib in the future.

## Requirements

`unreasonable-llama` has a single requirement - `httpx` library, see `pyproject.toml` for details.

Requirements for `llama.cpp` scripts can be found in `requirements/` directory of theirs repository.

## Usage

See [test file](./tests/__main__.py) for example usage.
