"""
llama.cpp typed python bindings

* stateless
* fully typed
* uses the same env variables as llama.cpp server (listed below)

If server's host/port is not specified, following environmental variables are
used instead:
    * `LLAMA_ARG_HOST`
    * `LLAMA_ARG_PORT`

This library is still WIP and v0.x, major rewrites should be expected between
minor releases.
I'm trying to keep it up-to-date with llama.cpp master, but on major changes
it usually takes me a while to notice and fix stuff - PRs are welcome!

Currently supported endpoints (methods) [functions that support them]:
    * `/health` (GET) [health()]
    * `/props` (GET) [props()]
    * `/models` (GET) [models()]
    * `/completions` (POST) [complete(request), streamed_complete(request)]
    * `/tokenize` (POST) [tokenize(message)]
    * `/detokenize` (POST) [detokenize(tokens)]
    * `/slots` (GET) [slots()]

Note: `complete` and `streamed_complete` accept both tokenized and raw prompt.

This library uses `httpx`. In case of connection issues, expect
`httpx.ConnectError` to happen.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """
    Model information, fetched via `/model` endpoint
    Use `models()` to
    """

    id: str
    """Model's ID"""
    created: int
    """Creation date"""
    vocab_type: str
    """Vocabulary type"""
    n_vocab: int
    """Size of the vocabulary"""
    n_ctx_train: int
    """Size of the context the model was trained on"""
    n_embd: int
    n_params: int
    """Amount of model's parameters"""
    size: int
    """Model's size"""
