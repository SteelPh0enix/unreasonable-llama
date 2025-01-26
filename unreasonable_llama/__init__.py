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

import os
from dataclasses import dataclass, field

import httpx
from dataclasses_json import Undefined, config, dataclass_json


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelInfo:
    id: str
    created: int
    vocab_type: str
    n_vocab: int
    n_ctx_train: int
    n_embd: int
    n_params: int
    size: int


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class GenerationParams:
    n_predict: int
    seed: int
    temperature: float
    dynatemp_range: float
    dynatemp_exponent: float
    top_k: int
    top_p: float
    min_p: float
    xtc_probability: float
    xtc_threshold: float
    typical_p: float
    repeat_last_n: int
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    dry_multiplier: float
    dry_base: float
    dry_allowed_length: int
    dry_penalty_last_n: int
    dry_sequence_breakers: list[str]
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    stop: list[str]
    max_tokens: int
    n_keep: int
    n_discard: int
    ignore_eos: bool
    stream: bool
    logit_bias: list[int]
    n_probs: int
    min_keep: int
    grammar: str
    samplers: list[str]
    speculative_n_max: int = field(metadata=config(field_name="speculative.n_max"))
    speculative_n_min: int = field(metadata=config(field_name="speculative.n_min"))
    speculative_p_min: float = field(metadata=config(field_name="speculative.p_min"))
    timings_per_token: bool
    post_sampling_probs: bool
    lora: list[str]


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class GenerationSettings:
    id: int
    id_task: int
    n_ctx: int
    speculative: bool
    is_processing: bool
    non_causal: bool
    params: GenerationParams
    prompt: str
    next_token: NextToken


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class NextToken:
    has_next_token: bool
    has_new_line: bool
    n_remain: int
    n_decoded: int
    stopping_word: str


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelProps:
    default_generation_settings: GenerationSettings
    total_slots: int
    model_path: str
    chat_template: str
    build_info: str


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class Slot:
    id: int
    id_task: int
    n_ctx: int
    speculative: bool
    is_processing: bool
    non_casual: bool
    params: GenerationParams
    prompt: str
    next_token: NextToken


class LlamaError(RuntimeError):
    """llama.cpp base exception, should be subclassed if more details
    useful to the user can be provided."""

    def __init__(self, details: str, *args: object) -> None:
        self.details = details
        super().__init__(*args)

    def __str__(self) -> str:
        return f"LlamaError: {self.details}"


class LlamaUrlError(LlamaError):
    """Exception thrown in case of invalid/unknown URL of llama.cpp server"""

    def __init__(self, details: str, host: str | None, port: int | None, *args: object) -> None:
        self.host = host
        self.port = port
        super().__init__(f"{details} ({host}:{port})", *args)


def _make_llama_server_url(host: str | None, port: int | None) -> str:
    """creates llama.cpp server URL out of host/port, if not provided - will
    try to fetch them from environment. Validates the port range."""
    if host is None:
        if env_host := os.getenv("LLAMA_ARG_HOST"):
            host = env_host
        else:
            raise LlamaUrlError("Unknown llama.cpp server host!", host, port)

    if port is None:
        if env_port := os.getenv("LLAMA_ARG_PORT"):
            port = int(env_port)
        else:
            raise LlamaUrlError("Unknown llama.cpp server port!", host, port)

    if port <= 0 or port > 65535:  # noqa: PLR2004
        raise LlamaUrlError("Invalid llama.cpp server port!", host, port)

    return f"http://{host}:{port}"


def health(
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> bool:
    """Returns `True` if server is alive and ready, `False` if it's not ready"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/health", timeout=timeout).json()
    return bool(response.get("status", "") == "ok")


def props(
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> bool:
    """Returns `True` if server is alive and ready, `False` if it's not ready"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/props", timeout=timeout).read()
    return ModelProps.from_json(response)
