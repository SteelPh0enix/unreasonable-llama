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
    * `/health` (GET) [is_alive()]
    * `/props` (GET) [props()]
    * `/completions` (POST) [complete(request), stream_completion(request)]
    * `/tokenize` (POST) [tokenize(message)]
    * `/detokenize` (POST) [detokenize(tokens)]
    * `/apply-template` (POST) [apply_template(messages)]

Note: `complete` and `streamed_complete` accept both tokenized and raw prompt.

This library uses `httpx`. In case of connection issues, expect
`httpx.ConnectError` to happen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx
from dataclasses_json import Undefined, config, dataclass_json  # pyright: ignore[reportUnknownVariableType]

DEFAULT_REQUEST_TIMEOUT: float = 60.0


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelInfo:
    created: int
    id: str
    n_ctx_train: int
    n_embd: int
    n_params: int
    n_vocab: int
    size: int
    vocab_type: str


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class GenerationParams:
    chat_format: str
    dry_allowed_length: int
    dry_base: float
    dry_multiplier: float
    dry_penalty_last_n: int
    dry_sequence_breakers: list[str]
    dynatemp_exponent: float
    dynatemp_range: float
    frequency_penalty: float
    grammar: str
    grammar_lazy: bool
    grammar_triggers: list[str]
    ignore_eos: bool
    logit_bias: list[int]
    lora: list[str]
    max_tokens: int
    min_keep: int
    min_p: float
    mirostat: int
    mirostat_eta: float
    mirostat_tau: float
    n_discard: int
    n_keep: int
    n_predict: int
    n_probs: int
    post_sampling_probs: bool
    presence_penalty: float
    preserved_tokens: list[int]
    reasoning_format: str
    reasoning_in_content: bool
    repeat_last_n: int
    repeat_penalty: float
    samplers: list[str]
    seed: int
    speculative_n_max: int = field(metadata=config(field_name="speculative.n_max"))  # pyright: ignore[reportUnknownArgumentType]
    speculative_n_min: int = field(metadata=config(field_name="speculative.n_min"))  # pyright: ignore[reportUnknownArgumentType]
    speculative_p_min: float = field(metadata=config(field_name="speculative.p_min"))  # pyright: ignore[reportUnknownArgumentType]
    stop: list[str]
    stream: bool
    temperature: float
    thinking_forced_open: bool
    timings_per_token: bool
    top_k: int
    top_n_sigma: float
    top_p: float
    typical_p: float
    xtc_probability: float
    xtc_threshold: float


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class GenerationSettings:
    id: int
    id_task: int
    is_processing: bool
    n_ctx: int
    next_token: NextToken
    params: GenerationParams
    prompt: str
    speculative: bool


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class NextToken:
    has_new_line: bool
    has_next_token: bool
    n_decoded: int
    n_remain: int
    stopping_word: str


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelModalities:
    audio: bool
    vision: bool


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelProps:
    bos_token: str
    build_info: str
    chat_template: str
    default_generation_settings: GenerationSettings
    eos_token: str
    modalities: ModelModalities
    model_path: str
    total_slots: int


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ChatMessage:
    role: str
    content: str


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


def is_alive(
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> bool:
    """Returns `True` if server is alive and ready, `False` if it's not ready (model is still loading)"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/health", timeout=timeout).json()
    return bool(response.get("status", "") == "ok")


def props(
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> ModelProps:
    """Returns `True` if server is alive and ready, `False` if it's not ready"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/props", timeout=timeout).read()
    return ModelProps.from_json(response)  # type: ignore


def apply_template(
    messages: list[ChatMessage],
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """Applies chat template to provided list of messages, and returns a single message that's
    ready to be passed to /completion endpoint."""
    server_url = _make_llama_server_url(server_host, server_port)
    messages_list = [message.to_dict() for message in messages]  # type: ignore
    response = httpx.post(f"{server_url}/apply-template", timeout=timeout, json={"messages": messages_list}).json()  # pyright: ignore[reportUnknownArgumentType]
    return response.get("prompt")  # type: ignore
