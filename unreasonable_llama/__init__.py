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

This librarty uses `httpx`. In case of connection issues, expect
`httpx.ConnectError` to happen.

Authenthication and error handling is not implemented yet.

I develop this library mostly for myself - if you want to see more endpoints
supported, make PRs.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, cast

import httpx


@dataclass(frozen=True)
class LlamaGenerationSettings:
    """LLM generation settings"""

    n_ctx: int
    """Context length"""
    n_predict: int
    """Maximum amount of tokens to predict"""
    model: str
    """Model name"""
    seed: int
    """Seed used for RNG"""
    seed_cur: int
    temperature: float
    """Temperature controls the probability distribution of tokens selected by LLM.
    Temperature shouldn't be below 0."""
    dynatemp_range: float
    """Dynamic temperature range, if non-zero, defines the range of temperature used during token prediction.
    Final temperature will be applied based on tokens entropy."""
    dynatemp_exponent: float
    """Dynamic temperature exponent, 1 by default"""
    top_k: int
    """Top-K sampling limits the number of tokens considered during prediction to specified value."""
    top_p: float
    """Top-P sampling limits the number of tokens considered during prediction based on their cumulative probability."""
    min_p: float
    """Min-P defines the minimum probability of token to be considered during prediction."""
    xtc_probability: float
    """This parameter tweaks the chance of XTC sampling happening."""
    xtc_threshold: float
    """XTC removes tokens with probability above specified threshold, except least probable one of them."""
    # NOTE: This was removed in recent llama.cpp release, i'm keeping it here commented for legacy purposes.
    # tfs_z: float
    # """Tail-free sampling removes the tokens with less-than-desired second derivative of it's probability.
    # This parameter defines the probability in (0, 1] range, where 1 == TFS disabled."""
    typical_p: float
    """Locally typical sampling can increase diversity of the text without major coherence degradation by choosing tokens that are typical or expected based on the context.
    This parameter defines probability in range (0, 1], where 1 == locally typical sampling disabled."""
    repeat_last_n: int
    """Amount of last tokens to penalize for repetition. Setting this to 0 disabled penalization, and -1 penalizes the whole context."""
    repeat_penalty: float
    """Penalty for repeating the tokens in generated text."""
    presence_penalty: float
    """Penalty for re-using the tokens that are already present in generated text. 0 == presence penalty disabled."""
    frequency_penalty: float
    """Penalty applied for re-using the tokens that are already present in generated text, based on the frequency of their appearance. 0 == frequency penalty disabled."""
    dry_multiplier: float
    """DRY sampling repetition penalty multiplier. Penalty is calculated with following formula: multiplier * base ^ (length of sequence before token - allowed length).
    DRY sampling is described here: https://github.com/oobabooga/text-generation-webui/pull/5677"""
    dry_base: float
    """DRY sampling base penalty. See dry_multiplier docs for details."""
    dry_allowed_length: int
    """Tokens extending repetitions beyond this receive penalty. See dry_multiplier docs for details."""
    dry_penalty_last_n: int
    """How many tokens should be scanned for repetition (0 = penalization disabled, -1 = whole context)"""
    dry_sequence_breakers: list[str]
    """DRY sampler's sequence breakers."""
    mirostat: int
    """Mirostat type, 1 - Mirostat, 2 - Mirostat 2.0, 0 - disabled.
    ENABLING MIROSTAT DISABLES OTHER SAMPLERS!"""
    mirostat_tau: float
    """Mirostat target entropy, desired perplexity for generated text."""
    mirostat_eta: float
    """Mirostat learning rate."""
    penalize_nl: bool
    """Penalize newline tokens?"""
    stop: list[str]
    """List of strings stopping the generation."""
    max_tokens: int
    """Maximum amount of generated tokens."""
    n_keep: int
    """Amount of tokens to keep from initial prompt when context is filled and it shifts."""
    n_discard: int
    """Number of tokens after n_keep that may be discarded when shifront context, 0 default to half."""
    ignore_eos: bool
    """Ignore end-of-sentence token?"""
    stream: bool
    """Is completion streamed?"""
    n_probs: int
    """If greater than 0, llama.cpp server will output the probabilities of top n_probs tokens.
    Not supported in requests yet."""
    min_keep: int
    """If greater than 0, forces the sampler to return at least min_keep tokens."""
    grammar: str
    """Custom, optional BNF-like grammar to constrain sampling."""
    samplers: list[str]
    """List of used samplers in order."""

    @staticmethod
    def _from_llama_cpp_response(response: dict[str, Any]) -> LlamaGenerationSettings:
        """hard-coded conversion from JSON to LlamaGenerationSettings"""
        return LlamaGenerationSettings(
            n_ctx=response["n_ctx"],
            n_predict=response["n_predict"],
            model=response["model"],
            seed=response["seed"],
            seed_cur=response["seed_cur"],
            temperature=response["temperature"],
            dynatemp_range=response["dynatemp_range"],
            dynatemp_exponent=response["dynatemp_exponent"],
            top_k=response["top_k"],
            top_p=response["top_p"],
            min_p=response["min_p"],
            xtc_probability=response["xtc_probability"],
            xtc_threshold=response["xtc_threshold"],
            # tfs_z=response["tfs_z"],
            typical_p=response["typical_p"],
            repeat_last_n=response["repeat_last_n"],
            repeat_penalty=response["repeat_penalty"],
            presence_penalty=response["presence_penalty"],
            frequency_penalty=response["frequency_penalty"],
            dry_multiplier=response["dry_multiplier"],
            dry_base=response["dry_base"],
            dry_allowed_length=response["dry_allowed_length"],
            dry_penalty_last_n=response["dry_penalty_last_n"],
            dry_sequence_breakers=response["dry_sequence_breakers"],
            mirostat=response["mirostat"],
            mirostat_tau=response["mirostat_tau"],
            mirostat_eta=response["mirostat_eta"],
            penalize_nl=response["penalize_nl"],
            stop=response["stop"],
            max_tokens=response["max_tokens"],
            n_keep=response["n_keep"],
            n_discard=response["n_discard"],
            ignore_eos=response["ignore_eos"],
            stream=response["stream"],
            n_probs=response["n_probs"],
            min_keep=response["min_keep"],
            grammar=response["grammar"],
            samplers=response["samplers"],
        )


@dataclass(frozen=True)
class LlamaProps:
    """llama.cpp server properties."""

    default_generation_settings: LlamaGenerationSettings
    """Default generation settings for currently loaded model."""
    total_slots: int
    """Amount of slots supported by the server."""
    chat_template: str
    """Chat template for currently loaded model."""

    @staticmethod
    def _from_llama_cpp_response(response: dict[str, Any]) -> LlamaProps:
        """hard-coded converstion from JSON to LlamaProps"""
        generation_settings = LlamaGenerationSettings._from_llama_cpp_response(response["default_generation_settings"])
        return LlamaProps(
            default_generation_settings=generation_settings,
            total_slots=response["total_slots"],
            chat_template=response["chat_template"],
        )


class LlamaVocabType(IntEnum):
    """llama.cpp vocabulary type"""

    NONE = 0
    """For models without vocab"""
    SPM = 1
    """LlaMA tokenizer based on byte-level BPE with byte fallback"""
    BPE = 2
    """GPT-2 tokenizer based on byte-level BPE"""
    WPM = 3
    """BERT tokenizer based on WordPiece"""
    UGM = 4
    """T5 tokenizer based on Unigram"""
    RWKV = 5
    """RWKV tokenizer based on greedy tokenization"""


@dataclass(frozen=True)
class LlamaModelMeta:
    """llama.cpp model metadata"""

    vocab_type: LlamaVocabType
    """Vocabulary type"""
    n_vocab: int
    """Vocabulary length"""
    n_ctx_size: int
    """Context size the model was trained on"""
    n_embd: int
    """Embeddings output length"""
    n_params: int
    """Number of parameters"""
    size: int
    """Size"""

    @staticmethod
    def _from_llama_cpp_response(response: dict[str, Any]) -> LlamaModelMeta:
        """hard-coded converstion from JSON to LlamaModelMeta"""
        return LlamaModelMeta(
            vocab_type=LlamaVocabType(response["vocab_type"]),
            n_vocab=response["n_vocab"],
            n_ctx_size=response["n_ctx_train"],
            n_embd=response["n_embd"],
            n_params=response["n_params"],
            size=response["size"],
        )


@dataclass(frozen=True)
class LlamaModel:
    """llama.cpp model"""

    id: str
    """ID (model's alias)"""
    created: datetime
    """Creation date"""
    meta: LlamaModelMeta
    """Metadata"""

    @staticmethod
    def _from_llama_cpp_response(response: dict[str, Any]) -> LlamaModel:
        """hard-coded converstion from JSON to LlamaModel"""
        return LlamaModel(
            id=response["id"],
            created=datetime.fromtimestamp(response["created"]),
            meta=LlamaModelMeta._from_llama_cpp_response(response["meta"]),
        )


@dataclass
class LlamaCompletionRequest:
    """llama.cpp completion request.
    Values left `None` will fallback to server defaults."""

    prompt: str | list[int]
    """prompt to complete, must be a string or a list of tokens"""
    n_predict: int | None = None
    """Maximum amount of tokens to predict."""
    top_k: int | None = None
    """Top-K sampling limits the number of tokens considered during prediction to specified value."""
    top_p: float | None = None
    """Top-P sampling limits the number of tokens considered during prediction based on their cumulative probability."""
    min_p: float | None = None
    """Min-P defines the minimum probability of token to be considered during prediction."""
    xtc_probability: float | None = None
    """This parameter tweaks the chance of XTC sampling happening."""
    xtc_threshold: float | None = None
    """XTC removes tokens with probability above specified threshold, except least probable one of them."""
    tfs_z: float | None = None
    """Tail-free sampling removes the tokens with less-than-desired second derivative of it's probability.
    This parameter defines the probability in (0, 1] range, where 1 == TFS disabled."""
    typical_p: float | None = None
    """Locally typical sampling can increase diversity of the text without major coherence degradation by choosing tokens that are typical or expected based on the context.
    This parameter defines probability in range (0, 1], where 1 == locally typical sampling disabled."""
    temperature: float | None = None
    """Temperature controls the probability distribution of tokens selected by LLM.
    Temperature shouldn't be below 0."""
    dynatemp_range: float | None = None
    """Dynamic temperature range, if non-zero, defines the range of temperature used during token prediction.
    Final temperature will be applied based on tokens entropy."""
    dynatemp_exponent: float | None = None
    """Dynamic temperature exponent, 1 by default"""
    repeat_last_n: int | None = None
    """Amount of last tokens to penalize for repetition. Setting this to 0 disabled penalization, and -1 penalizes the whole context."""
    repeat_penalty: float | None = None
    """Penalty for repeating the tokens in generated text."""
    presence_penalty: float | None = None
    """Penalty for re-using the tokens that are already present in generated text. 0 == presence penalty disabled."""
    frequency_penalty: float | None = None
    """Penalty applied for re-using the tokens that are already present in generated text, based on the frequency of their appearance. 0 == frequency penalty disabled."""
    mirostat: int | None = None
    """Mirostat type, 1 - Mirostat, 2 - Mirostat 2.0, 0 - disabled.
    ENABLING MIROSTAT DISABLES OTHER SAMPLERS!"""
    mirostat_tau: float | None = None
    """Mirostat target entropy, desired perplexity for generated text."""
    mirostat_eta: float | None = None
    """Mirostat learning rate."""
    penalize_nl: bool | None = None
    """Penalize newline tokens?"""
    n_keep: int | None = None
    """Amount of tokens to keep from initial prompt when context is filled and it shifts."""
    n_discard: int | None = None
    """Number of tokens after n_keep that may be discarded when shifront context, 0 default to half."""
    seed: int | None = None
    """Seed used for RNG"""
    min_keep: int | None = None
    """If greater than 0, forces the sampler to return at least min_keep tokens."""
    predition_time_limit_ms: int | None = None
    """Time limit for prediction (text-generation) phase in milliseconds.
    Starts counting when first token is generated.
    If <= 0, timeout is disabled."""
    ignore_eos: bool | None = None
    """Ignore end-of-sentence token?"""
    stop: list[str] | None = None
    """List of strings stopping the generation."""
    samplers: list[str] | None = None
    """List of used samplers in order."""
    grammar: str | None = None
    """Custom, optional BNF-like grammar to constrain sampling."""

    def _to_llama_cpp_request_json(self) -> dict[str, Any]:
        request_dict = {
            "prompt": self.prompt,
            "n_predict": self.n_predict,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "xtc_probability": self.xtc_probability,
            "xtc_threshold": self.xtc_threshold,
            "tfs_z": self.tfs_z,
            "typical_p": self.typical_p,
            "temperature": self.temperature,
            "dynatemp_range": self.dynatemp_range,
            "dynatemp_exponent": self.dynatemp_exponent,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "penalize_nl": self.penalize_nl,
            "n_keep": self.n_keep,
            "n_discard": self.n_discard,
            "seed": self.seed,
            "min_keep": self.min_keep,
            "t_max_predict_ms": self.predition_time_limit_ms,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
            "samplers": self.samplers,
            "grammar": self.grammar,
        }
        return {k: v for k, v in request_dict.items() if v is not None}


@dataclass(frozen=True)
class LlamaTimings:
    """llama.cpp timings"""

    predicted_ms: float
    """Time taken to predict tokens in milliseconds"""
    predicted_n: int
    """Number of tokens predicted"""
    predicted_per_second: float
    """Prediction rate in tokens per second"""
    predicted_per_token_ms: float
    """Time taken to predict each token in milliseconds"""
    prompt_ms: float
    """Time taken to process the prompt in milliseconds"""
    prompt_n: int
    """Number of tokens in the prompt"""
    prompt_per_second: float
    """Prompt processing rate in tokens per second"""
    prompt_per_token_ms: float
    """Time taken to process each token in the prompt in milliseconds"""

    @staticmethod
    def _from_llama_cpp_response(response: dict[str, Any]) -> LlamaTimings:
        """hard-coded converstion from JSON to LlamaTimings"""
        return LlamaTimings(
            predicted_ms=response["predicted_ms"],
            predicted_n=response["predicted_n"],
            predicted_per_second=response["predicted_per_second"],
            predicted_per_token_ms=response["predicted_per_token_ms"],
            prompt_ms=response["prompt_ms"],
            prompt_n=response["prompt_n"],
            prompt_per_second=response["prompt_per_second"],
            prompt_per_token_ms=response["prompt_per_token_ms"],
        )


@dataclass(frozen=True)
class LlamaCompletionResponse:
    """llama.cpp completion response"""

    content: str
    """Generated text"""
    generation_settings: LlamaGenerationSettings | None
    """Generation settings used for the completion"""
    has_new_line: bool | None
    """Whether the response has a new line at the end"""
    id_slot: int
    """ID of the slot used for the completion"""
    index: int
    model: str | None
    """Model used for the completion"""
    prompt: str | None
    """Prompt used for the completion"""
    stop: bool
    """Whether the completion was stopped"""
    stopped_eos: bool | None
    """Whether the completion was stopped due to reaching the end of sentence"""
    stopped_limit: bool | None
    """Whether the completion was stopped due to reaching the token limit"""
    stopped_word: bool | None
    """Whether the completion was stopped due to reaching a stopping word"""
    stopping_word: str | None
    """Stopping word that caused the completion to stop"""
    timings: LlamaTimings | None
    """Completion timings"""
    tokens_cached: int | None
    """Number of tokens cached"""
    tokens_evaluated: int | None
    """Number of tokens evaluated"""
    tokens_predicted: int | None
    """Number of tokens predicted"""
    truncated: bool | None
    """Whether the completion was truncated"""

    @staticmethod
    def _from_llama_cpp_response(response: dict[str, Any]) -> LlamaCompletionResponse:
        """hard-coded converstion from JSON to LlamaCompletionResponse"""
        generation_settings = None
        if response_gen_settings := response.get("generation_settings", None):
            generation_settings = LlamaGenerationSettings._from_llama_cpp_response(response_gen_settings)

        timings = None
        if response_timings := response.get("timings", None):
            timings = LlamaTimings._from_llama_cpp_response(response_timings)

        return LlamaCompletionResponse(
            content=response["content"],
            generation_settings=generation_settings,
            has_new_line=response.get("has_new_line", None),
            id_slot=response["id_slot"],
            index=response["index"],
            model=response.get("model", None),
            prompt=response.get("prompt", None),
            stop=response["stop"],
            stopped_eos=response.get("stopped_eos", None),
            stopped_limit=response.get("stopped_limit", None),
            stopped_word=response.get("stopped_word", None),
            stopping_word=response.get("stopping_word", None),
            timings=timings,
            tokens_cached=response.get("tokens_cached", None),
            tokens_evaluated=response.get("tokens_evaluated", None),
            tokens_predicted=response.get("tokens_predicted", None),
            truncated=response.get("truncated", None),
        )


class LlamaException(RuntimeError):
    """llama.cpp base exception, should be subclassed if more details
    useful to the user can be provided."""

    def __init__(self, details: str, *args: object) -> None:
        self.details = details
        super().__init__(*args)

    def __str__(self) -> str:
        return f"LlamaException: {self.details}"


class LlamaUrlException(LlamaException):
    """Exception thrown in case of invalid/unknown URL of llama.cpp server"""

    def __init__(self, details: str, host: str | None, port: int | None, *args: object) -> None:
        self.host = host
        self.port = port
        super().__init__(f"{details} (server's host:port -> {host}:{port})", *args)


def _make_llama_server_url(host: str | None, port: int | None) -> str:
    """creates llama.cpp server URL out of host/port, if not provided - will
    try to fetch them from environment. Validates the port range."""
    if host is None:
        if env_host := os.getenv("LLAMA_ARG_HOST"):
            host = env_host
        else:
            raise LlamaUrlException("Unknown llama.cpp server host!", host, port)

    if port is None:
        if env_port := os.getenv("LLAMA_ARG_PORT"):
            port = int(env_port)
        else:
            raise LlamaUrlException("Unknown llama.cpp server port!", host, port)

    if port <= 0 or port > 65535:
        raise LlamaUrlException("Invalid llama.cpp server port!", host, port)

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
) -> LlamaProps:
    """Returns server properties"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/props", timeout=timeout).json()
    return LlamaProps._from_llama_cpp_response(response)


def models(
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> list[LlamaModel]:
    """Returns list of currently loaded models"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/models", timeout=timeout).json()
    return [LlamaModel._from_llama_cpp_response(model) for model in response["data"]]


def complete(
    request: LlamaCompletionRequest,
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> LlamaCompletionResponse:
    """Request completion from llama.cpp server.
    Blocks until response is received, so it may take a while.
    If you'd prefer to fetch chunks of the response, use `streamed_complete` function."""
    server_url = _make_llama_server_url(server_host, server_port)
    request_dict = request._to_llama_cpp_request_json()
    # non-streaming function
    request_dict["stream"] = False
    # we don't support n_probs in requests yet
    request_dict["n_probs"] = 0
    request_json = json.dumps(request_dict)

    response = httpx.post(
        f"{server_url}/completions",
        content=request_json,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    ).json()
    return LlamaCompletionResponse._from_llama_cpp_response(response)


async def streamed_complete(
    request: LlamaCompletionRequest,
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> AsyncIterator[LlamaCompletionResponse]:
    """Request completion from llama.cpp server.
    Yields the response in chunks.
    If you'd prefer to fetch chunks of the response, use `streamed_complete` function."""
    server_url = _make_llama_server_url(server_host, server_port)
    request_dict = request._to_llama_cpp_request_json()
    # streaming function
    request_dict["stream"] = True
    # we don't support n_probs in requests yet
    request_dict["n_probs"] = 0
    request_json = json.dumps(request_dict)

    with httpx.stream(
        "POST",
        f"{server_url}/completions",
        content=request_json,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    ) as response:
        for chunk in response.iter_lines():
            if len(chunk) > 0:
                chunk_data = chunk.removeprefix("data: ")
                response_json = json.loads(chunk_data)
                yield LlamaCompletionResponse._from_llama_cpp_response(response_json)


def tokenize(
    message: str,
    add_special_tokens: bool = False,
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> list[int]:
    """Returns a list of tokens corresponding to tokenized message"""
    server_url = _make_llama_server_url(server_host, server_port)
    request = json.dumps(
        {
            "content": message,
            "add_special": add_special_tokens,
            "with_pieces": False,
        }
    )

    response = httpx.post(
        f"{server_url}/tokenize",
        content=request,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    ).json()

    return cast(list[int], response["tokens"])


def detokenize(
    tokens: list[int],
    server_host: str | None = None,
    server_port: int | None = None,
    timeout: float = 60.0,
) -> str:
    """Returns detokenized message"""
    server_url = _make_llama_server_url(server_host, server_port)
    request = json.dumps(
        {
            "tokens": tokens,
        }
    )

    response = httpx.post(
        f"{server_url}/detokenize",
        content=request,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    ).json()

    return cast(str, response["content"])
