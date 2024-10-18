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
    * `/completions` (POST) [complete(prompt), streamed_complete(prompt)]
    * `/tokenize` (POST) [tokenize(raw_prompt)]
    * `/detokenize` (POST) [detokenize(tokenized_prompt)]
    * `/slots` (GET) [slots()]

Note: `complete` and `streamed_complete` accept both tokenized and raw prompt.

This librarty uses `httpx`. In case of connection issues, expect
`httpx.ConnectError` to happen.

Authenthication and error handling is not implemented yet.

I develop this library mostly for myself - if you want to see more endpoints
supported, make PRs. Don't forget about pre-commit checks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

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
    tfs_z: float
    """Tail-free sampling removes the tokens with less-than-desired second derivative of it's probability.
    This parameter defines the probability in (0, 1] range, where 1 == TFS disabled."""
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
    def _from_llama_cpp_response(response: dict) -> LlamaGenerationSettings:
        """hard-coded converstion from JSON to LlamaGenerationSettings"""
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
            tfs_z=response["tfs_z"],
            typical_p=response["typical_p"],
            repeat_last_n=response["repeat_last_n"],
            repeat_penalty=response["repeat_penalty"],
            presence_penalty=response["presence_penalty"],
            frequency_penalty=response["frequency_penalty"],
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
    def _from_llama_cpp_response(response: dict) -> LlamaProps:
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
    def _from_llama_cpp_response(response: dict) -> LlamaModelMeta:
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
    def _from_llama_cpp_response(response: dict) -> LlamaModel:
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

    def _to_llama_cpp_request_json(self) -> dict:
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
    def _from_llama_cpp_response(response: dict) -> LlamaTimings:
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
    generation_settings: LlamaGenerationSettings
    """Generation settings used for the completion"""
    has_new_line: bool
    """Whether the response has a new line at the end"""
    id_slot: int
    """ID of the slot used for the completion"""
    index: int
    model: str
    """Model used for the completion"""
    prompt: str
    """Prompt used for the completion"""
    stop: bool
    """Whether the completion was stopped"""
    stopped_eos: bool
    """Whether the completion was stopped due to reaching the end of sentence"""
    stopped_limit: bool
    """Whether the completion was stopped due to reaching the token limit"""
    stopped_word: bool
    """Whether the completion was stopped due to reaching a stopping word"""
    stopping_word: str
    """Stopping word that caused the completion to stop"""
    timings: LlamaTimings
    """Completion timings"""
    tokens_cached: int
    """Number of tokens cached"""
    tokens_evaluated: int
    """Number of tokens evaluated"""
    tokens_predicted: int
    """Number of tokens predicted"""
    truncated: bool
    """Whether the completion was truncated"""

    @staticmethod
    def _from_llama_cpp_response(response: dict) -> LlamaCompletionResponse:
        """hard-coded converstion from JSON to LlamaCompletionResponse"""
        generation_settings = LlamaGenerationSettings._from_llama_cpp_response(response["generation_settings"])
        timings = LlamaTimings._from_llama_cpp_response(response["timings"])
        return LlamaCompletionResponse(
            content=response["content"],
            generation_settings=generation_settings,
            has_new_line=response["has_new_line"],
            id_slot=response["id_slot"],
            index=response["index"],
            model=response["model"],
            prompt=response["prompt"],
            stop=response["stop"],
            stopped_eos=response["stopped_eos"],
            stopped_limit=response["stopped_limit"],
            stopped_word=response["stopped_word"],
            stopping_word=response["stopping_word"],
            timings=timings,
            tokens_cached=response["tokens_cached"],
            tokens_evaluated=response["tokens_evaluated"],
            tokens_predicted=response["tokens_predicted"],
            truncated=response["truncated"],
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
    return response.get("status", "") == "ok"


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
