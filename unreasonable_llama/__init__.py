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
    """Amount of tokens to predict"""
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
    Not supported yet. Keep at 0."""
    min_keep: int
    """If greater than 0, forces the sampler to return at least min_keep tokens."""
    grammar: str
    """Custom, optional BNF-like grammar to constrain sampling."""
    samplers: list[str]
    """List of used samplers in order."""

    @staticmethod
    def from_llama_cpp_response(response: dict) -> LlamaGenerationSettings:
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
    def from_llama_cpp_response(response: dict) -> LlamaProps:
        """hard-coded converstion from JSON to LlamaProps"""
        generation_settings = LlamaGenerationSettings.from_llama_cpp_response(response["default_generation_settings"])
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
    def from_llama_cpp_response(response: dict) -> LlamaModelMeta:
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
    def from_llama_cpp_response(response: dict) -> LlamaModel:
        """hard-coded converstion from JSON to LlamaModel"""
        return LlamaModel(
            id=response["id"],
            created=datetime.fromtimestamp(response["created"]),
            meta=LlamaModelMeta.from_llama_cpp_response(response["meta"]),
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


def health(server_host: str | None = None, server_port: int | None = None) -> bool:
    """Returns `True` if server is alive and ready, `False` if it's not ready"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/health").json()
    return response.get("status", "") == "ok"


def props(server_host: str | None = None, server_port: int | None = None) -> LlamaProps:
    """Returns server properties"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/props").json()
    return LlamaProps.from_llama_cpp_response(response)


def models(server_host: str | None = None, server_port: int | None = None) -> list[LlamaModel]:
    """Returns list of currently loaded models"""
    server_url = _make_llama_server_url(server_host, server_port)
    response = httpx.get(f"{server_url}/models").json()
    return [LlamaModel.from_llama_cpp_response(model) for model in response["data"]]


def complete(prompt: str, server_host: str | None = None, server_port: int | None = None):
    pass
