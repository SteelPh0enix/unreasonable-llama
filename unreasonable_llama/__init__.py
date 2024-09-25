from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Self

import httpx


class LlamaException(RuntimeError):
    def __init__(self, details: str, *args: object) -> None:
        self.details = details
        super().__init__(*args)

    def __str__(self) -> str:
        return f"LlamaException: {self.details}"


def _remove_none_values(dictionary: dict) -> dict:
    """Recursively removes None values from a Python dictionary.

    Args:
      dictionary: The dictionary to remove None values from.

    Returns:
      A new dictionary with all None values removed.
    """

    new_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            new_dict[key] = _remove_none_values(value)
        elif value is not None:
            new_dict[key] = value
    return new_dict


class ToJson:
    def to_json(self) -> str:
        return json.dumps(_remove_none_values(asdict(self)))  # type: ignore


class ToDict:
    def to_dict(self) -> dict:
        return _remove_none_values(asdict(self))  # type: ignore


class FromJson:
    @classmethod
    def from_json(cls, data: dict) -> Self:
        return cls(**data)


# request data types

type LlamaPrompt = str | list[str] | list[int]


@dataclass
class LlamaCompletionRequest(ToJson, ToDict):
    prompt: LlamaPrompt
    cache_prompt: bool | None = None
    dynatemp_exponent: float | None = None
    dynatemp_range: float | None = None
    frequency_penalty: float | None = None
    grammar: object | None = None  # todo: type this correctly
    id_slot: int | None = None
    ignore_eos: bool | None = None
    image_data: list | None = None
    json_schema: dict[str, object] | list[str] | None = None
    logit_bias: list | None = None  # todo: type this correctly
    min_keep: int | None = None
    min_p: float | None = None
    mirostat: int | None = None
    mirostat_eta: float | None = None
    mirostat_tau: float | None = None
    n_keep: int | None = None
    n_predict: int | None = None
    n_probs: int | None = None
    penalize_nl: bool | None = None
    penalty_prompt: LlamaPrompt | None = None
    presence_penalty: float | None = None
    repeat_last_n: int | None = None
    repeat_penalty: float | None = None
    samplers: list[str] | None = None
    seed: int | None = None
    stop: list[str] | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    tfs_z: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    typical_p: float | None = None


@dataclass
class LlamaTokenizeRequest(ToJson):
    content: str
    add_special: bool = False
    with_pieces: bool = False


# response data types


@dataclass(frozen=True)
class LlamaNextToken(FromJson):
    has_next_token: bool
    n_decoded: int
    n_remain: int
    stopped_eos: bool
    stopped_limit: bool
    stopped_word: bool
    stopping_word: str


@dataclass()
class LlamaSlot(FromJson):
    dynatemp_exponent: float
    dynatemp_range: float
    frequency_penalty: float
    grammar: str
    id: int
    id_task: int
    ignore_eos: bool
    max_tokens: int
    min_keep: int
    min_p: float
    mirostat: int
    mirostat_eta: float
    mirostat_tau: float
    model: str
    n_ctx: int
    n_discard: int
    n_keep: int
    n_predict: int
    n_probs: int
    next_token: LlamaNextToken
    penalize_nl: bool
    presence_penalty: float
    prompt: str
    repeat_last_n: int
    repeat_penalty: float
    samplers: list[str]
    seed: int
    seed_cur: int
    state: int
    stop: list[str]
    stream: bool
    temperature: float
    tfs_z: float
    top_k: int
    top_p: float
    typical_p: float

    @classmethod
    def from_json(cls, data: dict) -> Self:
        response = super().from_json(data)
        response.next_token = LlamaNextToken.from_json(data["next_token"])
        return response


@dataclass(frozen=True)
class LlamaGenerationSettings(FromJson):
    dynatemp_exponent: float
    dynatemp_range: float
    frequency_penalty: float
    grammar: str
    ignore_eos: bool
    max_tokens: int
    min_keep: int
    min_p: float
    mirostat: int
    mirostat_eta: float
    mirostat_tau: float
    model: str
    n_ctx: int
    n_discard: int
    n_keep: int
    n_predict: int
    n_probs: int
    penalize_nl: bool
    presence_penalty: float
    repeat_last_n: int
    repeat_penalty: float
    samplers: list[str]
    seed: int
    seed_cur: int
    stop: list[str]
    stream: bool
    temperature: float
    tfs_z: float
    top_k: int
    top_p: float
    typical_p: float


@dataclass()
class LlamaProps(FromJson):
    system_prompt: str
    default_generation_settings: LlamaGenerationSettings
    total_slots: int
    chat_template: str

    @classmethod
    def from_json(cls, data: dict) -> Self:
        response = super().from_json(data)
        response.default_generation_settings = LlamaGenerationSettings.from_json(data["default_generation_settings"])
        return response


@dataclass(frozen=True)
class LlamaTimings(FromJson):
    predicted_ms: float
    predicted_n: int
    predicted_per_second: float
    predicted_per_token_ms: float
    prompt_ms: float
    prompt_n: int
    prompt_per_second: float
    prompt_per_token_ms: float


@dataclass
class LlamaCompletionResponse(FromJson):
    content: str
    id_slot: int
    index: int
    stop: bool
    generation_settings: LlamaGenerationSettings | None = None
    model: str | None = None
    multimodal: str | None = None
    prompt: str | None = None
    seed_cur: int | None = None
    stopped_eos: bool | None = None
    stopped_limit: bool | None = None
    stopped_word: bool | None = None
    stopping_word: str | None = None
    timings: LlamaTimings | None = None
    tokens_cached: int | None = None
    tokens_evaluated: int | None = None
    tokens_predicted: int | None = None
    truncated: bool | None = None

    @classmethod
    def from_json(cls, data: dict) -> Self:
        if "error" in data:
            raise LlamaException(data["error"])

        if "timings" in data:
            data["timings"] = LlamaTimings.from_json(data["timings"])

        response = super().from_json(data)

        # add missing `generation_settings` fields
        if "generation_settings" in data:
            response.generation_settings = LlamaGenerationSettings.from_json(data["generation_settings"])

        return response


type LlamaTokens = list[int] | list[dict[str, int | str]] | list[dict[str, int | list[int]]]


class UnreasonableLlama:
    def __init__(
        self,
        server_url: str = "",
        request_timeout: int = 10000,
    ):
        if server_url == "":
            if server_url_from_env := os.getenv("LLAMA_CPP_SERVER_URL"):
                server_url = server_url_from_env
            else:
                raise LlamaException("Missing llama.cpp server URL!")

        self.server_url = server_url
        self.client = httpx.Client(
            headers={"Content-Type": "application/json"},
            timeout=request_timeout,
            base_url=server_url,
        )

    def close(self) -> None:
        self.client.close()

    def is_alive(self) -> bool:
        response = self.client.get("health").json()
        return "status" in response and "error" not in response and response["status"] == "ok"

    def slots(self) -> list[LlamaSlot]:
        response = self.client.get("slots").json()
        return [LlamaSlot.from_json(slot) for slot in response]

    def props(self) -> LlamaProps:
        response = self.client.get("props")
        return LlamaProps.from_json(response.json())

    def get_completion(self, request: LlamaCompletionRequest) -> LlamaCompletionResponse:
        request_dict = request.to_dict()
        request_dict["stream"] = False
        request_json = json.dumps(request_dict)

        response = self.client.post("completions", data=request_json)  # type: ignore
        return LlamaCompletionResponse.from_json(response.json())

    async def get_streamed_completion(self, request: LlamaCompletionRequest) -> AsyncIterator[LlamaCompletionResponse]:
        request_dict = request.to_dict()
        request_dict["stream"] = True
        request_json = json.dumps(request_dict)

        with self.client.stream("POST", "completions", data=request_json) as response:  # type: ignore
            for chunk in response.iter_lines():
                if chunk.startswith("data: "):
                    chunk = chunk.removeprefix("data: ")
                    yield LlamaCompletionResponse.from_json(json.loads(chunk))

    def tokenize(self, request: LlamaTokenizeRequest) -> LlamaTokens:
        request_json = request.to_json()
        response = self.client.post("tokenize", data=request_json)  # type: ignore
        return response.json()["tokens"]  # type: ignore

    def detokenize(self, tokens: LlamaTokens) -> str:
        request_json = json.dumps({"tokens": tokens})
        response = self.client.post("detokenize", data=request_json)  # type: ignore
        return response.json()["content"]  # type: ignore
