from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from typing import Self

import httpx


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
    system_prompt: str | None = None
    stop: list[str] | None = None
    cache_prompt: bool | None = None
    temperature: float | None = None
    dynatemp_range: float | None = None
    dynatemp_exponent: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    n_predict: int | None = None
    n_keep: int | None = None
    tfs_z: float | None = None
    typical_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    penalize_nl: bool | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_prompt: LlamaPrompt | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    grammar: object | None = None  # todo: type this correctly
    json_schema: dict[str, object] | list[str] | None = None
    seed: int | None = None
    ignore_eos: bool | None = None
    logit_bias: list | None = None  # todo: type this correctly
    n_probs: int | None = None
    min_keep: int | None = None
    image_data: list | None = None
    id_slot: int | None = None
    cache_prompt: bool | None = None
    samplers: list[str] | None = None


@dataclass
class LlamaInfillRequest(ToJson, ToDict):
    input_prefix: str
    input_suffix: str
    system_prompt: str | None = None
    stop: list[str] | None = None
    cache_prompt: bool | None = None
    temperature: float | None = None
    dynatemp_range: float | None = None
    dynatemp_exponent: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    n_predict: int | None = None
    n_keep: int | None = None
    tfs_z: float | None = None
    typical_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    penalize_nl: bool | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_prompt: LlamaPrompt | None = None
    mirostat: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    grammar: object | None = None  # todo: type this correctly
    json_schema: dict[str, object] | list[str] | None = None
    seed: int | None = None
    ignore_eos: bool | None = None
    logit_bias: list | None = None  # todo: type this correctly
    n_probs: int | None = None
    min_keep: int | None = None
    image_data: list | None = None
    id_slot: int | None = None
    cache_prompt: bool | None = None
    samplers: list[str] | None = None


# response data types


@dataclass(frozen=True)
class LlamaNextToken(FromJson):
    has_next_token: bool
    n_remain: int
    n_decoded: int
    stopped_eos: bool
    stopped_word: bool
    stopped_limit: bool
    stopping_word: str


@dataclass(frozen=True)
class LlamaSlot(FromJson):
    n_ctx: int
    n_predict: int
    max_tokens: int
    model: str
    seed: int
    temperature: float
    dynatemp_range: float
    dynatemp_exponent: float
    top_k: int
    top_p: float
    min_p: float
    tfs_z: float
    typical_p: float
    repeat_last_n: int
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    penalty_prompt_tokens: list
    use_penalty_prompt_tokens: bool
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    penalize_nl: bool
    stop: list[str]
    n_keep: int
    n_discard: int
    ignore_eos: bool
    stream: bool
    logit_bias: list
    n_probs: int
    min_keep: int
    grammar: str
    samplers: list[str]
    id: int
    id_task: int
    state: int
    prompt: str
    next_token: LlamaNextToken


@dataclass(frozen=True)
class LlamaHealth(FromJson):
    status: str
    slots_idle: int | None = None
    slots_processing: int | None = None
    slots: list[LlamaSlot] | None = None

    @classmethod
    def from_json(cls, data: dict) -> Self:
        if "slots" in data:
            data["slots"] = [LlamaSlot.from_json(slot) for slot in data["slots"]]
        return super().from_json(data)


@dataclass(frozen=True)
class LlamaGenerationSettings(FromJson):
    n_ctx: int
    n_predict: int
    max_tokens: int
    model: str
    seed: int
    temperature: float
    dynatemp_range: float
    dynatemp_exponent: float
    top_k: int
    top_p: float
    min_p: float
    tfs_z: float
    typical_p: float
    repeat_last_n: int
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    penalty_prompt_tokens: list
    use_penalty_prompt_tokens: bool
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    penalize_nl: bool
    stop: list[str]
    n_keep: int
    n_discard: int
    ignore_eos: bool
    stream: bool
    logit_bias: list
    n_probs: int
    min_keep: int
    grammar: str
    samplers: list[str]


@dataclass(frozen=True)
class LlamaTimings(FromJson):
    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float


type LlamaError = str


@dataclass()
class LlamaCompletionResponse(FromJson):
    content: str
    id_slot: int
    stop: bool
    multimodal: str | None = None
    model: str | None = None
    tokens_predicted: int | None = None
    tokens_evaluated: int | None = None
    generation_settings: LlamaGenerationSettings | None = None
    prompt: str | None = None
    truncated: bool | None = None
    stopped_eos: bool | None = None
    stopped_word: bool | None = None
    stopped_limit: bool | None = None
    stopping_word: str | None = None
    tokens_cached: int | None = None
    timings: LlamaTimings | None = None

    @classmethod
    def from_json(cls, data: dict) -> Self:
        if "error" in data:
            return data["error"]

        if "timings" in data:
            data["timings"] = LlamaTimings.from_json(data["timings"])

        response = super().from_json(data)

        # add missing `generation_settings` fields
        if "generation_settings" in data:
            response.generation_settings = LlamaGenerationSettings.from_json(data["generation_settings"])

        return response


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
                raise RuntimeError("Missing llama.cpp server URL!")

        self.server_url = server_url
        self.client = httpx.Client(
            headers={"Content-Type": "application/json"},
            timeout=request_timeout,
            base_url=server_url,
        )

    def close(self):
        self.client.close()

    def get_health(self, include_slots: bool = False) -> LlamaHealth:
        response = self.client.get("health", params="include_slots" if include_slots else "").json()
        return LlamaHealth.from_json(response)

    def get_completion(self, request: LlamaCompletionRequest) -> LlamaCompletionResponse:
        request_dict = request.to_dict()
        request_dict["stream"] = False
        request_json = json.dumps(request_dict)

        response = self.client.post("completions", data=request_json)  # type: ignore
        return LlamaCompletionResponse.from_json(response.json())

    async def get_streamed_completion(self, request: LlamaCompletionRequest) -> AsyncGenerator:
        request_dict = request.to_dict()
        request_dict["stream"] = True
        request_json = json.dumps(request_dict)

        with self.client.stream("POST", "completions", data=request_json) as response:  # type: ignore
            for chunk in response.iter_lines():
                if chunk.startswith("data: "):
                    chunk = chunk.removeprefix("data: ")
                    yield LlamaCompletionResponse.from_json(json.loads(chunk))

    def get_infill(self, request: LlamaInfillRequest) -> LlamaCompletionResponse:
        request_dict = request.to_dict()
        request_dict["stream"] = False
        request_json = json.dumps(request_dict)

        response = self.client.post("infill", data=request_json)  # type: ignore
        return LlamaCompletionResponse.from_json(response.json())

    async def get_streamed_infill(self, request: LlamaInfillRequest) -> AsyncGenerator:
        request_dict = request.to_dict()
        request_dict["stream"] = True
        request_json = json.dumps(request_dict)

        with self.client.stream("POST", "infill", data=request_json) as response:  # type: ignore
            for chunk in response.iter_lines():
                if chunk.startswith("data: "):
                    chunk = chunk.removeprefix("data: ")
                    yield LlamaCompletionResponse.from_json(json.loads(chunk))
