from __future__ import annotations

import json
import os
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
        return json.dumps(_remove_none_values(asdict(self)))


class FromJson:
    @classmethod
    def from_json(cls, data: dict) -> Self:
        return cls(**data)


# request data types

type LlamaPrompt = str | list[str] | list[int]


@dataclass
class LlamaCompletionRequest(ToJson):
    prompt: LlamaPrompt
    system_prompt: str | None = None
    stream: bool | None = None
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


@dataclass(frozen=True)
class LlamaGenerationSettings(FromJson):
    n_ctx: int
    n_predict: int
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


@dataclass(frozen=True)
class LlamaCompletionResponse(FromJson):
    content: str
    id_slot: int
    stop: bool
    model: str
    tokens_predicted: int
    tokens_evaluated: int
    generation_settings: LlamaGenerationSettings
    prompt: str
    truncated: bool
    stopped_eos: bool
    stopped_word: bool
    stopped_limit: bool
    stopping_word: str
    tokens_cached: int
    timings: LlamaTimings


class UnreasonableLlama:
    def __init__(
        self,
        server_url: str = "",
        system_prompt: str = "",
        request_timeout: int = 10000,
    ):
        if server_url == "":
            server_url = os.getenv("LLAMA_CPP_SERVER_URL")
            if server_url is None:
                raise RuntimeError("Missing llama.cpp server URL!")

        self.system_prompt = system_prompt

        self.client = httpx.Client(
            headers={"Content-Type": "application/json"},
            timeout=request_timeout,
            base_url=server_url,
        )

    def close(self):
        self.client.close()

    def get_health(self, include_slots: bool = False):
        response = self.client.get(
            "health", params="include_slots" if include_slots else ""
        ).json()
        return LlamaHealth.from_json(response)

    def get_completion(self, request: LlamaCompletionRequest):
        response = self.client.post("completions", data=request.to_json()).json()
        return LlamaCompletionResponse.from_json(response)
