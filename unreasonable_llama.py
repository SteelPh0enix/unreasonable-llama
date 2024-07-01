from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import httpx


def _remove_none_values(dictionary):
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


@dataclass
class LlamaHealth(ToJson):
    status: str
    slots_idle: int | None = None
    slots_processing: int | None = None


type LlamaPrompt = str | list[str] | list[int]


@dataclass
class LlamaSystemPrompt(ToJson):
    prompt: LlamaPrompt
    anti_prompt: str
    assistant_name: str


@dataclass
class LlamaCompletionRequest(ToJson):
    prompt: LlamaPrompt
    system_prompt: LlamaSystemPrompt | None = None
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

    def get_health(self):
        response = self.client.get("health").json()
        return LlamaHealth(**response)

    def get_completion(self, request: LlamaCompletionRequest):
        response = self.client.post("completions", data=request.to_json()).json()
        return json.dumps(response, indent=2)
