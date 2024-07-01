from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import httpx
import json


@dataclass
class LlamaHealth:
    status: str
    slots_idle: int | None = None
    slots_processing: int | None = None


@dataclass
class LlamaSystemPrompt:
    prompt: str
    anti_prompt: str
    assistant_name: str


@dataclass
class LlamaCompletionRequest:
    prompt: str | list[str] | list[int]
    system_prompt: LlamaSystemPrompt | None = None
    stream: bool | None = None
    stop: list[str] | None = None
    cache_prompt: bool | None = None


def _filter_nones_from_dict(input: dict) -> dict:
    return {k: v for k, v in input.items() if v is not None}


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
        request_dict = _filter_nones_from_dict(asdict(request))
        response = self.client.post("completions", data=json.dumps(request_dict)).json()
        return response
