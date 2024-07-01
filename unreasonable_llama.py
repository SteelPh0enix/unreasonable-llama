from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass
class LlamaHealth:
    status: str
    slots_idle: int | None
    slots_processing: int | None


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
