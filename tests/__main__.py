import asyncio
import sys

import httpx

from unreasonable_llama import (
    LlamaCompletionRequest,
    LlamaInfillRequest,
    UnreasonableLlama,
)

# This will load the URL from `LLAMA_CPP_SERVER_URL` environmental variable,
# or throw an error if it doesn't exist.
llama = UnreasonableLlama()
# Alternatively, you can do this:
# llama = UnreasonableLlama("http://localhost:8080/")

try:
    # you can use `get_health()` to check server connection and get info about
    # loaded models
    health = llama.get_health(include_slots=True)
    print(f"Currently loaded model: {health.slots[0].model}")
except httpx.ConnectError:
    print(
        f"Unable to connect to llama.cpp server @ {llama.server_url}, verify if it's running under correct host/port?"
    )
    sys.exit(1)


# You can request completion in synchronous or asynchronous way.
# It's up to you to format the prompt for the loaded model, this library
# is as simple as it can possibly be and doesn't provide any high-level
# functions. Therefore, the following test may result in model spewing
# nonsense, but as long as it responds, you can assume it's working correctly.
request = LlamaCompletionRequest(
    prompt="Briefly describe highest mountain on Mars.\n", n_predict=200
)
response = llama.get_completion(request)
print(f"Synchronous response: {response.content}\n")


async def get_async_completion(
    llama: UnreasonableLlama, request: LlamaCompletionRequest
):
    print("Asynchronous response: ", end="", flush=True)
    async for chunk in llama.get_streamed_completion(request):
        # chunk is LlamaCompletionResponse
        print(chunk.content, end="", flush=True)

    print("\n")


request = LlamaCompletionRequest(
    prompt="Briefly describe highest mountain on Earth.\n", n_predict=300
)
asyncio.run(get_async_completion(llama, request))


async def get_infill(llama: UnreasonableLlama, request: LlamaInfillRequest):
    print("Requesting infill for:")
    print(request.input_prefix)
    async for chunk in llama.get_infill(request):
        print(chunk.content, end="", flush=True)


infill_request = LlamaInfillRequest(
    input_prefix="def calculate_square_root(number: float) -> float:",
    input_suffix="",
)
asyncio.run(get_infill(llama, infill_request))

# closing is recommended
llama.close()
