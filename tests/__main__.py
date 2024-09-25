import asyncio
import sys
from typing import cast
from pprint import pprint

import httpx

from unreasonable_llama import (
    LlamaCompletionRequest,
    LlamaTokenizeRequest,
    UnreasonableLlama,
)

print("==================== INIT ====================")

# This will load the URL from `LLAMA_CPP_SERVER_URL` environmental variable,
# or throw an error if it doesn't exist.
llama = UnreasonableLlama()
# Alternatively, you can do this:
# llama = UnreasonableLlama("http://localhost:8080/")

try:
    # you can use `is_alive()` to check server connection
    if llama.is_alive():
        print("llama server is alive!")
        print("slots:")
        for slot in llama.slots():
            pprint(slot)

        props = llama.props()
        print(f"Loaded model: {props.default_generation_settings.model}")
        print("\nprops:")
        pprint(props)
    else:
        print("llama server is not alive, quitting!")
        sys.exit(1)
except httpx.ConnectError:
    print(
        f"Unable to connect to llama.cpp server @ {llama.server_url}, verify if it's running under correct host/port?"
    )
    sys.exit(1)

print("==================== COMPLETION ====================")

# You can request completion in synchronous or asynchronous way.
# It's up to you to format the prompt for the loaded model, this library
# is as simple as it can possibly be and doesn't provide any high-level
# functions. Therefore, the following test may result in model spewing
# nonsense, but as long as it responds, you can assume it's working correctly.
request = LlamaCompletionRequest(prompt="Briefly describe highest mountain on Mars.\n", n_predict=100)
response = llama.get_completion(request)
print(f"Synchronous response: {response.content}\n")


async def get_async_completion(llama: UnreasonableLlama, request: LlamaCompletionRequest):
    print("Asynchronous response: ", end="", flush=True)
    async for chunk in llama.get_streamed_completion(request):
        # chunk is LlamaCompletionResponse
        print(chunk.content, end="", flush=True)

    print("\n")


request = LlamaCompletionRequest(prompt="Briefly describe highest mountain on Earth.\n", n_predict=100)
asyncio.run(get_async_completion(llama, request))

# test tokenization and detokenization
print("==================== (DE)TOKENIZATION ====================")

text_to_tokenize = "Hello world, pls tokenize me!"
tokenize_request = LlamaTokenizeRequest(text_to_tokenize)
tokenized_text = cast(list[int], llama.tokenize(tokenize_request))
print(f"Tokenized text: {tokenized_text}")
detokenized_text = llama.detokenize(tokenized_text)
print(f"Detokenized text: {detokenized_text}")

# closing is recommended
llama.close()
