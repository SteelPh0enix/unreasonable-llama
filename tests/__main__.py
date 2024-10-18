import argparse
import asyncio
from pprint import pprint

import unreasonable_llama as llama

parser = argparse.ArgumentParser()
parser.add_argument("--test-completion", action="store_true", help="Run completion tests")
args = parser.parse_args()

server_is_alive = llama.health()
if server_is_alive:
    print("Server is alive!")
else:
    print("Server is NOT alive!")
    exit(1)

props = llama.props()
pprint(props)

models = llama.models()
pprint(models)

if args.test_completion:
    example_small_completion = llama.LlamaCompletionRequest(
        prompt="Here's a random fact:",
        n_predict=100,
    )

    print("Requesting blocking completion for")
    pprint(example_small_completion)
    sync_completion = llama.complete(example_small_completion)
    pprint(sync_completion)

    async def perform_async_completion(request: llama.LlamaCompletionRequest):
        async for chunk in llama.streamed_complete(request):
            pprint(chunk)

    print("Requesting asynchronous completion for")
    pprint(example_small_completion)
    asyncio.run(perform_async_completion(example_small_completion))

text_to_tokenize = "Hi, this is a sample sentence that's gonna be tokenized!"
print(f"Tokenizing '{text_to_tokenize}'")
tokens = llama.tokenize(text_to_tokenize)
print(f"Tokens: {tokens}")
detokenized_text = llama.detokenize(tokens)
print(f"Detokenized text: '{detokenized_text}'")
