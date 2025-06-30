import argparse

# import asyncio
import sys
from pprint import pprint

import unreasonable_llama as llama

parser = argparse.ArgumentParser()
parser.add_argument("--test-completion", action="store_true", help="Run completion tests")
args = parser.parse_args()

if llama.is_alive():
    print("Server is alive!")
else:
    print("Server is NOT alive!")
    sys.exit(1)

props = llama.props()
pprint(props)

example_conversation = [
    llama.ChatMessage(role="system", content="You are a helpful assistant."),
    llama.ChatMessage(role="user", content="What's 2+2?"),
    llama.ChatMessage(role="assistant", content="It's 4."),
    llama.ChatMessage(role="user", content="Thanks, now tell me what's 12*34."),
]

templated_conversation = llama.apply_template(example_conversation)
pprint(templated_conversation)

# models = llama.models()
# pprint(models)
#
# if args.test_completion:
#     example_small_completion = llama.LlamaCompletionRequest(
#         prompt="Here's a random fact:",
#         n_predict=100,
#     )
#
#     print("Requesting blocking completion for")
#     pprint(example_small_completion)
#     sync_completion = llama.complete(example_small_completion)
#     pprint(sync_completion)
#
#     async def perform_async_completion(request: llama.LlamaCompletionRequest) -> None:
#         async for chunk in llama.streamed_complete(request):
#             pprint(chunk)
#
#     print("Requesting asynchronous completion for")
#     pprint(example_small_completion)
#     asyncio.run(perform_async_completion(example_small_completion))
#
# text_to_tokenize = "Hi, this is a sample sentence that's gonna be tokenized!"
# print(f"Tokenizing '{text_to_tokenize}'")
# tokens = llama.tokenize(text_to_tokenize)
# print(f"Tokens: {tokens}")
# detokenized_text = llama.detokenize(tokens)
# print(f"Detokenized text: '{detokenized_text}'")
