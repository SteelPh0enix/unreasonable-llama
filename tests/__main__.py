from pprint import pprint

import unreasonable_llama as llama

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

example_small_completion = llama.LlamaCompletionRequest(
    prompt="Here's a random fact:",
    n_predict=100,
)

print("Requesting completion for")
pprint(example_small_completion)
completion = llama.complete(example_small_completion)
pprint(completion)
