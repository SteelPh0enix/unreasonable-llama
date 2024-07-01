from unreasonable_llama import UnreasonableLlama, LlamaCompletionRequest

request = LlamaCompletionRequest(prompt="How high is the mount everest?")

llama = UnreasonableLlama()
print(llama.get_health(True))
print(llama.get_completion(request))
llama.close()
