from unreasonable_llama import UnreasonableLlama, LlamaCompletionRequest
import asyncio

request = LlamaCompletionRequest(prompt="How high is the mount everest?")

llama = UnreasonableLlama()
print(llama.get_health(True))


async def test_streaming():
    async for response in llama.get_streamed_completion(request):
        print(response)


asyncio.run(test_streaming())
llama.close()
