from deepeval.models import DeepEvalBaseLLM
from gigachat import GigaChat
import asyncio

class CustomGigaChat(DeepEvalBaseLLM):
    def __init__(self, credentials: str = None, model: str = "GigaChat-Pro"):
        # credentials — authorization data (token or key
        self.model_name = model
        self.giga = GigaChat(
            credentials=credentials,
            model=model,
            verify_ssl_certs=False,  # for tests
            timeout=60
        )
    
    def load_model(self):
        return self.giga
    
    def generate(self, prompt: str) -> str:
        giga = self.load_model()
        response = giga.chat(prompt)
        return response.choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        # Asynchronous version (can be done via a separate thread)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)
    
    def get_model_name(self):
        return f"GigaChat-{self.model_name}"