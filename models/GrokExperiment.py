from xai_sdk import Client
from xai_sdk.chat import user, system
from models.BaseExperiment import BaseExperiment

class GrokExperiment(BaseExperiment):
    def _provider_name(self):
        return "Grok"
    
    def _build_client(self):
        return Client(api_key=self.api_key)
    
    def _call_model(self, model, system_prompt, user_content, temperature, max_tokens) -> str:
        chat = self.client.chat.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

        chat.append(system(system_prompt))
        chat.append(user(user_content))
        return chat.sample().content