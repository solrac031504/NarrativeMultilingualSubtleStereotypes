from anthropic import Anthropic
from models.BaseExperiment import BaseExperiment

class ClaudeExperiment(BaseExperiment):
    def _provider_name(self):
        return "Claude"
    
    def _build_client(self):
        return Anthropic(api_key=self.api_key)
    
    def _call_model(self, model, system_prompt, user_content, temperature, max_tokens) -> str:
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        return message.content[0].text.strip()