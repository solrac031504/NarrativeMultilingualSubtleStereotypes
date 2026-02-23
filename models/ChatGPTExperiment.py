from openai import OpenAI
from models.BaseExperiment import BaseExperiment

class ChatGPTExperiment(BaseExperiment):
    def _provider_name(self):
        return "ChatGPT"
    
    def _build_client(self):
        return OpenAI(api_key=self.api_key)
    
    def _call_model(self, model, system_prompt, user_content, temperature, max_tokens) -> str:
        message = self.client.responses.create(
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            instructions=system_prompt,
            input=[
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        return message.output_text