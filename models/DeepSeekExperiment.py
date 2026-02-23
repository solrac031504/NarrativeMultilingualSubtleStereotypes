from openai import OpenAI
from models.BaseExperiment import BaseExperiment

class DeepSeekExperiment(BaseExperiment):
    def _provider_name(self):
        return "DeepSeek"
    
    def _build_client(self):
        return OpenAI(
            api_key=self.api_key, 
            base_url="https://api.deepseek.com"
          )
    
    def _call_model(self, model, system, user_content, temperature, max_tokens) -> str:
        message = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        return message.choices[0].message.content