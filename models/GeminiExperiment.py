from google import genai
from google.genai import types, Client
from models.BaseExperiment import BaseExperiment

class GeminiExperiment(BaseExperiment):
    def _provider_name(self):
        return "Gemini"
    
    def _build_client(self):
        return Client(api_key=self.api_key)
    
    def _call_model(self, model, system_prompt, user_content, temperature, max_tokens) -> str:
        message = self.client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature
            ),
            contents=user_content
        )
        return message.text