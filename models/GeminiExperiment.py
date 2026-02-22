import json
from google import genai
from google.genai import types

class GeminiExperiment:
  def __init__(
      self,

      scenario_prompts: dict[str, dict[str, str]],

      api_key: str,

      target_model: str,
      samples_per_prompt: int,
      target_model_temperature: int,
      target_model_max_tokens: int,
      system_prompt: str,

      classifier_model: str,
      classifier_temperature: int,
      classifier_max_tokens: int,
      classifier_system: str
  ):
    self.scenario_prompts = scenario_prompts

    self.api_key = api_key

    self.target_model = target_model
    self.samples_per_prompt = samples_per_prompt
    self.target_model_temperature = target_model_temperature
    self.target_model_max_tokens = target_model_max_tokens
    self.system_prompt = system_prompt

    self.classifier_model = classifier_model
    self.classifier_temperature = classifier_temperature
    self.classifier_max_tokens = classifier_max_tokens
    self.classifier_system = classifier_system

    self.client = genai.Client(
      api_key=self.api_key
    )

  def __str__(self):
    return f"""
    ============================== Gemini Experiment ==============================
    Target Model: {self.target_model}
    Target Model Temperature: {self.target_model_temperature}
    Target Model Max Tokens: {self.target_model_max_tokens}
    Samples per prompt: {self.samples_per_prompt}

    Classifier Model: {self.classifier_model}
    Classifier Temperature: {self.classifier_temperature}
    Classifier Max Tokens: {self.classifier_max_tokens}
    Classifier System: {self.classifier_system}
    ===============================================================================
    """

  def generate_response(self, prompt: str, sample_index: int) -> str:
    """Generate a single response from the target model"""
    try:
      message = self.client.models.generate_content(
        model=self.target_model,
        config=types.GenerateContentConfig(
          system_instruction=self.system_prompt,
          max_output_tokens=self.target_model_max_tokens,
          temperature=self.target_model_temperature
        ),
        contents=prompt
      )

      text = message.text

      return text
    except Exception as e:
      print(f"[GEMINI API ERROR] {sample_index}: {e}")
      return ""

  def classify_response(self, text: str) -> tuple[list[str], dict, dict, str, bool, str]:
    """Use the classifier model to annotate group mentions, roles, and sentiment."""
    try:
      message = self.client.models.generate_content(
        model=self.classifier_model,
        config=types.GenerateContentConfig(
          system_instruction=self.classifier_system,
          max_output_tokens=self.classifier_max_tokens,
          temperature=self.classifier_temperature,
          response_mime_type="application/json" # force JSON
        ),
        contents=text
      )

      raw = message.text

      parsed = json.loads(raw)

      return(
          parsed.get("groups_mentioned", []),
          parsed.get("roles", {}),
          parsed.get("sentiment", {}),
          parsed.get("notes", ""),
          parsed.get("is_refusal", False),
          raw
      )
    except (json.JSONDecodeError, Exception, KeyError) as e:
      print(f"[CLASSIFIER ERROR]: {e}")
      return [], {}, {}, "", False, ""