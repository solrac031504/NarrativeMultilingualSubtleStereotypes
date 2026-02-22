import json
from xai_sdk import Client
from xai_sdk.chat import user, system
from xai_sdk.types import ResponseFormat

class GrokExperiment:
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

    self.client = Client(
        api_key = self.api_key,
    )

  def __str__(self):
    return f"""
    ============================== Grok Experiment ==============================
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
      chat = self.client.chat.create(
        model=self.target_model,
        max_tokens=self.target_model_max_tokens,
        temperature=self.target_model_temperature
      )

      chat.append(system(self.system_prompt))
      chat.append(user(prompt))

      text = chat.sample()

      return text
    except Exception as e:
      print(f"[DEEPSEEK API ERROR] {sample_index}: {e}")
      return ""

  def classify_response(self, text: str) -> tuple[list[str], dict, dict, str, bool, str]:
    """Use the classifier model to annotate group mentions, roles, and sentiment."""
    try:
      chat = self.client.chat.create(
        model=self.classifier_model,
        max_tokens=self.classifier_max_tokens,
        temperature=self.classifier_temperature,
        response_format="json_object"
      )

      chat.append(system(self.classifier_system))
      chat.append(user(f"Text to annotate:\n\n{text}"))

      raw = chat.sample()

      # Strip markdown fences if in response
      if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
          raw = raw[4:]
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