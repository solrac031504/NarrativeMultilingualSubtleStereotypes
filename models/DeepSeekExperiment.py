class DeepSeekExperiment:
  def __init__(
      self,

      scenario_prompts: dict[str, dict[str, str]],

      api_key: str,

      target_model: str,
      samples_per_prompt: int,
      target_model_temperature: int,
      target_model_max_tokens: int,

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

    self.classifier_model = classifier_model
    self.classifier_temperature = classifier_temperature
    self.classifier_max_tokens = classifier_max_tokens
    self.classifier_system = classifier_system