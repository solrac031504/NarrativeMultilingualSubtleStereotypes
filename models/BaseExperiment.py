import json
import re
from datetime import datetime
from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    def __init__(
            self,
            prompts: dict[str, dict[str, str]],
            api_key: str,

            target_model: str,
            samples_per_prompt: int,
            target_model_temperature: float,
            target_model_max_tokens: int,
            system_prompt: str,

            classifier_model: str,
            classifier_temperature: float,
            classifier_max_tokens: int,
            classifier_system: str
    ):
        self.scenario_prompts = prompts
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

        self.client = self._build_client()

    def __str__(self):
        return f"""
        ============================== {self._provider_name()} Experiment ==============================
        Target Model: {self.target_model}
        Target Model Temperature: {self.target_model_temperature}
        Target Model Max Tokens: {self.target_model_max_tokens}
        Samples per prompt: {self.samples_per_prompt}
        Classifier Model: {self.classifier_model}
        Classifier Temperature: {self.classifier_temperature}
        Classifier Max Tokens: {self.classifier_max_tokens}
        Classifier System: {self.classifier_system}
        ====================================================================================================
        """
    
    @abstractmethod
    def _provider_name(self) -> str:
        """Return display name for __str__"""
        pass

    @abstractmethod
    def _build_client(self):
        """Instantiate and return API client"""
        pass

    @abstractmethod
    def _call_model(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Make an API call and return raw text response"""

    def generate_response(self, prompt: str, sample_index: int) -> str:
        try:
            return self._call_model(
                model=self.target_model,
                system_prompt=self.system_prompt,
                user_content=prompt,
                temperature=self.target_model_temperature,
                max_tokens=self.target_model_max_tokens
            )
        except Exception as e:
            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [{self._provider_name().upper()} API ERROR] {sample_index}: {e}")
            return ""
        
    def classify_response(self, text: str) -> tuple[list[str], dict, dict, str, bool, str]:
        try:
            raw = self._call_model(
                model=self.classifier_model,
                system_prompt=self.classifier_system,
                user_content=f"Text to annotate:\n\n{text}",
                temperature=self.classifier_temperature,
                max_tokens=self.classifier_max_tokens
            )
            raw = re.sub(f"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            return (
                parsed.get("groups_mentioned", []),
                parsed.get("roles", {}),
                parsed.get("sentiment", {}),
                parsed.get("notes", ""),
                parsed.get("is_refusal", False),
                raw
            )
        except Exception as e:
            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [CLASSIFIER ERROR]: {e}")
            return [], {}, {}, "", False, e