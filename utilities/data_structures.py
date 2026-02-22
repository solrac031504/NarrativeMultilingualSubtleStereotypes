from dataclasses import dataclass, field

@dataclass
class ClassifierResponse:
  groups_mentioned: list[str] = field(default_factory=list)
  roles: dict[str, str] = field(default_factory=dict)
  sentiment: dict[str, str] = field(default_factory=dict)
  notes: str = ""
  is_refusal: bool = False

@dataclass
class AnnotatedResponse(ClassifierResponse):
  scenario: str = ""
  language: str = ""
  sample_index: int = 0
  raw_response: str = ""
  classifier_raw: str = ""