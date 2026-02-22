from dataclasses import dataclass, field

@dataclass
class AnnotatedResponse:
  scenario: str
  language: str
  sample_index: int
  raw_response: str
  groups_mentioned: list[str] = field(default_factory=list)
  roles: dict[str, str] = field(default_factory=dict)
  sentiment: dict[str, str] = field(default_factory=dict)
  notes: str = ""
  refusal: bool = False
  classifier_raw: str = ""