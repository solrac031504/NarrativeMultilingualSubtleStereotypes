# General imports
import json
import time
from typing import Optional
from collections import defaultdict

# Data structure imports
from data_structures import AnnotatedResponse

# Model imports
from models.ChatGPTExperiment import ChatGPTExperiment
from models.ClaudeExperiment import ClaudeExperiment

def run_experiments(
      model: ClaudeExperiment | ChatGPTExperiment,
      scenarios: Optional[list[str]] = None,
      languages: Optional[list[str]] = None
  ) -> list[AnnotatedResponse]:
    """
    Run the full experiment across the specified scenarios and languages

    <INPUTS>
    scenarios: subset of SCENARIO_PROMPTS keys. Defaults to all
    langauges: subset of langauge codes. Defaults to all available

    <OUTPUTS>
    List of AnnotatedResponses objects with generation and classification
    """

    target_scenarios = scenarios or list(model.scenario_prompts.keys())
    results: list[AnnotatedResponse] = []

    for scenario in target_scenarios:
      prompt_bank = model.scenario_prompts[scenario]
      target_languages = languages or list(prompt_bank.keys())

      for language in target_languages:
        if language not in prompt_bank:
          print(f"[EXPERIMENT] No prompt for scenario={scenario}, lang={language}")
          continue

        prompt = prompt_bank[language]
        print(f"[EXPERIMENT] Scenario: {scenario} | Language: {language}")

        for i in range(model.samples_per_prompt):
          print(f"[EXPERIMENT] Sample {i+1}/{model.samples_per_prompt}")

          # Step 1: Generate response
          response_text = model.generate_response(prompt, i)

          # Step 2: Classify
          groups, roles, sentiment, notes, is_refusal = ([], {}, {}, "", False)
          if response_text:
            groups, roles, sentiment, notes, is_refusal = model.classify_response(response_text)

          annotated = AnnotatedResponse(
              scenario=scenario,
              language=language,
              sample_index=i,
              raw_response=response_text,
              groups_mentioned=groups,
              roles=roles,
              sentiment=sentiment
          )
          results.append(annotated)
          print(f"Groups found: {groups or 'none'} | Refusal: {is_refusal}")

          # Rate limiting
          time.sleep(0.5)
    return results

def compute_statistics(results: list[AnnotatedResponse]) -> dict:
  """
  Compute distributional statistics from annotated results.

  <OUTPUTS>
  Nested dict: {
    scenario: {
      language: {
        group: {
          role_counts,
          sentiment_counts,
          mention_rate
        }
      }
    }
  }
  """

  stats: dict = defaultdict(lambda: defaultdict(lambda: {
      "total_samples": 0,
      "refusal_count": 0,
      "group_mentions": defaultdict(int),
      "role_counts": defaultdict(lambda: defaultdict(int)),
      "sentiment_counts": defaultdict(lambda: defaultdict(int))
  }))

  for r in results:
    cell = stats[r.scenario][r.language]
    cell["total_samples"] += 1
    if r.refusal:
      cell["refusal_count"] += 1

    for group in r.groups_mentioned:
      cell["group_mentions"][group] += 1
      role = r.roles.get(group, "unspecified")
      sent = r.sentiment.get(group, "neutral")
      cell["role_counts"][group][role] += 1
      cell["sentiment_counts"][group][sent] += 1

    # Compute mention rates
    output = {}
    for scenario, lang_data in stats.items():
      output[scenario] = {}
      for lang, cell in lang_data.items():
        n = cell["total_samples"]
        output[scenario][lang] = {
            "total_samples": n,
            "refusal_rate": cell["refusal_count"] / n if n else 0,
            "groups": {}
        }

      for group, count in cell["group_mentions"].items():
        output[scenario][lang]["groups"][group] = {
            "mention_rate": count / n,
            "mention_count": count,
            "role_distribution": dict(cell["role_counts"][group]),
            "sentiment_distribution": dict(cell["sentiment_counts"][group])
        }

  return output
  
def print_summary(stats: dict):
  """Nice looking print"""
  print("\n" + "="*70)
  print("EXPERIMENT SUMMARY")
  print("="*70)

  for scenario, lang_data in stats.items():
    print(f"\n Scenario: {scenario.upper()}")
    for lang, data in lang_data.items():
      print(f"Language: {lang}")
      print(f"Samples: {data['total_samples']} | Refusal rate: {data['refusal_rate']:.1%}")
      if not data["groups"]:
        print("     No protected groups detected")
      for group, gdata in sorted(data["groups"].items(), key=lambda x: -x[1]["mention_rate"]):
        top_role = max(gdata["role_distribution"], key=gdata["role_distribution"].get, default="-")
        top_sent = max(gdata["sentiment_distribution"], key=gdata["sentiment_distribution"].get, default="-")
        print(f"     {group}: mention_rate={gdata['mention_rate']:.1%},"
              f"top_role={top_role}, top_sentiment={top_sent}")