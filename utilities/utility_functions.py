# General imports
import json
import os
import time
import sys
from datetime import datetime
from typing import Optional
from collections import defaultdict
from pathlib import Path

# Data structure imports
from utilities.data_structures import AnnotatedResponse
from utilities.Tee import Tee

# Email sending
from utilities.EmailNotifier import EmailNotifer

# Model imports
from models.ChatGPTExperiment import ChatGPTExperiment
from models.ClaudeExperiment import ClaudeExperiment
from models.DeepSeekExperiment import DeepSeekExperiment
from models.GeminiExperiment import GeminiExperiment
from models.GrokExperiment import GrokExperiment

def run_experiments(
      model: ClaudeExperiment | ChatGPTExperiment | DeepSeekExperiment | GeminiExperiment | GrokExperiment,

      classifiers: list[ClaudeExperiment | ChatGPTExperiment | DeepSeekExperiment | GeminiExperiment | GrokExperiment],

      notifier: EmailNotifer,
      prefix: str,

      log_dir: str,
      log_filename: str,

      output_dir: str = "outputs",
      output_filename: str = "results.json",

      scenarios: Optional[list[str]] = None,
      languages: Optional[list[str]] = None,
  ) -> list[AnnotatedResponse]:
    """
    Run the full experiment across the specified scenarios and languages

    <INPUTS>
    model: The model class being tested. Instantiated with the model already
    classifiers: List of classifier models used to analyze responses
    notifier: Notifer class to send email updates
    prefix: Shorthand version of the provider and model name. Used to send emails
    log_dir: Directory to write log/out info
    log_filename: Name of log file being written to
    output_dir: Directory to write final results to
    output_filename: Name of file with final results
    scenarios: subset of SCENARIO_PROMPTS keys. Defaults to all
    langauges: subset of langauge codes. Defaults to all available
    """

    # Setup logging if logging filepath was provided
    tee = None
    if log_dir and log_filename:
      if not log_filename.endswith(".out"):
        log_filename += ".out"
      Path(log_dir).mkdir(parents=True, exist_ok=True)
      log_path = os.path.join(log_dir, log_filename)
      tee = Tee(log_path)
      sys.stdout = tee
      print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [LOG] Session started")

    try:
      print(model)

      target_scenarios = scenarios or list(model.scenario_prompts.keys())
      results: list[AnnotatedResponse] = []

      for scenario in target_scenarios:
        prompt_bank = model.scenario_prompts[scenario]
        target_languages = languages or list(prompt_bank.keys())

        for language in target_languages:
          if language not in prompt_bank:
            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [EXPERIMENT] No prompt for scenario={scenario}, lang={language}")
            continue

          notifier.notify_update(
            prefix=prefix,
            model=model.target_model,
            scenario=scenario,
            lang=language
          )

          prompt = prompt_bank[language]
          print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [EXPERIMENT] Scenario: {scenario} | Language: {language}")

          for i in range(model.samples_per_prompt):
            print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [EXPERIMENT] Sample {i+1}/{model.samples_per_prompt}")

            # Step 1: Generate response
            response_text = model.generate_response(prompt, i)

            # Step 2: Classify with each classifier
            for classifier in classifiers:
              groups, roles, sentiment, notes, is_refusal, raw = ([], {}, {}, "", False, "")
              if response_text:
                groups, roles, sentiment, notes, is_refusal, raw = classifier.classify_response(response_text)

              annotated = AnnotatedResponse(
                  classifier=classifier.target_model,
                  scenario=scenario,
                  language=language,
                  sample_index=i,
                  raw_response=response_text,
                  groups_mentioned=groups,
                  roles=roles,
                  sentiment=sentiment,
                  notes=notes,
                  is_refusal=is_refusal,
                  classifier_raw=raw
              )
              results.append(annotated)
              print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [EXPERIMENT] Classifier: {classifier.target_model} | Groups found: {groups or 'none'} | Refusal: {is_refusal}")

            # Rate limiting
            time.sleep(0.5)

      # Compute statistics (aggregated across all classifiers)
      stats = compute_statistics(results)

      # Print stats
      print_summary(stats)

      # Save to JSON
      save_results(
        results=results,
        stats=stats,
        output_dir=output_dir,
        filename=output_filename,
        model=model,
        classifiers=classifiers,
      )
    except Exception as e:
      print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [EXPERIMENT] Exception: {e}")
      raise e
    finally:
      # Always restore stdout
      if tee:
        print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [LOG] Session ended")
        sys.stdout = tee.terminal
        tee.close()

def compute_statistics(results: list[AnnotatedResponse]) -> dict:
  """
  Compute distributional statistics from annotated results, aggregated across all classifiers.

  mention_rate: fraction of samples in which at least one classifier mentioned the group (i.e. per-sample coverage, not per-classifier frequency)

  classifier_agreement: average number of classifiers that flagged the group per sample, across only the samples where it was mentioned at least once

  <OUTPUTS>
  Nested dict: {
    scenario: {
      language: {
        group: {
          role_counts,
          sentiment_counts,
          mention_rate,
          classifier_agreement
        }
      }
    }
  }
  """

  stats: dict = defaultdict(lambda: defaultdict(lambda: {
      "total_samples": 0,
      "refusal_count": 0,
      # group -> set of sample indices where >= 1 classifier mentioned it
      "group_sample_hits": defaultdict(set),
      # group -> total classifier mentions across all samples (for agreement calc)
      "group_classifier_hits": defaultdict(int),
      "role_counts": defaultdict(lambda: defaultdict(int)),
      "sentiment_counts": defaultdict(lambda: defaultdict(int))
  }))

  # Track unique (scenario, language, sample_index) tuples to count samples once
  seen_samples: set = set()

  # Track number of classifiers per (scenario, language)
  classifier_ids: dict = defaultdict(lambda: defaultdict(set))

  for r in results:
    cell = stats[r.scenario][r.language]
    classifier_ids[r.scenario][r.language].add(r.classifier)

    sample_key = (r.scenario, r.language, r.sample_index)
    if sample_key not in seen_samples:
      seen_samples.add(sample_key)
      cell["total_samples"] += 1
      if r.is_refusal:
        cell["refusal_count"] += 1

    for group in r.groups_mentioned:
      # Record that this sample had the group flagged (de-duped per sample)
      cell["group_sample_hits"][group].add(r.sample_index)
      # Count every classifier flag for agreement calculation
      cell["group_classifier_hits"][group] += 1

      role = r.roles.get(group, "unspecified")
      sent = r.sentiment.get(group, "neutral")
      cell["role_counts"][group][role] += 1
      cell["sentiment_counts"][group][sent] += 1

  output = {}
  for scenario, lang_data in stats.items():
    output[scenario] = {}
    for lang, cell in lang_data.items():
      n_samples = cell["total_samples"]
      n_classifiers = len(classifier_ids[scenario][lang])

      output[scenario][lang] = {
          "total_samples": n_samples,
          "refusal_rate": cell["refusal_count"] / n_samples if n_samples else 0,
          "groups": {}
      }

      for group, sample_hit_set in cell["group_sample_hits"].items():
        # Samples where >= 1 classifier mentioned this group
        samples_with_mention = len(sample_hit_set)

        # mention_rate: fraction of samples where the group appeared at all
        mention_rate = samples_with_mention / n_samples if n_samples else 0

        # classifier_agreement: average classifiers per sample that flagged it,
        # computed only over the samples where it was mentioned at least once
        total_classifier_flags = cell["group_classifier_hits"][group]
        classifier_agreement = (
            total_classifier_flags / (samples_with_mention * n_classifiers)
            if samples_with_mention and n_classifiers
            else 0
        )

        output[scenario][lang]["groups"][group] = {
            "mention_rate": mention_rate,
            "classifier_agreement": classifier_agreement,
            "mention_count": total_classifier_flags,
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
    print(f"\nScenario: {scenario.upper()}")
    for lang, data in lang_data.items():
      print(f"Language: {lang}")
      print(f"Samples: {data['total_samples']} | Refusal rate: {data['refusal_rate']:.1%}")
      if not data["groups"]:
        print("     No protected groups detected")
      for group, gdata in sorted(data["groups"].items(), key=lambda x: -x[1]["mention_rate"]):
        top_role = max(gdata["role_distribution"], key=gdata["role_distribution"].get, default="-")
        top_sent = max(gdata["sentiment_distribution"], key=gdata["sentiment_distribution"].get, default="-")
        print(f"     {group}: mention_rate={gdata['mention_rate']:.1%}, "
              f"classifier_agreement={gdata['classifier_agreement']:.1%}, "
              f"top_role={top_role}, top_sentiment={top_sent}")

def save_results(
    results: list[AnnotatedResponse],
    stats: dict,
    output_dir: str,
    filename: str,
    model: ClaudeExperiment | ChatGPTExperiment | DeepSeekExperiment | GeminiExperiment | GrokExperiment,
    classifiers: list[ClaudeExperiment | ChatGPTExperiment | DeepSeekExperiment | GeminiExperiment | GrokExperiment],
    indent: int = 2
):
  """
  Serialize experiment results and write to JSON file

  <INPUTS>
  results: List of AnnotatedResponse objects from run_experiments()
  stats: Computed statistics dict from compute_statistics()
  output_dir: Directory to write the file into (created if doesn't exist)
  filename: Output filename (.json appended if missing)
  model: The model object tested
  classifiers: List of classifier models used
  indent: JSON indentation level. Default: 2
  """

  # Make sure file name ends with .json
  if not filename.endswith(".json"):
    filename += ".json"

  # Create output directory if needed
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  output_path = os.path.join(output_dir, filename)

  # Group results by scenario -> language -> sample_index
  grouped_results: dict[str, dict[str, dict[int, list[AnnotatedResponse]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
  )
  for r in results:
    grouped_results[r.scenario][r.language][r.sample_index].append(r)

  scenarios = []

  for scenario, lang_data in grouped_results.items():
    languages = []

    for language, sample_data in lang_data.items():
      language_stats = stats.get(scenario, {}).get(language, {})

      serialized_responses = []
      for sample_index in sorted(sample_data.keys()):
        annotations = sample_data[sample_index]

        # All annotations for this sample share the same raw_response
        raw_response = annotations[0].raw_response if annotations else ""

        classifier_entries = [
          {
            "classifier": r.classifier,
            "groups_mentioned": r.groups_mentioned,
            "roles": r.roles,
            "sentiment": r.sentiment,
            "notes": r.notes,
            "is_refusal": r.is_refusal,
            "classifier_raw": r.classifier_raw
          }
          for r in annotations
        ]

        serialized_responses.append({
          "sample_index": sample_index,
          "raw_response": raw_response,
          "classifiers": classifier_entries
        })

      groups_summary = [
        {
          "group": group,
          "mention_rate": gdata["mention_rate"],
          "classifier_agreement": gdata["classifier_agreement"],
          "top_role": max(gdata["role_distribution"], key=gdata["role_distribution"].get, default="-"),
          "top_sentiment": max(gdata["sentiment_distribution"], key=gdata["sentiment_distribution"].get, default="-")
        }
        for group, gdata in language_stats.get("groups", {}).items()
      ]

      languages.append({
        "language": language,
        "responses": serialized_responses,
        "stats": {
          "refusal_rate": language_stats.get("refusal_rate", 0.0),
          "groups": groups_summary
        }
      })

    scenarios.append({
      "scenario": scenario,
      "languages": languages
    })

  output = {
    "samples_per_prompt": model.samples_per_prompt,
    "target_model": {
      "name": model.target_model,
      "temperature": model.target_model_temperature,
      "max_tokens": model.target_model_max_tokens
    },
    "classifier_models": [
      {
        "name": c.target_model,
        "temperature": c.target_model_temperature,
        "max_tokens": c.target_model_max_tokens,
        "system": c.system_prompt
      }
      for c in classifiers
    ],
    "scenarios": scenarios
  }

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=indent, ensure_ascii=False)

  print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')} [SAVE] Results written to {output_path}")