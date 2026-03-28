"""
Since the JSON failed to generate, need to summarize the experiment summaries from the .out files
"""

import re
import csv
import sys
from pathlib import Path

def parse_summary(input_path: str) -> list[dict]:
    """
    Parse the experiment summary text file and return a list of row dicts
    """

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.split("\n")

    rows = []
    scenario = None
    language = None

    # Pattern from the experiment
    group_pattern = re.compile(
        r"^(.+?):\s+mention_rate=([\d.]+)%,\s+classifier_agreement=([\d.]+)%,"
        r"\s+top_role=(.+?),\s+top_sentiment=(.+)$"
    )

    # Iterate over lines
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Scenario:"):
            scenario = stripped.split("Scenario:", 1)[1].strip()

        elif stripped.startswith("Language:"):
            language = stripped.split("Language:", 1)[1].strip()

        else:
            m = group_pattern.match(stripped)
            if m:
                rows.append({
                    "Scenario": scenario,
                    "Language": language,
                    "Group": m.group(1).strip(),
                    "Mention Rate": round(float(m.group(2)) / 100, 10),
                    "Classifier Agreement": round(float(m.group(3)) / 100, 10),
                    "Top Role": m.group(4).strip(),
                    "Top Sentiment": m.group(5).strip()
                })

    return rows

def write_csv(rows: list[dict], output_path: str):
    """
    Write a parsed row to a CSV file
    """

    fieldnames = [
        "Scenario",
        "Language",
        "Group",
        "Mention Rate",
        "Classifier Agreement",
        "Top Role",
        "Top Sentiment"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_experiment.py <input_file> [output_file]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        if output_path.is_dir():
            output_path = output_path / Path(input_path).with_suffix(".csv").name
        output_path = str(output_path)
    else:
        output_path = str(Path(input_path).with_suffix(".csv"))

    print(f"Parsing: {input_path}")
    rows = parse_summary(input_path=input_path)

    if not rows:
        print("Warning: no data rows were found")
        sys.exit(1)

    write_csv(rows=rows, output_path=output_path)
    print(f"Done parsing experiment. {len(rows):,} rows written to: {output_path}")

if __name__ == "__main__":
    main()