"""Curate a dataset from the real traces produced during the workshop.

This is the payoff loop: after participants chat with the agent (via the
Gradio UI or the notebook), their conversations are real traces in the
``arize-singapore-workshop`` project. This script:

  1. Exports recent spans from the project.
  2. Builds a new Arize dataset from those real interactions.

From there you iterate an eval template on this real data and then deploy it
as a continuous online evaluator with ``evals/setup_online_eval.py`` (Section
7 of the notebook). That closes the loop: build -> trace -> offline eval +
experiment -> generate real traces -> curate them -> ship an online eval.

Usage:
    export ARIZE_SPACE_ID=...  ARIZE_API_KEY=...
    python evals/curate_from_traces.py [--hours 24] [--limit 100]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._arize import PROJECT_NAME, get_client, get_space

CURATED_DATASET_NAME = "sunrise-support-from-traces"

# Span columns that may hold the user input / agent output (OpenInference attrs).
_INPUT_COLS = ["attributes.input.value", "input.value", "input"]
_OUTPUT_COLS = ["attributes.output.value", "output.value", "output"]


def _first_present(row, columns):
    for col in columns:
        if col in row and row[col] not in (None, ""):
            return str(row[col])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="Look back this many hours.")
    parser.add_argument("--limit", type=int, default=100, help="Max spans to pull.")
    args = parser.parse_args()

    client = get_client()
    space = get_space()

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)

    df = client.spans.list(
        project=PROJECT_NAME,
        space=space,
        start_time=start,
        end_time=end,
        limit=args.limit,
    ).to_df()

    if df is None or len(df) == 0:
        print("No spans found in that window. Chat with the agent first, then re-run.")
        return

    examples, seen = [], set()
    for _, row in df.iterrows():
        user_input = _first_present(row, _INPUT_COLS)
        agent_output = _first_present(row, _OUTPUT_COLS)
        if not user_input or user_input in seen:
            continue
        seen.add(user_input)
        examples.append({"input": user_input, "agent_output": agent_output})

    if not examples:
        print(f"Pulled {len(df)} spans but found none with usable input/output columns.")
        print("Available columns:", list(df.columns))
        return

    dataset = client.datasets.create(
        name=CURATED_DATASET_NAME,
        space=space,
        examples=examples,
    )
    print(f"Created dataset '{CURATED_DATASET_NAME}' from {len(examples)} real interactions.")
    print(f"Dataset id: {dataset.id}")
    print(
        "\nNext steps:\n"
        "  1. (Optional) Route a few rows through a labeling queue in Arize to add ground truth.\n"
        "  2. Iterate an eval template against this dataset until it scores sensibly.\n"
        "  3. Deploy it as a continuous online evaluator: python evals/setup_online_eval.py"
    )


if __name__ == "__main__":
    main()
