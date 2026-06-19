"""Offline evaluation of the support agent against the golden dataset.

Runs the agent over every example in the ``sunrise-support-golden`` dataset as
an Arize experiment, scoring each output with four evaluators:

  - tool_selection         (code)      - did it call the expected tool?
  - escalation_appropriate (code)      - did it escalate iff expected?
  - correctness            (LLM judge) - does the reply match the expected answer?
  - groundedness           (LLM judge) - is the reply supported by retrieved context?

Results and traces log to Arize, where you can open the run and inspect
per-example scores.

Usage:
    export GOOGLE_API_KEY=...  ARIZE_SPACE_ID=...  ARIZE_API_KEY=...
    python evals/offline_evals.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_agent
from evals._arize import DATASET_NAME, get_client, get_space
from evals._common import EVALUATORS, make_task


def main() -> None:
    from datetime import datetime

    client = get_client()
    space = get_space()
    agent = build_agent()

    # Timestamped so re-running the cell doesn't hit "experiment already exists".
    name = "offline-baseline-" + datetime.now().strftime("%m%d-%H%M")
    experiment, df = client.experiments.run(
        name=name,
        dataset=DATASET_NAME,
        space=space,
        task=make_task(agent),
        evaluators=EVALUATORS,
        concurrency=4,
    )

    print(f"Ran offline evaluation on '{DATASET_NAME}' over {len(df)} examples.")
    if experiment is not None:
        print(f"Experiment id: {experiment.id}")
    print("Open the experiment in Arize to see per-example scores across all evaluators.")


if __name__ == "__main__":
    main()
