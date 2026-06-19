"""Upload the curated golden dataset to Arize.

Reads ``evals/datasets/golden_support.json`` (a small, hand-labeled set of
Sunrise Outfitters support examples, including edge cases and "should
escalate" rows) and creates a dataset in your Arize space. The dataset then
shows up in the Arize Datasets UI and is what ``offline_evals.py`` and
``run_experiment.py`` run against.

Usage:
    export ARIZE_SPACE_ID=...  ARIZE_API_KEY=...
    python evals/build_golden_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._arize import DATASET_NAME, get_client, get_space

DATA_FILE = Path(__file__).parent / "datasets" / "golden_support.json"


def main() -> None:
    examples = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    client = get_client()
    space = get_space()

    dataset = client.datasets.create(
        name=DATASET_NAME,
        space=space,
        examples=examples,
    )
    print(f"Created dataset '{DATASET_NAME}' with {len(examples)} examples.")
    print(f"Dataset id: {dataset.id}")
    print("Open it in Arize under Datasets to inspect the examples.")


if __name__ == "__main__":
    main()
