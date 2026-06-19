"""Small helpers shared by the eval scripts for talking to Arize.

Uses the unified Arize Python client (``arize`` v8+). Credentials come from
the same env vars the workshop already uses for tracing:

  ARIZE_SPACE_ID   - your Arize Space ID
  ARIZE_API_KEY    - your Arize API key (Settings -> Space API Keys)

For local runs, values are also read from a ``.env`` file (see ``.env.example``).
On Colab the keys are set via ``getpass`` in the notebook instead.

If a datasets/experiments call returns 401, your space may require a developer
key; set ARIZE_API_KEY to a key from app.arize.com/admin > API Keys.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from arize import ArizeClient

load_dotenv()

PROJECT_NAME = os.getenv("ARIZE_PROJECT", "arize-singapore-workshop")
DATASET_NAME = "sunrise-support-golden"


def get_client() -> ArizeClient:
    api_key = os.getenv("ARIZE_API_KEY")
    if not api_key:
        raise SystemExit("ARIZE_API_KEY is not set.")
    return ArizeClient(api_key=api_key)


def get_space() -> str:
    space = os.getenv("ARIZE_SPACE_ID")
    if not space:
        raise SystemExit("ARIZE_SPACE_ID is not set.")
    return space
