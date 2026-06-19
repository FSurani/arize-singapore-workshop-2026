"""Set up a continuous online evaluator for the agent using the Arize `ax` CLI.

Promoted to a core part of the workshop (Section 7). It:

  1. Creates (or reuses) a Gemini AI integration in Arize from your GOOGLE_API_KEY.
  2. Creates a template (LLM-as-a-judge) evaluator: resolved / not_resolved.
  3. Attaches it to the `arize-singapore-workshop` project as a CONTINUOUS,
     trace-level evaluation task, then triggers an on-demand run.

Prerequisites:
  pip install arize-ax-cli
  export GOOGLE_API_KEY=...
  export ARIZE_SPACE_ID=U3BhY2U6...
  export ARIZE_API_KEY=ak-...        # CLI auth; a developer key from
                                     # app.arize.com/admin > API Keys also works

Usage:
  python evals/setup_online_eval.py [--project arize-singapore-workshop]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

EVALUATOR_NAME = "support-resolution"
TASK_NAME = "support-resolution-online"

# Arize AI-integration provider + judge model for Gemini.
PROVIDER = "gemini"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")

RESOLUTION_TEMPLATE = """You are evaluating a customer-support agent for Sunrise Outfitters, an online outdoor retailer.

Customer message:
{{input}}

Agent reply:
{{output}}

Did the agent correctly and helpfully resolve the customer's request, with accurate order/policy details and an appropriate tone?
Answer "resolved" if it did, or "not_resolved" if it was inaccurate, unhelpful, or ignored the request."""


def ax(args, parse=False):
    """Run an `ax` command. Return parsed JSON if parse=True, else stdout text."""
    res = subprocess.run(["ax", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"`ax {' '.join(args[:2])}` failed:\n{res.stderr or res.stdout}")
    if not parse:
        if res.stdout.strip():
            print(res.stdout)
        return res.stdout
    out = res.stdout
    starts = [i for i in (out.find("{"), out.find("[")) if i != -1]
    if not starts:
        raise RuntimeError(f"No JSON returned:\n{out}")
    return json.loads(out[min(starts):])


def gemini_integration_id():
    data = ax(["ai-integrations", "list", "-o", "json"], parse=True)
    for i in data.get("ai_integrations", []):
        if i.get("provider", "").lower() == PROVIDER.lower():
            return i["id"]
    return None


def evaluator_id(name, space):
    data = ax(["evaluators", "list", "--name", name, "--space", space, "-o", "json"], parse=True)
    for e in data.get("evaluators", []):
        if e.get("name") == name:
            return e["id"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="arize-singapore-workshop")
    args = parser.parse_args()

    space = os.environ.get("ARIZE_SPACE_ID")
    if not space:
        raise SystemExit("ARIZE_SPACE_ID is not set.")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set (needed to create the AI integration).")

    # 1) AI integration
    integration = gemini_integration_id()
    if integration is None:
        print("Creating a Gemini AI integration from GOOGLE_API_KEY...")
        ax([
            "ai-integrations", "create",
            "--name", "workshop-gemini",
            "--provider", PROVIDER,
            "--api-key", os.environ["GOOGLE_API_KEY"],
            "--enable-default-models",
            "--function-calling-enabled",
            "-o", "json",
        ], parse=True)
        integration = gemini_integration_id()
    print("AI integration id:", integration)

    # 2) Evaluator
    evaluator = evaluator_id(EVALUATOR_NAME, space)
    if evaluator is None:
        ax([
            "evaluators", "create-template-evaluator",
            "--name", EVALUATOR_NAME,
            "--space", space,
            "--commit-message", "initial version",
            "--template-name", "resolution",
            "--template", RESOLUTION_TEMPLATE,
            "--ai-integration-id", integration,
            "--model-name", JUDGE_MODEL,
            "--classification-choices", '{"resolved": 1, "not_resolved": 0}',
            "--include-explanation",
            "--data-granularity", "trace",
            "-o", "json",
        ], parse=True)
        evaluator = evaluator_id(EVALUATOR_NAME, space)
    print("Evaluator id:", evaluator)

    # 3) Continuous evaluation task on the project
    evaluators_cfg = json.dumps([{
        "evaluator_id": evaluator,
        "column_mappings": {"input": "attributes.input.value", "output": "attributes.output.value"},
    }])
    try:
        ax([
            "tasks", "create-evaluation",
            "--name", TASK_NAME,
            "--task-type", "template_evaluation",
            "--evaluators", evaluators_cfg,
            "--project", args.project,
            "--space", space,
            "--is-continuous",
            "--sampling-rate", "1.0",
            "-o", "json",
        ], parse=True)
        print(f"Created continuous evaluation task '{TASK_NAME}'.")
    except RuntimeError as e:
        print(f"Task '{TASK_NAME}' may already exist - continuing.\n{e}")

    # Trigger a run now and show recent runs
    ax(["tasks", "trigger-run", TASK_NAME])
    time.sleep(3)
    ax(["tasks", "list-runs", TASK_NAME, "-l", "5"])
    print(f"\nDone. Open the '{args.project}' project in Arize - traces now carry a 'resolution' eval.")


if __name__ == "__main__":
    main()
