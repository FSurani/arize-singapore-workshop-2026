"""Experiment: compare two agent variants on the same golden dataset.

Runs two experiments against ``sunrise-support-golden`` so you can open the
Experiment Comparison view in Arize and see the evaluator scores side by side:

  - "retrieval-on"  : the full agent (knowledge-base retrieval enabled)
  - "retrieval-off" : the same agent with the retrieval tool removed

Expect groundedness and correctness on policy questions to drop noticeably
without retrieval (the model has to answer policy questions from memory),
while order/escalation behavior stays similar. That contrast is the point:
retrieval is what keeps policy answers grounded.

Usage:
    export OPENAI_API_KEY=...  ARIZE_SPACE_ID=...  ARIZE_API_KEY=...
    python evals/run_experiment.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import SYSTEM_PROMPT, build_agent
from agent.tools import check_refund_eligibility, escalate_to_human, lookup_order
from evals._arize import DATASET_NAME, get_client, get_space
from evals._common import EVALUATORS, make_task

NO_RETRIEVAL_TOOLS = [lookup_order, check_refund_eligibility, escalate_to_human]


def _run(client, space, name, agent):
    experiment, df = client.experiments.run(
        name=name,
        dataset=DATASET_NAME,
        space=space,
        task=make_task(agent),
        evaluators=EVALUATORS,
        concurrency=4,
    )
    print(f"  {name}: {len(df)} runs" + (f" (experiment {experiment.id})" if experiment else ""))


def main() -> None:
    from datetime import datetime

    client = get_client()
    space = get_space()

    # Shared suffix keeps the pair comparable and avoids "experiment already
    # exists" errors when the cell is re-run.
    suffix = datetime.now().strftime("%m%d-%H%M")

    print("Running variant 1/2: retrieval-on")
    _run(client, space, f"retrieval-on-{suffix}", build_agent())

    print("Running variant 2/2: retrieval-off")
    no_retrieval_prompt = SYSTEM_PROMPT.replace(
        "use `search_knowledge_base` and answer grounded in the retrieved passages. "
        "If the knowledge base does not cover it, say so rather than guessing.",
        "answer from your own knowledge (no knowledge-base tool is available).",
    )
    _run(
        client,
        space,
        f"retrieval-off-{suffix}",
        build_agent(prompt=no_retrieval_prompt, tools=NO_RETRIEVAL_TOOLS),
    )

    print(
        f"\nDone. Open 'sunrise-support-golden' in Arize and compare the "
        f"'retrieval-on-{suffix}' vs 'retrieval-off-{suffix}' experiments side by side."
    )


if __name__ == "__main__":
    main()
