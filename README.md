# Arize Singapore Workshop

A hands-on workshop that takes a **LangGraph customer-support agent** through the
core **Arize AX** dev loop: **build → trace → offline-eval → experiment →
chat live & watch traces**.

You build "Sunny", the support assistant for a fictional outdoor retailer
(Sunrise Outfitters). The agent does **real retrieval** over a small policy
knowledge base and **escalates** hard or high-risk requests to a human — the
most common enterprise customer-support pattern — then you trace and evaluate
it in Arize. The agent's workhorse model is OpenAI's **`gpt-4o-mini`**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FSurani/arize-singapore-workshop-2026/blob/main/notebook/arize_workshop.ipynb)

## What you'll do

1. **Build** a tool-using agent (orders, refunds, knowledge-base retrieval, escalation).
2. **Trace** it into Arize with two lines of auto-instrumentation (incl. retrieval spans).
3. **Offline-eval** it against a curated golden dataset (a code check + an LLM judge).
4. **Experiment**: compare two models (`gpt-4o-mini` vs `gpt-4o`) side by side.
5. **Chat live** via a Gradio UI and watch your own traces appear in Arize.

## Before you arrive (prerequisites)

The venue has limited time, so please have these ready **before** the session:

1. **An OpenAI API key** (`OPENAI_API_KEY`) — get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. **A free Arize account** at [app.arize.com](https://app.arize.com), and your
   **Space ID + API key** from **Settings → Space API Keys**.

That's the only input you need — the notebook prompts for all three keys at the top.

## Suggested agenda (~50–60 min hands-on)

| # | Section | ~Time |
|---|---------|-------|
| 0 | Intro + slides on agent observability | 10–15 min |
| 1 | Setup & keys | 5 min |
| 2 | Build the agent (retrieval + escalation), run untraced + try the UI | 10 min |
| 3 | Add Arize tracing, explore traces | 10 min |
| 4 | Offline evals against the golden dataset | 10–15 min |
| 5 | Experiment: compare two models | 10 min |
| 6 | Live chat via Gradio — everyone generates traces | 5–10 min |
| 7 | (Optional / take-home) single → multi-agent | — |

## Running the notebook

The notebook (`notebook/arize_workshop.ipynb`) is the workshop. It imports the
**agent** from `agent/` and writes all the **evaluation code inline**, so you
see every Arize SDK call as you run it. Run it from a checkout of this repo —
locally with Jupyter, or by opening the repo in Colab. Just run the cells top to
bottom and enter your three keys when prompted.

## Run the local Gradio app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add OPENAI_API_KEY (+ Arize keys to enable tracing)
python app.py          # opens the Gradio chat UI, traced into Arize
```

## Eval scripts (CLI equivalents of the notebook)

The `evals/` folder mirrors the notebook's eval steps as standalone scripts, and
includes **all four evaluators** (the notebook leaves two as exercises — these
are the reference implementations):

```bash
python evals/build_golden_dataset.py   # upload the golden dataset
python evals/offline_evals.py          # offline eval experiment (4 evaluators)
python evals/run_experiment.py         # compare two agent variants
```

Optional "going further" extras (not part of the core workshop):

```bash
python evals/curate_from_traces.py     # build a dataset from real traces
python evals/setup_online_eval.py      # deploy a continuous online evaluator (needs the `ax` CLI)
```

## Project layout

```
arize-singapore-workshop/
├── agent/
│   ├── knowledge/      # tiny policy KB (shipping, returns, sizing, warranty, payments)
│   ├── retrieval.py    # in-memory vector store + search_knowledge_base()
│   ├── tools.py        # order lookup, refund check, KB retrieval, escalation
│   └── graph.py        # LangGraph ReAct agent factory (OpenAI)
├── app.py              # Gradio chat UI (traces into Arize if keys are set)
├── notebook/
│   └── arize_workshop.ipynb   # the workshop notebook
├── evals/
│   ├── datasets/golden_support.json  # curated golden dataset
│   ├── build_golden_dataset.py       # upload the golden dataset to Arize
│   ├── offline_evals.py              # offline eval over the dataset
│   ├── run_experiment.py             # compare two variants
│   ├── curate_from_traces.py         # (extra) build a dataset from real spans
│   ├── setup_online_eval.py          # (extra) deploy a continuous online evaluator
│   └── README.md                     # eval commands & details
├── requirements.txt
└── .env.example
```

## How tracing works

LangGraph runs on LangChain runnables, so a single OpenInference instrumentor
captures the whole graph (agent reasoning, every LLM call, every tool call, and
the knowledge-base retrieval) — in just two lines:

```python
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    space_id=ARIZE_SPACE_ID,
    api_key=ARIZE_API_KEY,
    project_name="arize-singapore-workshop",
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

## How evals work

Datasets and experiments use the unified Arize Python client (`arize` v8+):

```python
from arize import ArizeClient

client = ArizeClient(api_key=ARIZE_API_KEY)
client.datasets.create(name="sunrise-support-golden", space=SPACE_ID, examples=[...])

# Run the agent over the dataset and score each row with your evaluators.
client.experiments.run(name="gpt-4o-mini", dataset="sunrise-support-golden",
                       space=SPACE_ID, task=task, evaluators=[tool_selection, groundedness])
```

Run two experiments over the same dataset (e.g. `gpt-4o-mini` vs
`gpt-4o`) to compare them in Arize's Experiment Comparison view.
