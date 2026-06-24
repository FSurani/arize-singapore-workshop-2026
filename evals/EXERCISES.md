# Exercise: add two more evals as Online Evals (in the Arize UI)

In Section 4 the notebook runs two evaluators **offline** against the golden
dataset. In this exercise you'll add two more — but as **online, LLM-as-a-judge
evaluators** that score your agent's **live traces** continuously, from the
Arize UI. Online traffic has **no ground-truth columns**, so each judge reasons
from the trace itself (the customer's message + the agent's reply):

| Offline eval (dataset) | Online version (live traces) | How you'll build it |
|------------------------|------------------------------|---------------------|
| `correctness` | **response-correctness** — is the final reply accurate & helpful? | New Eval Task (UI) |
| `escalation_appropriate` | **escalation-appropriateness** — did it escalate (or not) appropriately? | **Alyx** (Enterprise) |

## Prerequisites

- Traces in your project — run the notebook through **Section 6** (live chat) so
  `arize-singapore-workshop` has real traffic to score.
- An **OpenAI AI integration** in your space for the LLM judge
  (**Settings → AI Providers/Integrations → add OpenAI**); use `gpt-4o-mini`.

> On this LangGraph agent, at **trace** scope the trace attributes are LangChain
> **messages JSON**, not bare text:
> - `attributes.input.value` → `{"messages": [{"type": "human", ... "content": "<customer message>"}]}`
> - `attributes.output.value` → the **full conversation state**: every turn
>   (human, the tool-calling AI turn, the tool result, and the **final** AI reply),
>   so the `escalate_to_human` tool call is visible here when the agent escalates.

---

## Evaluator 1 — `response-correctness` (build it in the UI)

In Arize you build the **evaluator and the task together** in one flow.

1. Left sidebar → **Evaluators** → **New Eval Task** (you can also start from the
   Projects page or from within a span).
2. **Name** the task and set the **data source** to your project
   (`arize-singapore-workshop`).
3. Click **Add Evaluator** → **LLM-as-a-Judge**, then configure:
   - **Judge model / provider:** your OpenAI integration, `gpt-4o-mini`.
   - **Labels:** `correct` / `incorrect`; explanations **on**.
   - **Column mappings:** `{{input}}` → `attributes.input.value`,
     `{{output}}` → `attributes.output.value`.
   - **Template:**

```
You are evaluating a customer-support agent for Sunrise Outfitters, an online outdoor retailer.

Customer message (a JSON "messages" object):
{{input}}

The agent's full run, as a JSON list of messages. The LAST "ai" message with
non-empty content is the reply shown to the customer:
{{output}}

Looking at that final assistant reply, is it accurate and genuinely helpful for
the customer's request, with no made-up order details or policies?
Answer "correct" if it is, or "incorrect" if it is inaccurate, unhelpful, or ignores the request.
```

4. Set **granularity** to **Trace**, **cadence** to **Run continuously on new
   data**, **sampling** to **100%**, then click **Create Task**.

---

## Evaluator 2 — `escalation-appropriateness` (build it with Alyx)

> **Requires the Arize Enterprise plan.** [Alyx](https://arize.com/docs/ax/alyx)
> is Arize's built-in AI assistant — describe the evaluator in plain English and
> it creates the LLM-as-a-judge evaluator and the continuous task for you.

Open **Alyx** (the assistant panel in the Arize UI) and drive it in two short,
high-level steps — let Alyx figure out the details.

**Step 1 — create the eval:**

```
Create an eval that checks whether the support agent escalated to a human
appropriately.
```

**Step 2 — run it on the project:**

```
Run it on the arize-singapore-workshop project, continuously on new traces and
backfill the existing ones.
```

From those two asks, Alyx works out the rest — it proposes an LLM-as-a-judge
with a suitable prompt, labels (e.g. appropriate / inappropriate), the judge
model, and the column mappings to your trace attributes, then creates the
continuous task and a backfill run. Review what it drafts and refine in natural
language if needed. This is a nice way to show Alyx reasoning through the setup
rather than hand-specifying everything.

---

## Verify

- Chat with the agent (Section 6) to generate new traces.
- Open the **`arize-singapore-workshop`** project → **Traces**. Within a minute or
  two each trace shows both eval columns: `response-correctness`
  (`correct`/`incorrect`) and `escalation-appropriateness`
  (`appropriate`/`inappropriate`), with explanations.
- Check task health under **Evaluators → Running Tasks → View Logs**
  (and **View Traces** to jump to the scored spans).

## Doing it in code instead

The `ax` CLI scripts the same flow — see
[`setup_online_eval.py`](setup_online_eval.py), which creates the AI integration,
a template evaluator, and a continuous trace-level task in one command.
