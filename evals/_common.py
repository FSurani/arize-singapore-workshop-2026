"""Shared task + evaluators for the offline eval and experiment scripts.

Kept in one place so ``offline_evals.py`` and ``run_experiment.py`` stay in
sync. Nothing here talks to Arize directly - these are plain functions the
Arize experiment runner calls for each dataset row.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from arize.experiments import EvaluationResult

from agent.graph import message_text

RETRIEVAL_TOOL = "search_knowledge_base"
ESCALATION_TOOL = "escalate_to_human"


def agent_response(agent, message: str) -> dict:
    """Run the agent once and return the reply plus which tools/context it used.

    Returns a dict with ``reply`` (str), ``tools_used`` (list[str]), and
    ``context`` (str, the concatenated knowledge-base passages it retrieved).
    """
    result = agent.invoke({"messages": [HumanMessage(content=message)]})
    messages = result["messages"]

    tools_used: list[str] = []
    context_chunks: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in getattr(m, "tool_calls", None) or []:
                tools_used.append(call["name"])
        elif isinstance(m, ToolMessage) and m.name == RETRIEVAL_TOOL:
            context_chunks.append(str(m.content))

    reply = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            reply = message_text(m.content)
            if reply:
                break

    return {
        "reply": reply,
        "tools_used": tools_used,
        "context": "\n\n".join(context_chunks),
    }


def make_task(agent):
    """Build the experiment task: run the agent and return a JSON-string output.

    The output is JSON so evaluators can inspect the reply, the tools the agent
    chose, and the retrieved context - all visible on the run in Arize.
    """

    def task(dataset_row) -> str:
        message = dataset_row.get("input", "")
        return json.dumps(agent_response(agent, message))

    return task


def _parse(output) -> dict:
    try:
        data = json.loads(output) if isinstance(output, str) else dict(output)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"reply": str(output), "tools_used": [], "context": ""}
    return data


# --- Code-based evaluators --------------------------------------------------

def tool_selection(output, dataset_row) -> EvaluationResult:
    """Did the agent call the tool we expected for this example?"""
    expected = dataset_row.get("expected_tool")
    used = _parse(output).get("tools_used", [])
    if not expected:
        return EvaluationResult(label="not_applicable", explanation="No specific tool expected.")
    correct = expected in used
    return EvaluationResult(
        score=int(correct),
        label="correct" if correct else "incorrect",
        explanation=f"Expected '{expected}'; agent used {used or 'no tools'}.",
    )


def escalation_appropriate(output, dataset_row) -> EvaluationResult:
    """Did the agent escalate iff escalation was expected?"""
    expected = bool(dataset_row.get("expect_escalation"))
    escalated = ESCALATION_TOOL in _parse(output).get("tools_used", [])
    correct = expected == escalated
    return EvaluationResult(
        score=int(correct),
        label="correct" if correct else "incorrect",
        explanation=f"Expected escalation={expected}, agent escalated={escalated}.",
    )


# --- LLM-as-a-judge evaluators (Gemini) ------------------------------------

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")


def _judge(system: str, user: str) -> tuple[str, str]:
    """Tiny LLM-judge helper using Gemini. Returns (label, explanation)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    text = message_text(resp.content).strip()
    first = text.splitlines()[0].lower() if text else ""
    return first, text


def correctness(output, dataset_row) -> EvaluationResult:
    """LLM judge: does the reply match the expected answer?"""
    data = _parse(output)
    expected = dataset_row.get("expected_output", "")
    label, expl = _judge(
        "You grade a customer-support reply against the expected answer. "
        "Reply with 'correct' or 'incorrect' on the first line, then one sentence why.",
        f"Expected answer:\n{expected}\n\nAgent reply:\n{data.get('reply', '')}",
    )
    correct = "correct" in label and "incorrect" not in label
    return EvaluationResult(score=int(correct), label="correct" if correct else "incorrect", explanation=expl)


def groundedness(output, dataset_row) -> EvaluationResult:
    """LLM judge: is the reply supported by the retrieved context (no hallucination)?

    If the row is a policy question (retrieval expected) but the agent answered
    without retrieving anything, that is ungrounded (score 0). For rows where
    retrieval is not expected (orders, escalation), groundedness is not
    applicable. Always emitting a score on the policy rows also keeps the
    experiment's score column well-typed for Arize.
    """
    data = _parse(output)
    context = data.get("context", "")
    expects_retrieval = dataset_row.get("expected_tool") == RETRIEVAL_TOOL
    if not context:
        if expects_retrieval:
            return EvaluationResult(
                score=0,
                label="ungrounded",
                explanation="A policy question was answered without retrieving any knowledge-base context.",
            )
        return EvaluationResult(label="not_applicable", explanation="Retrieval not expected for this row.")
    label, expl = _judge(
        "You check whether a support reply is grounded in the retrieved policy context. "
        "Reply 'grounded' if every claim is supported by the context, else 'hallucinated'. "
        "Put the label on the first line, then one sentence why.",
        f"Retrieved context:\n{context}\n\nAgent reply:\n{data.get('reply', '')}",
    )
    grounded = "grounded" in label and "hallucinated" not in label
    return EvaluationResult(score=int(grounded), label="grounded" if grounded else "hallucinated", explanation=expl)


# Default evaluator set used by both offline evals and experiments.
EVALUATORS = [tool_selection, escalation_appropriate, correctness, groundedness]
