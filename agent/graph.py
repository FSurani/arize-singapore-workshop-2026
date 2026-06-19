"""LangGraph customer-support agent for Sunrise Outfitters.

Uses LangGraph's prebuilt ReAct agent with Gemini (ChatGoogleGenerativeAI) and
the support tools. Because LangGraph runs on LangChain runnables, a single
LangChain OpenInference instrumentor (set up in the notebook / ``app.py``)
captures the graph, the LLM calls, and every tool invocation - including the
knowledge-base retrieval - as spans in Arize.
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agent.tools import SUPPORT_TOOLS

SYSTEM_PROMPT = (
    "You are Sunny, the customer-support assistant for Sunrise Outfitters, an "
    "online outdoor-apparel retailer based in Singapore. Be warm, concise, and "
    "helpful.\n\n"
    "Guidelines:\n"
    "- Use `lookup_order` to get real order details before answering; never "
    "invent order statuses, tracking numbers, or prices.\n"
    "- For refund or return questions, check eligibility with "
    "`check_refund_eligibility` before promising anything.\n"
    "- For policy questions (shipping, returns, sizing, warranty, payments), "
    "use `search_knowledge_base` and answer grounded in the retrieved passages. "
    "If the knowledge base does not cover it, say so rather than guessing.\n"
    "- Escalate with `escalate_to_human` when the customer is upset, or the "
    "request is out of scope or high risk (e.g. billing disputes, legal "
    "complaints, anything the tools cannot resolve). Do not escalate routine "
    "questions you can already answer.\n"
    "- Keep replies friendly and under ~120 words."
)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def build_agent(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    prompt: str = SYSTEM_PROMPT,
    tools=SUPPORT_TOOLS,
):
    """Build and return the compiled LangGraph ReAct agent.

    Args:
        model: Gemini chat model name.
        temperature: Sampling temperature.
        prompt: System prompt. Override to compare prompt variants in experiments.
        tools: Tool list. Override (e.g. retrieval off) to compare variants.

    Returns:
        A compiled LangGraph agent that accepts ``{"messages": [...]}``.
    """
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
    return create_react_agent(llm, tools=tools, prompt=prompt)


def message_text(content) -> str:
    """Flatten a LangChain message ``content`` to plain text.

    Gemini "thinking" models can return ``content`` as a list of parts
    (e.g. ``[{"type": "text", "text": "..."}]``) instead of a string. This
    normalizes both shapes to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts).strip()
    return str(content)


def run_agent(agent, user_message: str) -> str:
    """Run one turn of the agent and return the final assistant reply text.

    Args:
        agent: A compiled agent from :func:`build_agent`.
        user_message: The customer's message.

    Returns:
        The agent's final text reply.
    """
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})
    messages = result["messages"]
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            text = message_text(message.content)
            if text:
                return text
    return message_text(messages[-1].content) if messages else ""


def run_agent_chat(agent, history: list[dict]) -> str:
    """Run the agent with full conversation history so it remembers the chat.

    Args:
        agent: A compiled agent from :func:`build_agent`.
        history: Chat history as a list of ``{"role", "content"}`` dicts
            (Gradio "messages" format), ending with the latest user message.

    Returns:
        The agent's final text reply.
    """
    messages = []
    for turn in history:
        content = turn.get("content", "")
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    result = agent.invoke({"messages": messages})
    out = result["messages"]
    for message in reversed(out):
        if isinstance(message, AIMessage) and message.content:
            text = message_text(message.content)
            if text:
                return text
    return message_text(out[-1].content) if out else ""
