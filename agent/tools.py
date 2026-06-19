"""Customer-support tools for Sunrise Outfitters.

The order and refund tools use in-memory fake data, standing in for a real
order-management API so the workshop needs no external services. The FAQ tool
has been upgraded to **real retrieval** over a small knowledge base (see
``agent/retrieval.py``), so the agent grounds policy answers in retrieved
context - and that retrieval shows up as its own span in Arize.
"""

from __future__ import annotations

from langchain_core.tools import tool

from agent.retrieval import search_knowledge_base as _search_kb

# --- Fake order-management backend ------------------------------------------

_ORDERS: dict[str, dict] = {
    "A1001": {
        "status": "shipped",
        "item": "Trailblazer Rain Jacket (M, Forest Green)",
        "carrier": "DHL",
        "tracking": "DHL-SG-77123",
        "ordered_days_ago": 3,
        "delivered": False,
        "price_usd": 129.00,
    },
    "A1002": {
        "status": "delivered",
        "item": "Summit Hiking Boots (US 9)",
        "carrier": "SingPost",
        "tracking": "SP-99812",
        "ordered_days_ago": 20,
        "delivered": True,
        "price_usd": 159.00,
    },
    "A1003": {
        "status": "processing",
        "item": "Coastline Windbreaker (L, Navy)",
        "carrier": None,
        "tracking": None,
        "ordered_days_ago": 1,
        "delivered": False,
        "price_usd": 89.00,
    },
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up the current status and details of a customer order by its ID.

    Args:
        order_id: The order identifier, e.g. "A1001".
    """
    order = _ORDERS.get(order_id.strip().upper())
    if not order:
        return (
            f"No order found with ID '{order_id}'. Ask the customer to "
            "double-check the ID from their confirmation email."
        )
    parts = [
        f"Order {order_id.upper()}: {order['item']}",
        f"Status: {order['status']}",
        f"Placed: {order['ordered_days_ago']} day(s) ago",
        f"Price: ${order['price_usd']:.2f}",
    ]
    if order["tracking"]:
        parts.append(f"Carrier: {order['carrier']} (tracking {order['tracking']})")
    return ". ".join(parts) + "."


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether an order is eligible for a refund under the return policy.

    Policy: delivered items can be refunded within 30 days of the order date.
    Undelivered items can be cancelled for a full refund while still processing.

    Args:
        order_id: The order identifier, e.g. "A1002".
    """
    order = _ORDERS.get(order_id.strip().upper())
    if not order:
        return f"No order found with ID '{order_id}', so eligibility cannot be checked."

    if order["status"] == "processing":
        return (
            f"Order {order_id.upper()} is still processing and can be cancelled "
            "now for a full refund."
        )
    if order["delivered"]:
        if order["ordered_days_ago"] <= 30:
            return (
                f"Order {order_id.upper()} was delivered and is within the 30-day "
                "window, so it IS eligible for a full refund."
            )
        return (
            f"Order {order_id.upper()} was delivered more than 30 days ago, so it "
            "is NOT eligible for a standard refund. Offer store credit instead."
        )
    return (
        f"Order {order_id.upper()} has shipped but is not yet delivered. The "
        "customer can refuse delivery or return it within 30 days of arrival."
    )


@tool
def search_knowledge_base(query: str) -> str:
    """Search Sunrise Outfitters' policy knowledge base for the answer to a question.

    Covers shipping, returns/refunds, sizing & fit, warranty/repairs, and
    payments. Returns the most relevant policy passages so you can answer
    grounded in the retrieved context. Prefer this over guessing policy details.

    Args:
        query: A natural-language question or keywords from the customer.
    """
    return _search_kb(query)


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human support agent and open a ticket.

    Use this when the customer is upset, the request is out of scope or high
    risk (e.g. billing disputes, legal/complaints), or the other tools cannot
    resolve it. Do NOT escalate routine questions the tools can already answer.

    Args:
        reason: A short summary of why the issue needs a human.
    """
    ticket_id = f"TCK-{abs(hash(reason)) % 90000 + 10000}"
    return (
        f"Escalated to a human agent. Ticket {ticket_id} created with note: "
        f"'{reason}'. A specialist will reply by email within 24 hours."
    )


SUPPORT_TOOLS = [
    lookup_order,
    check_refund_eligibility,
    search_knowledge_base,
    escalate_to_human,
]
