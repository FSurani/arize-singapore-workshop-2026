"""Gradio chat app for the Sunrise Outfitters support agent.

A clean, Apple-inspired chat UI built on gr.Blocks (custom theme + CSS).

Run locally:

    cp .env.example .env   # fill in your keys
    pip install -r requirements.txt
    python app.py

Only GOOGLE_API_KEY is strictly required. If ARIZE_SPACE_ID and ARIZE_API_KEY
are also set, every conversation is traced into Arize.
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from agent.graph import build_agent, run_agent

load_dotenv()

EXAMPLES = [
    "Where is my order A1001?",
    "Can I get a refund on order A1002?",
    "How long does shipping take to Singapore?",
    "Order A1003 hasn't arrived and I'm frustrated. Help!",
]

# Apple-inspired theme: system font stack, generous radius, blue accent.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.gray,
    radius_size=gr.themes.sizes.radius_lg,
    font=[
        "-apple-system",
        "BlinkMacSystemFont",
        "SF Pro Text",
        "Inter",
        "system-ui",
        "sans-serif",
    ],
).set(
    body_background_fill="#f5f5f7",
    body_background_fill_dark="#000000",
    block_background_fill="#ffffff",
    block_border_width="0px",
    block_shadow="0 8px 30px rgba(0, 0, 0, 0.06)",
    button_primary_background_fill="#0071e3",
    button_primary_background_fill_hover="#0077ed",
    button_primary_text_color="#ffffff",
    button_large_radius="980px",
    button_small_radius="980px",
    input_radius="22px",
)

CSS = """
.gradio-container {max-width: 820px !important; margin: 0 auto !important;}
#app-header {text-align: center; padding: 22px 0 4px;}
#app-header .logo {font-size: 30px; line-height: 1;}
#app-header h1 {
  font-weight: 600; letter-spacing: -0.02em; margin: 8px 0 0;
  font-size: 26px; color: #1d1d1f;
}
#app-header p {color: #6e6e73; margin: 6px 0 0; font-size: 15px;}
/* chat bubbles: rounded, no borders */
.message-wrap .message, .bubble-wrap .message, div[class*="message"] {
  border-radius: 20px !important; border: none !important; box-shadow: none !important;
}
footer {display: none !important;}
.send-btn {min-width: 92px !important;}
"""

SUBTITLE = "AI support assistant - ask about orders, shipping, returns & sizing"


def _require_google_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "Google AI Studio API key, or export GOOGLE_API_KEY before running."
        )


def _setup_tracing() -> None:
    """Trace into Arize if keys are set. Same two lines as the notebook."""
    if not (os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY")):
        print("[tracing] Arize keys not set - running without tracing.")
        return
    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=os.getenv("ARIZE_PROJECT", "arize-singapore-workshop"),
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("[tracing] Arize tracing enabled.")


def build_demo(agent) -> gr.Blocks:
    """Build the Apple-styled chat UI bound to a compiled agent."""

    def on_submit(message: str, history: list):
        if not message or not message.strip():
            return "", history or []
        history = (history or []) + [{"role": "user", "content": message}]
        return "", history

    def on_reply(history: list):
        user_msg = history[-1]["content"]
        try:
            reply = run_agent(agent, user_msg)
        except Exception as exc:  # surface errors in the UI instead of crashing
            reply = f"Sorry, something went wrong: {exc}"
        return history + [{"role": "assistant", "content": reply}]

    with gr.Blocks(title="Sunrise Outfitters Support", fill_height=True,
                   theme=THEME, css=CSS) as demo:
        gr.HTML(
            '<div id="app-header">'
            '<div class="logo">&#9968;&#65039;</div>'
            "<h1>Sunrise Outfitters</h1>"
            f"<p>{SUBTITLE}</p>"
            "</div>"
        )
        chatbot = gr.Chatbot(
            type="messages",
            show_label=False,
            height=460,
            avatar_images=(None, "https://em-content.zobj.net/source/apple/391/sun_2600-fe0f.png"),
            placeholder="<div style='color:#86868b'>Say hello to Sunny &#128075;</div>",
        )
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Message Sunny...",
                autofocus=True,
                scale=8,
                container=False,
            )
            send = gr.Button("Send", variant="primary", scale=0, elem_classes="send-btn")
        gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one")
        clear = gr.Button("Clear conversation", variant="secondary", size="sm")

        msg.submit(on_submit, [msg, chatbot], [msg, chatbot]).then(on_reply, chatbot, chatbot)
        send.click(on_submit, [msg, chatbot], [msg, chatbot]).then(on_reply, chatbot, chatbot)
        clear.click(lambda: [], None, chatbot)

    return demo


if __name__ == "__main__":
    _require_google_key()
    _setup_tracing()
    demo = build_demo(build_agent())
    demo.launch()
