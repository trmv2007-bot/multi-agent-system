"""A scripted chat model, so the browser test needs no gateway and no key.

The end-to-end test drives the real page, the real HTTP and SSE routes and the
real graphs; only the model is scripted. ``DASHBOARD_FAKE_MODEL`` switches it
on, nothing in a normal run sets that variable, and every reply is fixed, so the
test can assert on answers and costs instead of on prose.

A reply is chosen by the asking agent's system prompt: each agent's prompt is
distinct, and the JSON sites answer with the shape their parser expects.
Anything unrecognised gets the plain answer, which is what the writer,
synthesizer and reporter paths read.
"""

import json
import os
from types import SimpleNamespace
from typing import Any

ENV_FLAG = "DASHBOARD_FAKE_MODEL"
_TRUTHY = {"1", "true", "yes", "on"}

ANSWER = (
    "The scripted model answered this turn, so the page, the event stream and "
    "the cost line were all exercised without a gateway."
)

CHARS_PER_TOKEN = 4
"""Rough characters to a token, which is close enough for a cost line."""

SECTION_HTML = (
    '<section class="container"><p class="eyebrow">Scripted</p>'
    "<h2>A section written by the test</h2>"
    "<p>This fragment exists so the preview has something to show.</p></section>"
)

THEME = {
    "feeling": "quiet and workmanlike",
    "palette": {
        "bg": "#0b0f16",
        "surface": "#131a26",
        "accent": "#f2b544",
        "text": "#eef2f8",
        "muted": "#93a2b8",
    },
    "radius": "10px",
    "display_size": "clamp(2.2rem, 6vw, 4rem)",
    "background_css": "background: radial-gradient(#182033, #0b0f16);",
    "extra_css": "",
    "motion_css": "",
    "extras_js": "",
}

SCRIPTS: tuple[tuple[str, str], ...] = (
    (
        "You are the orchestrator of a team",
        json.dumps(
            [
                {"task": "Summarise what the request is asking for", "tools": []},
                {"task": "Check the summary against the request", "tools": []},
            ]
        ),
    ),
    (
        "You plan research",
        json.dumps(
            [
                {"question": "What does the request need answered?", "why": "scope"},
                {"question": "What would a good answer contain?", "why": "quality"},
                {"question": "What is the current state of the art?", "why": "context"},
            ]
        ),
    ),
    ("You review research coverage", json.dumps([])),
    (
        "You plan one web page",
        json.dumps(
            [
                {"section": "hero", "brief": "What the page is for, in one line"},
                {"section": "features", "brief": "Three things it does"},
                {"section": "cta", "brief": "What to do next"},
            ]
        ),
    ),
    ("You review an assembled web page", json.dumps([])),
    ("You are the designer on this build", json.dumps(THEME)),
    ("Write one section of a web page as an HTML fragment", SECTION_HTML),
    (
        "You research what a web page needs",
        "The page needs a headline, three features and one call to action.",
    ),
)


def reply_for(system: str) -> str:
    """The scripted reply to whichever agent asked, chosen by its prompt."""
    for marker, reply in SCRIPTS:
        if marker in system:
            return reply
    return ANSWER


class ScriptedChatModel:
    """A chat model that answers from a script and never touches the network."""

    model = "scripted-model"

    def bind(self, **_options: Any) -> "ScriptedChatModel":
        """Accept request options; the script does not change with them."""
        return self

    def bind_tools(self, _tools: Any) -> "ScriptedChatModel":
        """Accept tools; the script never calls one."""
        return self

    async def ainvoke(self, messages: Any, **_kwargs: Any) -> Any:
        """Answer one prompt, reporting the tokens a provider would bill."""
        system = _system_of(messages)
        answer = reply_for(system)
        prompt_tokens = _tokens(system + _spoken_of(messages))
        completion_tokens = _tokens(answer)
        return SimpleNamespace(
            content=answer,
            tool_calls=None,
            response_metadata={"model_name": self.model, "finish_reason": "stop"},
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )


def scripted_enabled() -> bool:
    """Whether this process is the browser test's, with the model scripted out."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


def scripted_llm() -> ScriptedChatModel:
    """A fresh scripted model, so every turn starts from the same script."""
    return ScriptedChatModel()


def _content_of(message: Any) -> str:
    """One message's text, whichever shape it arrives in."""
    content = getattr(message, "content", message)
    if isinstance(content, dict):
        return str(content.get("content", ""))
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def _system_of(messages: Any) -> str:
    """The system prompt of a conversation, which is how an agent is recognised."""
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "system":
            return _content_of(message)
        if getattr(message, "type", "") == "system":
            return _content_of(message)
    return ""


def _spoken_of(messages: Any) -> str:
    """Everything the conversation carried besides the system prompt."""
    parts = []
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") != "system":
            parts.append(_content_of(message))
    return "\n".join(parts)


def _tokens(text: str) -> int:
    """A token count in the range a provider would report for this much text."""
    return max(1, len(text) // CHARS_PER_TOKEN)
