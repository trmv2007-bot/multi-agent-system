"""Grade claims against the source text actually fetched for them.

The validator makes one model call per pass: every claim in the pass goes out in
a single prompt, with the material its source produced, and comes back as a
verdict (``supported`` / ``unsupported`` / ``contradicted``). This module holds
the prompt, the verdict reader over the shared JSON repair path, and the
vocabulary the writer and the graph read that grade with; the model call itself
lives in :class:`src.agents.ValidatorAgent`.
"""

from collections.abc import Iterable
from typing import Any

from src.structured import json_payloads

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"
VERDICTS = (SUPPORTED, UNSUPPORTED, CONTRADICTED)

STATUS_BY_VERDICT: dict[str, str] = {
    SUPPORTED: "verified",
    UNSUPPORTED: "unverified",
    CONTRADICTED: "contradicted",
}
"""The status a verdict is recorded as, so readers of the old field keep working."""

SEVERITY_BY_VERDICT: dict[str, str] = {
    SUPPORTED: "low",
    UNSUPPORTED: "medium",
    CONTRADICTED: "high",
}

CONFIDENCE_BY_VERDICT: dict[str, float] = {
    SUPPORTED: 0.85,
    UNSUPPORTED: 0.4,
    CONTRADICTED: 0.1,
}

VERDICT_MAX_TOKENS = 4000
"""Budget for one verdict call.

The gateway pool routes to reasoning models, which spend most of a small budget
invisibly and then return empty content with ``finish_reason="length"``: with
3000 tokens both passes of one measured run came back empty. Below this the
whole pass degrades to unverified for no reason the sources justify.
"""

VERDICT_EXTRA_BODY: dict[str, object] = {"reasoning_effort": "none"}
"""Verdict calls ask the gateway to skip reasoning.

Pooled models are reasoning models: asked to think, they spend the whole
completion budget on reasoning and return empty content with
``finish_reason="length"``. Comparing a claim against a page of text is
mechanical, and with reasoning off the same five-claim prompt came back in about
300 tokens and two seconds, every time.
"""

SOURCE_CHARS = 1800
"""How much of one source's fetched text goes into the verdict prompt."""

CLAIM_CHARS = 500
"""How much of one claim goes into the verdict prompt."""


def claim_key(item: dict[str, Any]) -> str:
    """The text a verdict is matched back to its finding by.

    Both the validator (writing a record) and the writer (reading one) key on
    this, so a verdict attaches to the exact claim it graded.
    """
    text = item.get("content") or item.get("insight") or ""
    return str(text)[:200]


def verification_prompt(claims: list[dict[str, Any]]) -> str:
    """Build the one prompt that grades every claim in a pass."""
    blocks = []
    for index, item in enumerate(claims):
        source_text = str(item.get("source_content") or "").strip()
        if not source_text:
            source_text = "(no source text was fetched for this claim)"
        blocks.append(
            f"CLAIM {index}\n"
            f"text: {str(item.get('claim') or '')[:CLAIM_CHARS]}\n"
            f"source: {item.get('source') or 'unknown'}\n"
            f"source text: {source_text[:SOURCE_CHARS]}"
        )

    return (
        "Check every claim below against the source text supplied for it.\n"
        f"- {SUPPORTED}: the source text states the claim.\n"
        f"- {CONTRADICTED}: the source text says something incompatible with it.\n"
        f"- {UNSUPPORTED}: the source text is silent, off-topic, or does not cover it.\n"
        "Never mark a claim supported on your own knowledge: only the source text "
        "counts.\n\n" + "\n\n".join(blocks) + "\n\nReply with JSON only, one entry per claim: "
        '[{"index": 0, "verdict": "supported|unsupported|contradicted", '
        '"reason": "one line"}]'
    )


def parse_verdicts(reply: str, count: int) -> list[dict[str, str] | None]:
    """Read one verdict per claim out of the model's reply.

    An empty reply, a reply that is not JSON, or a missing entry leaves that
    slot ``None``. The caller records ``None`` as unverified: a reply the parser
    could not read is never agreement. The reading itself is the shared repair
    path, so a verdict that is only legible once its dropped quotes are put back
    still counts.
    """
    return verdicts_from(json_payloads(reply or ""), count)


def verdicts_from(payloads: Iterable[Any], count: int) -> list[dict[str, str] | None]:
    """Match parsed payloads to claims, newest reading that covers one winning."""
    if count <= 0:
        return []

    for payload in payloads:
        items: Any = payload
        if isinstance(items, dict):
            wrapped = items.get("verdicts") or items.get("results")
            items = wrapped if isinstance(wrapped, list) else [items]
        if not isinstance(items, list):
            continue
        slots = _read_payload(items, count)
        if any(slot is not None for slot in slots):
            return slots
    return [None] * count


def verdicts_of(
    validation_results: list[dict[str, Any]],
    turn: str = "",
    scope: str | None = None,
) -> dict[str, str]:
    """Map each claim's text to its verdict, newest record winning.

    ``turn`` limits the map to one run's grade; ``scope`` limits it to findings
    or analysis. Records with no verdict (an older run, or the build graph's
    review notes) are left out, so a caller cannot read them as a grade.
    """
    found: dict[str, str] = {}
    for record in reversed(validation_results):
        if turn and record.get("turn") != turn:
            continue
        if scope and record.get("scope") != scope:
            continue
        claim = str(record.get("claim", ""))
        verdict = str(record.get("verdict", ""))
        if claim and verdict in VERDICTS:
            found[claim] = verdict
    return found


def _read_payload(payload: list[Any], count: int) -> list[dict[str, str] | None]:
    """Read one candidate payload into one slot per claim."""
    slots: list[dict[str, str] | None] = [None] * count
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index", item.get("claim_index"))
        if raw_index is None:
            # A lone claim needs no index; the reply can only be about it.
            if count != 1:
                continue
            raw_index = 0
        try:
            index = int(str(raw_index))
        except ValueError:
            continue
        if not 0 <= index < count or slots[index] is not None:
            continue
        verdict = str(item.get("verdict") or item.get("status") or "").strip().lower()
        if verdict not in VERDICTS:
            continue
        slots[index] = {
            "verdict": verdict,
            "reason": str(item.get("reason", "")).strip()[:300],
        }
    return slots
