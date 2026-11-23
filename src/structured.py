"""One way to get JSON out of a model.

The provider pool breaks JSON in two measured ways: it writes object values
without quotes, and a reasoning model that spends its whole budget returns
``content: ""`` with the answer left in ``reasoning``. Each caller used to grow
its own regex against that, and every caller that had not yet done so degraded
silently instead of saying so.

This module is the one path: ask for JSON with reasoning off and a generous
budget, treat an empty reply as a failure, parse strictly, then repair only what
is unambiguous (code fences, one balanced object inside prose, unquoted values,
trailing commas), retry once with a blunter instruction, and finally raise
:class:`StructuredOutputError` carrying the raw reply so the caller decides
whether to degrade or stop.

Every call, repair, retry and failure is counted in :data:`STRUCTURED`, published
on the activity bus, and readable through
:func:`src.observability.structured_output_stats`: a provider that has started
mangling JSON is visible instead of being absorbed by fallbacks.
"""

import json
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from src.activity import activity

logger = logging.getLogger(__name__)

JSON_MAX_TOKENS = 4000
"""Budget for one JSON call.

The gateway pool routes to reasoning models, which spend most of a small budget
invisibly and then return empty content with ``finish_reason="length"``: with
3000 tokens both passes of one measured run came back empty. Below this the whole
call degrades for no reason the prompt justifies.
"""

JSON_EXTRA_BODY: dict[str, object] = {"reasoning_effort": "none"}
"""JSON calls ask the gateway to skip reasoning.

Pooled models are reasoning models: asked to think, they spend the whole
completion budget on reasoning and return empty content with
``finish_reason="length"``. Reading a reply into an object is mechanical, and
with reasoning off the same five-claim prompt came back in about 300 tokens and
two seconds, every time.
"""

RETRY_NUDGE = (
    "Your last reply could not be read as JSON. Answer again with the JSON only: "
    "no prose, no code fences, and no text before or after it."
)

_FENCE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_LITERALS = ("true", "false", "null")
"""The bare words JSON already reads as values."""


class StructuredOutputError(RuntimeError):
    """A reply that could not be read as JSON, carrying what the model said.

    The raw reply rides along so a caller that degrades can still log what the
    provider sent, and a caller that stops can show it to whoever reads the run.
    """

    def __init__(
        self,
        message: str,
        *,
        raw: str = "",
        site: str = "",
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.site = site
        self.attempts = attempts


class StructuredCounters:
    """How the JSON calls have gone since this process started.

    ``calls`` counts model calls, so a provider returning empty content shows up
    as ``empty`` tracking ``calls``; ``repaired`` means a reply was only readable
    after repair; ``failed`` counts calls that raised. Nothing here is per-site
    state, so it survives parallel agents without coordination.
    """

    FIELDS = ("calls", "parsed", "repaired", "empty", "unreadable", "retried", "failed")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = dict.fromkeys(self.FIELDS, 0)

    def add(self, **deltas: int) -> dict[str, int]:
        """Count these outcomes, and return every count so far."""
        with self._lock:
            for field, delta in deltas.items():
                self._counts[field] = self._counts.get(field, 0) + delta
            return dict(self._counts)

    def snapshot(self) -> dict[str, int]:
        """Every count so far."""
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        """Start counting again, for a fresh run or a test."""
        with self._lock:
            self._counts = dict.fromkeys(self.FIELDS, 0)


STRUCTURED = StructuredCounters()

Expect = type | tuple[type, ...] | None

ModelSender = Callable[[list[dict[str, Any]]], Awaitable[Any]]
"""One model call: takes the messages, returns the reply."""


def reply_text(response: Any) -> str:
    """The text a model replied with, whatever shape the client hands back."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", "")) for block in content
        )
    return "" if content is None else str(content)


def json_payloads(text: str) -> Iterator[Any]:
    """Every JSON value a reply could be holding, the most faithful reading first."""
    for payload, _repaired in _readings(text or ""):
        yield payload


def read_json(text: str, *, expect: Expect = None) -> tuple[Any, bool]:
    """Read a reply into JSON, and say whether reading it took a repair.

    ``expect`` restricts which JSON shapes count, so a caller asking for a list
    skips over any object that happens to come first in the reply.
    """
    for payload, repaired in _readings(text or ""):
        if expect is None or isinstance(payload, expect):
            return payload, repaired
    raise StructuredOutputError(_unreadable(text, expect), raw=text or "")


def parse_json(text: str, *, expect: Expect = None) -> Any:
    """Read a reply into JSON, or ``None`` when nothing usable is in it.

    For the callers whose documented failure behaviour is to carry on with a
    fallback; a caller that would rather stop uses :func:`read_json`.
    """
    try:
        payload, _repaired = read_json(text, expect=expect)
    except StructuredOutputError:
        return None
    return payload


async def call_json(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    retries: int = 1,
    expect: Expect = None,
    site: str = "structured",
) -> Any:
    """Ask a model for JSON and return what it said, parsed.

    ``llm`` is a chat model, which this binds to a generous budget with reasoning
    off, or any async callable taking the messages, which is then responsible for
    that tuning itself. Empty content is a failure, never a result. A reply that
    cannot be read is retried ``retries`` times with an explicit "return only
    JSON" instruction, so a provider that mangles JSON once does not cost the
    caller its whole answer. A reply that still cannot be read raises
    :class:`StructuredOutputError`, and ``site`` names the caller in the log.
    """
    conversation = list(messages)
    for attempt in range(retries + 1):
        response = await _send(llm, conversation)
        text = reply_text(response)
        empty = not text.strip()
        problem = f"empty content{_why_empty(response)}" if empty else ""
        if not empty:
            try:
                payload, repaired = read_json(text, expect=expect)
            except StructuredOutputError as error:
                problem = str(error)
            else:
                counts = STRUCTURED.add(calls=1, parsed=int(not repaired), repaired=int(repaired))
                if repaired:
                    logger.warning("%s: repaired a malformed JSON reply", site)
                activity.structured(counts, site)
                return payload

        last = attempt == retries
        counts = STRUCTURED.add(
            calls=1,
            empty=int(empty),
            unreadable=int(not empty),
            retried=int(not last),
            failed=int(last),
        )
        activity.structured(counts, site)
        if last:
            raise StructuredOutputError(
                f"{site}: no readable JSON after {attempt + 1} attempt(s): {problem}",
                raw=text,
                site=site,
                attempts=attempt + 1,
            )
        logger.warning("%s: %s; asking again for JSON only", site, problem)
        conversation = [*conversation, {"role": "user", "content": RETRY_NUDGE}]

    raise AssertionError("the loop returns or raises on its last attempt")


async def _send(llm: Any, messages: list[dict[str, Any]]) -> Any:
    """Call the model once: a chat model gets the JSON tuning, a callable is used as is."""
    if not hasattr(llm, "ainvoke"):
        return await llm(messages)
    bind = getattr(llm, "bind", None)
    bound = bind(max_tokens=JSON_MAX_TOKENS, extra_body=JSON_EXTRA_BODY) if callable(bind) else llm
    return await bound.ainvoke(messages)


def _why_empty(response: Any) -> str:
    """Why a reply carried no content, when the provider says."""
    metadata = getattr(response, "response_metadata", None)
    reason = ""
    if isinstance(metadata, dict):
        reason = str(metadata.get("finish_reason") or metadata.get("stop_reason") or "")
    extras = getattr(response, "additional_kwargs", None)
    reasoned = isinstance(extras, dict) and bool(
        extras.get("reasoning_content") or extras.get("reasoning")
    )
    described = ", ".join(
        part for part in (reason, "the answer went into reasoning" if reasoned else "") if part
    )
    return f" ({described})" if described else ""


def _unreadable(text: str, expect: Expect) -> str:
    """One line saying what was wanted and what arrived instead."""
    names = () if expect is None else (expect if isinstance(expect, tuple) else (expect,))
    wanted = " " + " or ".join(getattr(item, "__name__", str(item)) for item in names)
    head = " ".join((text or "").split())[:120]
    return f"no readable JSON{wanted} in the reply ({len(text or '')} chars): {head!r}"


def _readings(text: str) -> Iterator[tuple[Any, bool]]:
    """Every value the reply could be holding, faithful before repaired."""
    seen: set[str] = set()
    for source in [*_fenced(text), text]:
        for chunk in _containers(source):
            if chunk in seen:
                continue
            seen.add(chunk)
            for candidate, repaired in _repairs(chunk):
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                yield payload, repaired
                if isinstance(payload, dict):
                    for value in payload.values():
                        if isinstance(value, dict | list):
                            yield value, repaired
                break


def _fenced(text: str) -> list[str]:
    """The body of every fenced block, in the order they appear."""
    return [match.group(1).strip() for match in _FENCE.finditer(text)]


def _containers(text: str) -> Iterator[str]:
    """Balanced ``[...]`` and ``{...}`` spans that begin outside every other span.

    A span nested in one that never closes is not offered at all: half of a
    truncated reply must not come back looking like a whole one.
    """
    starts: list[int] = []
    expected: list[str] = []
    for index, char in enumerate(text):
        if char in "[{":
            starts.append(index)
            expected.append("]" if char == "[" else "}")
        elif char in "]}" and expected and expected[-1] == char:
            start = starts.pop()
            expected.pop()
            if not expected:
                yield text[start : index + 1]


def _repairs(chunk: str) -> Iterator[tuple[str, bool]]:
    """A chunk's readings, the untouched one first, each with "was repaired"."""
    quoted = _quote_unquoted_values(chunk)
    comma = _drop_trailing_commas(chunk)
    offered = (
        (chunk, False),
        (quoted, True),
        (comma, True),
        (_drop_trailing_commas(quoted), True),
        (_quote_unquoted_values(comma), True),
    )
    seen: set[str] = set()
    for candidate, repaired in offered:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate, repaired


def _quote_unquoted_values(chunk: str) -> str:
    """Quote object values a model left bare: not JSON, but unambiguous.

    Strings are copied verbatim, so a colon or comma inside one never reaches
    this. Numbers, booleans, ``null``, objects and arrays are left alone.
    """
    out: list[str] = []
    index = 0
    length = len(chunk)
    while index < length:
        char = chunk[index]
        if char == '"':
            end = _string_end(chunk, index)
            out.append(chunk[index : end + 1])
            index = end + 1
            continue
        out.append(char)
        index += 1
        if char != ":":
            continue
        start = index
        while start < length and chunk[start] in " \t":
            start += 1
        if start >= length or _reads_as_value(chunk, start):
            continue
        end = _value_end(chunk, start)
        bare = chunk[start:end].strip()
        if bare.endswith('"') and not bare.startswith('"'):
            bare = bare.rstrip('"').rstrip()
        index = end
        if bare:
            out.append(' "' + bare.replace('"', '\\"') + '"')
    return "".join(out)


def _string_end(chunk: str, start: int) -> int:
    """The index of the quote that closes the string opening at ``start``."""
    index = start + 1
    while index < len(chunk):
        if chunk[index] == "\\":
            index += 2
            continue
        if chunk[index] == '"':
            return index
        index += 1
    return len(chunk) - 1


def _reads_as_value(chunk: str, start: int) -> bool:
    """Whether the value at ``start`` is already something JSON reads."""
    char = chunk[start]
    if char in '"[{' or char == "-" or char.isdigit():
        return True
    return any(chunk.startswith(word, start) for word in _LITERALS)


def _value_end(chunk: str, start: int) -> int:
    """Where an unquoted value that begins at ``start`` ends.

    A ``}`` or ``]`` always ends it. A comma ends it only when what follows
    starts another key or element, so a thousands separator inside the value
    (``389,444.``) stays part of it.
    """
    index = start
    while index < len(chunk):
        char = chunk[index]
        if char in "}]":
            return index
        if char == ",":
            following = index + 1
            while following < len(chunk) and chunk[following] in " \t\r\n":
                following += 1
            if following >= len(chunk) or chunk[following] in '"[{]':
                return index
        index += 1
    return len(chunk)


def _drop_trailing_commas(chunk: str) -> str:
    """Remove a comma left before a closing bracket, which is not JSON."""
    return _TRAILING_COMMA.sub(r"\1", chunk)
