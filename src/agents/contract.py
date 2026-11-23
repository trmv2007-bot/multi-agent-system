"""The writer's output contract, measured instead of promised.

The writer's prompt asks for at most 180 words with no report scaffolding. A
prompt is not an enforcement mechanism: a pooled model answered that brief with
962 words and an "Executive Summary" heading. So the answer is measured after
it comes back, repaired once by a re-ask that names the violation, and then cut
deterministically to the cap, so nothing longer than the cap can leave the
writer even when the model ignores both requests.

The module also collects the URLs the run actually fetched, for the one
provenance footer the writer appends to the finished answer.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

WORD_CAP = 180
"""The word count the writer's prompt promises for a default answer."""

SOURCE_LIMIT = 5
"""How many fetched URLs the sources footer lists, one per domain."""

SCAFFOLD_LABELS = (
    "executive summary",
    "conclusions and recommendations",
    "conclusions & recommendations",
    "recommendations and conclusions",
    "key takeaways",
    "in conclusion",
    "in summary",
    "final answer",
    "next steps",
    "introduction",
    "conclusion",
    "conclusions",
    "recommendations",
    "takeaways",
    "overview",
    "summary",
    "background",
)
"""Headings that announce a report rather than answer the question."""

HEADING_WORDS = 6
"""How many words a line may hold and still read as a heading, not a sentence."""

_HEADING_MARK = re.compile(r"^#{1,6}\s+")
_BOLDED_LINE = re.compile(r"^(?:\*\*|__)(.+?)(?:\*\*|__)$")
_TOKEN = re.compile(r"\S+")
_URL = re.compile(r"https?://[^\s\"'<>)\]}]+")
_SENTENCE_BREAKS = ".!?\n"


def words_of(text: str) -> int:
    """Whitespace-separated tokens, which is what the prompt's cap counts."""
    return len(text.split())


def _heading_of(line: str) -> tuple[str, bool]:
    """The bare text of a line, and whether markdown marked it as a heading."""
    stripped = line.strip()
    decorated = _HEADING_MARK.match(stripped) is not None
    text = _HEADING_MARK.sub("", stripped)
    bolded = _BOLDED_LINE.fullmatch(text)
    if bolded:
        text, decorated = bolded.group(1), True
    return " ".join(text.strip().strip(":*-_# ").split()), decorated


def scaffolding_heading(text: str) -> str:
    """The scaffolding heading the answer opens with, or "" when it has none.

    Only the opening line counts: a heading later in the answer is something
    the model wrote, not padding in front of it. A line must look like a
    heading - marked up, or exactly one of the known labels - so a sentence
    that merely starts with "Summary" is left alone.
    """
    opening = next((line for line in text.splitlines() if line.strip()), "")
    label, decorated = _heading_of(opening)
    if not label or len(label.split()) > HEADING_WORDS:
        return ""
    normalized = label.lower()
    for scaffold in SCAFFOLD_LABELS:
        if normalized == scaffold or (decorated and normalized.startswith(f"{scaffold} ")):
            return label
    return ""


def strip_scaffolding(text: str) -> str:
    """Drop the scaffolding heading an answer opens with, keeping the answer."""
    if not scaffolding_heading(text):
        return text
    lines = text.splitlines()
    index = next(number for number, line in enumerate(lines) if line.strip())
    rest = lines[index + 1 :]
    while rest and not rest[0].strip():
        rest.pop(0)
    body = "\n".join(rest).strip()
    # A heading on its own is a broken answer, but it is not nothing.
    return body or text


def trim_to_cap(text: str, cap: int = WORD_CAP) -> str:
    """Cut an over-long answer to the cap at its last sentence boundary."""
    if words_of(text) <= cap:
        return text

    cut = 0
    for count, token in enumerate(_TOKEN.finditer(text), start=1):
        if count > cap:
            break
        cut = token.end()
    head = text[:cut]
    boundary = max(head.rfind(mark) for mark in _SENTENCE_BREAKS)
    # Only honour a boundary that keeps most of the room the cap allows; a
    # stray early full stop must not throw the answer away.
    if boundary > 0 and words_of(head[:boundary]) >= cap // 3:
        return head[: boundary + 1].strip()
    return head.rstrip(" ,;:.-") + "…"


def repair(text: str) -> str:
    """Hold an answer to the contract with no further model call."""
    return trim_to_cap(strip_scaffolding(text))


@dataclass(frozen=True)
class Violation:
    """How an answer broke the writer's contract."""

    words: int
    heading: str = ""

    def describe(self) -> str:
        """The violation in one clause, so a re-ask can name it concretely."""
        if self.words > WORD_CAP and self.heading:
            return f'{self.words} words with a "{self.heading}" heading'
        if self.words > WORD_CAP:
            return f"{self.words} words"
        return f'a "{self.heading}" heading'


def violation_of(text: str) -> Violation | None:
    """What broke the contract in an answer, or None when it kept it."""
    words = words_of(text)
    heading = scaffolding_heading(text)
    if words <= WORD_CAP and not heading:
        return None
    return Violation(words=words, heading=heading)


def repair_request(violation: Violation) -> str:
    """The re-ask that names what the last answer did wrong."""
    return (
        f"Your last answer was {violation.describe()}. The brief was at most {WORD_CAP} "
        f"words of plain prose with no section headings. Rewrite the whole answer now in "
        f"under {WORD_CAP} words: answer in the first sentence, no heading, and no preamble "
        "about the rewrite."
    )


def _domain(url: str) -> str:
    """The site a URL belongs to, so one page per source reaches the footer."""
    netloc = urlsplit(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def fetched_urls(findings: Iterable[dict[str, Any]], limit: int = SOURCE_LIMIT) -> list[str]:
    """The URLs the run actually fetched, one per domain, capped."""
    seen: set[str] = set()
    kept: list[str] = []
    for finding in findings:
        candidates = [str(url) for url in finding.get("sources") or []]
        source = str(finding.get("source") or "")
        if source.startswith("http"):
            candidates.append(source)
        candidates.extend(_URL.findall(str(finding.get("source_content") or "")))
        for candidate in candidates:
            url = candidate.rstrip(".,);")
            domain = _domain(url)
            if not domain or domain in seen:
                continue
            seen.add(domain)
            kept.append(url)
            if len(kept) == limit:
                return kept
    return kept
