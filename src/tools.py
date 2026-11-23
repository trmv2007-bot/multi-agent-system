"""Tool implementations shared by the MCP server and the in-process agents."""

import asyncio
import html
import logging
import math
import random
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from langchain_core.tools import BaseTool, StructuredTool

from src.config import settings

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) multi-agent-system/0.1"
DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE = "https://lite.duckduckgo.com/lite/"
SEARCH_ENDPOINTS = (DUCKDUCKGO_HTML, DUCKDUCKGO_LITE)
SEARCH_ATTEMPTS = 3
GITHUB_SEARCH = "https://api.github.com/search"
HTTP_FORBIDDEN = 403
_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_RE = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _strip_tags(fragment: str) -> str:
    """Turn a HTML fragment into plain text."""
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo wraps some links in a redirect; return the target URL."""
    if "duckduckgo.com/l/" not in href:
        return href
    target = parse_qs(urlparse(href).query).get("uddg")
    return target[0] if target else href


def parse_duckduckgo(page: str, limit: int) -> list[dict[str, str]]:
    """Extract titles, URLs and snippets from a DuckDuckGo HTML results page."""
    links = _RESULT_RE.findall(page)
    snippets = _SNIPPET_RE.findall(page)

    results = []
    for index, (href, title) in enumerate(links[:limit]):
        snippet = snippets[index] if index < len(snippets) else ""
        results.append(
            {
                "title": _strip_tags(title),
                "url": _unwrap_redirect(html.unescape(href)),
                "snippet": _strip_tags(snippet),
            }
        )
    return results


MIN_SOURCES = 2
"""How many on-topic pages one question needs before the relevance filter is trusted.

Below this, filtering would hand the researcher a thinner evidence base than the
search actually gave it, so the unfiltered set is kept instead and the run says so.
"""

GATHER_RESULTS = 4
"""How many results one ``gather_sources`` call returns."""

EXCERPT_CHARS = 2500
"""How much of one page is handed to the researcher."""

REQUIRED_TERM_SHARE = 0.6
"""Share of a query's distinctive terms a page must mention to count as on topic.

Rounded up, so a two-word query needs both words, three needs two, five needs three.
"""

TIER_AGGREGATOR = 0
TIER_GENERAL = 1
TIER_REFERENCE = 2
TIER_PRIMARY = 3

AGGREGATOR_HOSTS = frozenset(
    {
        "answers.com",
        "blogspot.com",
        "dev.to",
        "ehow.com",
        "facebook.com",
        "hashnode.dev",
        "linkedin.com",
        "medium.com",
        "pinterest.com",
        "quora.com",
        "reddit.com",
        "scribd.com",
        "slideshare.net",
        "substack.com",
        "tumblr.com",
        "twitter.com",
        "wikihow.com",
        "wordpress.com",
        "x.com",
        "youtube.com",
    }
)
"""Sites where anyone can publish: demoted below everything else."""

REFERENCE_HOSTS = frozenset(
    {"arxiv.org", "britannica.com", "nature.com", "sciencedirect.com", "wikipedia.org"}
)
"""Compiled reference works: ranked above general pages, below official ones."""

PRIMARY_HOST_PREFIXES = ("api.", "developer.", "developers.", "docs.", "learn.")
PRIMARY_HOST_SUFFIXES = (".edu", ".gov", ".int", ".mil")
PRIMARY_PATH_MARKERS = (
    "/api/",
    "/doc/",
    "/docs/",
    "/documentation/",
    "/guide/",
    "/manual/",
    "/reference/",
    "/rfc/",
    "/spec",
)
"""Documentation, standards, and official sites: ranked above everything else."""

TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src", "source"}
)
"""Query parameters that never change which page a URL points at."""

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below best between both but by can could did do does doing down during each few
    for from further good had has have having he her here hers him his how i if in into is
    it its itself just latest me more most my no nor not now of off on once only or other
    ours out over own same she should so some such than that the their them then there
    these they this those through to too top under until up us very was we were what when
    where which while who whom why will with would you your versus vs
    """.split()
)
"""Words that say nothing about what a query is about."""

MIN_TERM_CHARS = 2
"""Shorter tokens (``a``, ``x``, a stray digit) carry no topic on their own."""

QUERY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#_-]*")


def domain_of(url: str) -> str:
    """The site a URL belongs to, without ``www.`` or a port."""
    host = (urlparse(url).netloc or "").lower().rsplit("@", 1)[-1].split(":")[0]
    return host.removeprefix("www.")


def canonical_url(url: str) -> str:
    """The URL with everything that never changes which page it is removed.

    Scheme, ``www.``, a trailing slash, ``index.html``, an AMP suffix, and
    tracking parameters all go, so two links to one page compare equal.
    """
    parsed = urlparse(url)
    path = (parsed.path or "/").removesuffix("/index.html").removesuffix("/index.htm")
    path = path.removesuffix("/amp").rstrip("/") or "/"
    params = sorted(
        (name, value)
        for name, values in parse_qs(parsed.query).items()
        if not name.startswith("utm_") and name not in TRACKING_PARAMS
        for value in values
    )
    query = "&".join(f"{name}={value}" for name, value in params)
    return f"{domain_of(url)}{path}" + (f"?{query}" if query else "")


def dedupe_results(
    results: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Collapse results to one page per site before anything is fetched.

    Two links to the same page, and a second result from a site already in the
    list, are dropped with the reason recorded. The first occurrence wins: the
    engine ordered the results by relevance already, so the earliest one is the
    result it thought best.
    """
    kept: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    seen_pages: set[str] = set()
    seen_sites: set[str] = set()

    for result in results:
        url = str(result.get("url") or "")
        page, site = canonical_url(url), domain_of(url)
        if page in seen_pages:
            dropped.append({"url": url, "reason": "duplicate page"})
        elif site in seen_sites:
            dropped.append({"url": url, "reason": f"second result from {site}"})
        else:
            seen_pages.add(page)
            seen_sites.add(site)
            kept.append(result)
    return kept, dropped


def distinctive_terms(query: str) -> list[str]:
    """The words in a query that actually identify it, in the order they appear.

    Stopwords and one-character tokens say nothing about the topic, so a page
    could match them and still be about something else. Everything else is
    kept, version numbers included.
    """
    terms: list[str] = []
    for token in QUERY_TOKEN_RE.findall(query.lower()):
        word = token.strip("._-")
        if len(word) < MIN_TERM_CHARS or word in STOPWORDS:
            continue
        if word not in terms:
            terms.append(word)
    return terms


def matched_terms(terms: list[str], text: str) -> int:
    """How many of the query's terms this page mentions."""
    haystack = text.lower()
    return sum(1 for term in terms if term in haystack)


def is_relevant(terms: list[str], text: str) -> bool:
    """Whether a page mentions enough of the query's terms to be on topic.

    A query with no distinctive terms (``what is the best``) matches everything:
    there is nothing to judge it against, so nothing is dropped.
    """
    if not terms:
        return True
    return matched_terms(terms, text) >= math.ceil(len(terms) * REQUIRED_TERM_SHARE)


def source_tier(url: str) -> int:
    """How primary a page looks from its URL alone; higher is better.

    The rule is deliberately blunt and readable: known places where anyone can
    publish are demoted, documentation hosts and paths, official domains, and
    source-code links are promoted, compiled references sit between, and the rest
    keeps the middle.
    """
    host = domain_of(url)
    path = (urlparse(url).path or "/").lower()
    on_host = host if host in AGGREGATOR_HOSTS else ""

    if on_host or any(host.endswith(f".{farm}") for farm in AGGREGATOR_HOSTS):
        return TIER_AGGREGATOR
    if host.endswith(PRIMARY_HOST_SUFFIXES) or host.startswith(PRIMARY_HOST_PREFIXES):
        return TIER_PRIMARY
    if host == "github.com" and ("/blob/" in path or "/tree/" in path):
        return TIER_PRIMARY
    if any(marker in path for marker in PRIMARY_PATH_MARKERS):
        return TIER_PRIMARY
    if host in REFERENCE_HOSTS or any(host.endswith(f".{ref}") for ref in REFERENCE_HOSTS):
        return TIER_REFERENCE
    return TIER_GENERAL


def source_rank(source: dict[str, Any]) -> tuple[int, int]:
    """Sort key for one fetched page: primary first, then long-form."""
    return source_tier(str(source.get("url") or "")), min(
        len(str(source.get("content") or "")), 40_000
    )


def rank_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order pages so documentation and long-form come before farms, longest first."""
    return sorted(sources, key=source_rank, reverse=True)


def filter_sources(
    query: str, sources: list[dict[str, Any]], minimum: int = MIN_SOURCES
) -> dict[str, Any]:
    """Keep the fetched pages that are about the query, or keep them all and say why.

    Returns ``sources`` (what the caller should use), ``dropped`` (one reason line
    per rejected page), ``applied``, and a ``note`` when the filter was overruled.
    Fewer than ``minimum`` surviving pages means the unfiltered set comes back
    instead: an under-sourced answer is worse than a noisy one.
    """
    terms = distinctive_terms(query)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []

    for source in sources:
        text = str(source.get("content") or "")
        if not text or is_relevant(terms, text):
            kept.append(source)
        else:
            dropped.append(
                {
                    "url": str(source.get("url") or ""),
                    "reason": f"mentions {matched_terms(terms, text)} of {len(terms)} query terms",
                }
            )

    if len(kept) >= minimum or not dropped:
        return {"sources": kept, "dropped": dropped, "applied": bool(dropped), "note": ""}

    note = (
        f"only {len(kept)} of {len(sources)} pages were on topic, below the minimum of "
        f"{minimum}, so every fetched page was kept"
    )
    return {"sources": list(sources), "dropped": dropped, "applied": False, "note": note}


async def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search the web, tolerating several agents searching at the same time.

    Parallel workers hit the same endpoint at once, and DuckDuckGo throttles
    bursts. Each attempt jitters, then moves to the lighter endpoint, because a
    search failing silently costs a whole section of a page.
    """
    last = ""
    for attempt in range(SEARCH_ATTEMPTS):
        endpoint = SEARCH_ENDPOINTS[attempt % len(SEARCH_ENDPOINTS)]
        if attempt:
            await asyncio.sleep(0.6 * attempt + random.uniform(0, 0.8))

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.post(
                    endpoint,
                    data={"q": query},
                    headers={"User-Agent": USER_AGENT},
                )
            if response.status_code in (429, 503):
                last = f"throttled ({response.status_code})"
                continue
            response.raise_for_status()

            results = parse_duckduckgo(response.text, max_results * 2)
            results, duplicates = dedupe_results(results)
            results = results[:max_results]
            if duplicates:
                logger.info(
                    "search %r: kept %d results, dropped %d (%s)",
                    query,
                    len(results),
                    len(duplicates),
                    "; ".join(f"{item['url']} {item['reason']}" for item in duplicates),
                )
            if results:
                primitive = "duckduckgo"
                return {
                    "query": query,
                    "engine": primitive,
                    "results": results,
                    "dropped": len(duplicates),
                }
            last = "no parseable results"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"

    raise RuntimeError(f"search failed for {query!r} after {SEARCH_ATTEMPTS} attempts ({last})")


async def fetch_url(url: str, max_chars: int = 5000) -> dict[str, Any]:
    """Fetch a URL and return its text content, truncated to max_chars."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            return {"url": url, "error": str(e)}

    content = response.text
    if len(content) > max_chars:
        content = content[:max_chars] + "... [truncated]"

    return {
        "url": url,
        "status_code": response.status_code,
        "content": content,
        "content_type": response.headers.get("content-type", "unknown"),
    }


async def gather_sources(query: str, max_sources: int = GATHER_RESULTS) -> dict[str, Any]:
    """Search, drop duplicate sites, open the results, keep the pages that are on topic.

    The one call a researcher needs for evidence: it searches, collapses several
    results from one site into one, opens the survivors, drops the pages that are
    not about the query, ranks the rest documentation-first, and reports in one
    line what it dropped and why. When filtering would leave fewer than
    ``MIN_SOURCES`` pages it returns every fetched page and says so in ``note``.
    """
    search = await web_search(query, max_results=max_sources * 2)
    candidates = search["results"][: max_sources + 2]
    opened = await asyncio.gather(
        *(fetch_url(str(result["url"]), max_chars=EXCERPT_CHARS) for result in candidates)
    )

    titles = {str(result["url"]): str(result.get("title") or "") for result in candidates}
    fetched = [
        {**page, "title": titles.get(str(page.get("url")), "")}
        for page in opened
        if str(page.get("content") or "").strip()
    ]
    if not fetched:
        note = "no search result could be opened, so there is nothing to read"
        logger.info("sources %r: %s", query, note)
        return {"query": query, "sources": [], "dropped": [], "note": note}

    selection = filter_sources(query, fetched)
    sources = rank_sources(selection["sources"])[:max_sources]
    dropped = "; ".join(f"{item['url']} {item['reason']}" for item in selection["dropped"][:3])
    logger.info(
        "sources %r: opened %d, kept %d, dropped %d%s",
        query,
        len(fetched),
        len(sources),
        len(selection["dropped"]),
        f" ({dropped})" if dropped else "",
    )
    if selection["note"]:
        logger.info("sources %r: %s", query, selection["note"])

    return {
        "query": query,
        "sources": sources,
        "dropped": selection["dropped"],
        "note": selection["note"],
    }


def _github_item(item: dict[str, Any], search_type: str) -> dict[str, Any]:
    """Normalize one GitHub search hit."""
    if search_type == "repositories":
        return {
            "name": item.get("full_name"),
            "url": item.get("html_url"),
            "description": item.get("description"),
            "stars": item.get("stargazers_count"),
            "language": item.get("language"),
            "updated_at": item.get("updated_at"),
        }
    if search_type == "issues":
        return {
            "title": item.get("title"),
            "url": item.get("html_url"),
            "state": item.get("state"),
            "comments": item.get("comments"),
        }
    return {
        "path": item.get("path"),
        "repository": (item.get("repository") or {}).get("full_name"),
        "url": item.get("html_url"),
    }


async def search_github(query: str, search_type: str = "repositories") -> dict[str, Any]:
    """Search GitHub through its public search API."""
    if search_type == "code" and not settings.github_token:
        raise RuntimeError("GitHub code search requires GITHUB_TOKEN to be set")

    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{GITHUB_SEARCH}/{search_type}",
            params={"q": query, "per_page": 5},
            headers=headers,
        )

    if response.status_code == HTTP_FORBIDDEN:
        raise RuntimeError("GitHub search rate limit reached; set GITHUB_TOKEN to raise the limit")
    response.raise_for_status()
    payload = response.json()

    return {
        "query": query,
        "type": search_type,
        "total_count": payload.get("total_count", 0),
        "results": [_github_item(item, search_type) for item in payload.get("items", [])],
    }


async def current_time() -> dict[str, Any]:
    """Get the current UTC date and time."""
    now = datetime.now(UTC)
    return {"utc": now.isoformat(), "timestamp": now.timestamp(), "timezone": "UTC"}


async def build_page(html: str, title: str = "") -> str:
    """Hand a finished web page to the build box.

    Agents use this instead of pasting markup into a reply: the page lands in the
    preview, the conversation keeps a one-line note, and the page is saved with
    the thread.
    """
    from src.activity import activity

    document = str(html or "").strip()
    if not document:
        return "No HTML was given, so nothing was shown."
    if "<html" not in document.lower():
        document = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            "<title>" + (title or "Page") + "</title></head><body>" + document + "</body></html>"
        )

    activity.page(document, title)
    return (
        f"The page is now in the build box ({len(document)} characters). "
        "Do not paste the HTML in your reply; just say in one line what you built."
    )


async def set_theme(  # noqa: PLR0913, PLR0917 - one call, explicit fields
    feeling: str,
    bg: str,
    accent: str,
    text: str,
    surface: str = "",
    muted: str = "",
    radius: str = "6px",
    display_size: str = "clamp(2rem, 5vw, 3.4rem)",
    background_css: str = "",
    extra_css: str = "",
    motion_css: str = "",
    extras_js: str = "",
) -> str:
    """Set this page's visual system: colours, shape, background, motion."""
    from src.design import capture_theme

    capture_theme(
        feeling=feeling,
        bg=bg,
        surface=surface or bg,
        accent=accent,
        text=text,
        muted=muted or text,
        radius=radius,
        display_size=display_size,
        background_css=background_css,
        extra_css=extra_css,
        motion_css=motion_css,
        extras_js=extras_js,
    )
    return "Theme set. The page will be built in these colours."


def as_langchain_tools(include_computer: bool | None = None) -> list[BaseTool]:
    """Expose the tools to in-process agents as LangChain tools."""
    tools = _web_tools()
    tools.append(
        StructuredTool.from_function(
            coroutine=set_theme,
            name="set_theme",
            description=(
                "Set the page's visual system before it is written: hex colours for bg, "
                "surface, accent, text and muted, plus radius, display font size in CSS, "
                "a background_css declaration, extra_css, motion_css and optional "
                "extras_js. Call this once, first, then describe the design in a sentence."
            ),
        )
    )
    tools.append(
        StructuredTool.from_function(
            coroutine=build_page,
            name="build_page",
            description=(
                "Hand a finished web page to the build box. Pass the complete HTML as "
                "'html'. Use this instead of writing markup in your reply."
            ),
        )
    )
    wants_computer = settings.computer_use if include_computer is None else include_computer
    if wants_computer:
        from src.computer import computer_tools

        tools.extend(computer_tools())
    return tools


def _web_tools() -> list[BaseTool]:
    """The web and time tools, always available."""
    return [
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description=(
                "Search the web for current information. Returns titles, URLs and snippets, "
                "one result per site."
            ),
        ),
        StructuredTool.from_function(
            coroutine=gather_sources,
            name="gather_sources",
            description=(
                "Search, open the top results, and return the pages that are actually about "
                "the question, one per site, documentation first. Prefer this over "
                "web_search + fetch_url when you need evidence for a question."
            ),
        ),
        StructuredTool.from_function(
            coroutine=fetch_url,
            name="fetch_url",
            description="Fetch a URL and return its text content.",
        ),
        StructuredTool.from_function(
            coroutine=search_github,
            name="search_github",
            description="Search GitHub repositories or issues.",
        ),
        StructuredTool.from_function(
            coroutine=current_time,
            name="current_time",
            description="Get the current UTC date and time.",
        ),
    ]
