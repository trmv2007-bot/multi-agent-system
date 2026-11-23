"""Tool implementations shared by the MCP server and the in-process agents."""

import asyncio
import html
import random
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from langchain_core.tools import BaseTool, StructuredTool

from src.config import settings

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

            results = parse_duckduckgo(response.text, max_results)
            if results:
                primitive = "duckduckgo"
                return {"query": query, "engine": primitive, "results": results}
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
            description="Search the web for current information. Returns titles, URLs and snippets.",
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
