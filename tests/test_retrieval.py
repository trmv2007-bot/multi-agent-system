"""Retrieval quality: dedupe, relevance, ranking, and the under-source fallback.

Every test runs over canned search and fetch fixtures. Nothing here touches the
network, and nothing here needs a credential.
"""

import logging

import pytest

from src import tools
from src.tools import (
    TIER_AGGREGATOR,
    TIER_GENERAL,
    TIER_PRIMARY,
    TIER_REFERENCE,
    canonical_url,
    dedupe_results,
    distinctive_terms,
    domain_of,
    filter_sources,
    gather_sources,
    is_relevant,
    rank_sources,
    source_rank,
    source_tier,
)

DOCS = "https://docs.example.com/checkpointer/postgres"
GUIDE = "https://example.org/guides/persistence"
FARM = "https://medium.com/@someone/langgraph-for-everyone"
ON_TOPIC = (
    "LangGraph's Postgres checkpointer stores graph state in a Postgres table, so a "
    "checkpointer survives a restart without a second file store."
)
GUIDE_TEXT = "A Postgres checkpointer keeps LangGraph state in one table."
OFF_TOPIC = "The five best running shoes for the autumn season, reviewed and ranked."


def result(url: str, title: str = "") -> dict[str, str]:
    """One search hit, shaped as the search tool returns it."""
    return {"title": title or url.rsplit("/", 1)[-1], "url": url, "snippet": ""}


def page(url: str, content: str) -> dict[str, object]:
    """One fetched page, shaped as the fetch tool returns it."""
    return {"url": url, "content": content, "status_code": 200}


class TestDedupe:
    """Several results from one site collapse before anything is fetched."""

    def test_same_domain_duplicates_collapse(self):
        results = [
            result("https://docs.example.com/checkpointer/postgres"),
            result("https://docs.example.com/checkpointer/sqlite"),
            result("https://docs.example.com/tutorials/first-graph"),
            result("https://other.org/graph-state"),
        ]

        kept, dropped = dedupe_results(results)

        assert [item["url"] for item in kept] == [
            "https://docs.example.com/checkpointer/postgres",
            "https://other.org/graph-state",
        ]
        assert len(dropped) == 2
        assert all("docs.example.com" in item["reason"] for item in dropped)

    def test_the_first_result_from_a_site_wins(self):
        kept, _ = dedupe_results(
            [result("https://docs.example.com/first"), result("https://docs.example.com/second")]
        )

        assert kept == [result("https://docs.example.com/first")]

    def test_near_duplicate_urls_collapse(self):
        results = [
            result("https://example.com/page"),
            result("http://www.example.com/page/"),
            result("https://example.com/page?utm_source=newsletter&utm_medium=email"),
            result("https://example.com/page/amp"),
        ]

        kept, dropped = dedupe_results(results)

        assert len(kept) == 1
        assert all(item["reason"] == "duplicate page" for item in dropped)

    def test_one_page_through_four_shapes_is_one_url(self):
        shapes = [
            "https://example.com/post",
            "https://www.example.com/post/",
            "http://example.com/post/index.html",
            "https://example.com/post?ref=twitter",
        ]

        assert len({canonical_url(url) for url in shapes}) == 1
        assert canonical_url(shapes[0]) == "example.com/post"

    def test_a_real_query_string_is_kept(self):
        assert canonical_url("https://example.com/search?page=2") == "example.com/search?page=2"

    def test_domain_drops_www_and_port(self):
        assert domain_of("https://www.example.com:8443/a") == "example.com"


class TestRelevance:
    """A fetched page that does not mention the query is dropped."""

    def test_off_topic_page_is_dropped(self):
        selection = filter_sources(
            "langgraph postgres checkpointer",
            [page(DOCS, ON_TOPIC), page(GUIDE, GUIDE_TEXT), page(FARM, OFF_TOPIC)],
        )

        assert [item["url"] for item in selection["sources"]] == [DOCS, GUIDE]
        assert selection["applied"] is True
        assert selection["dropped"] == [{"url": FARM, "reason": "mentions 0 of 3 query terms"}]

    def test_stopwords_do_not_count_as_terms(self):
        assert distinctive_terms("what is the best way to use the checkpointer") == [
            "way",
            "use",
            "checkpointer",
        ]

    def test_single_characters_and_short_filler_are_ignored(self):
        assert distinctive_terms("a b c x asyncio") == ["asyncio"]

    def test_a_version_number_survives(self):
        assert distinctive_terms("python 3.13 asyncio") == ["python", "3.13", "asyncio"]

    def test_two_word_query_needs_both_terms(self):
        terms = distinctive_terms("borrow checker")

        assert is_relevant(terms, "The Rust borrow checker enforces lifetimes.")
        assert not is_relevant(terms, "Rust is a systems language.")

    def test_three_word_query_needs_two_terms(self):
        terms = distinctive_terms("langgraph postgres checkpointer")

        assert is_relevant(terms, "A checkpointer for Postgres.")
        assert not is_relevant(terms, "Checkpointing in general.")

    def test_punctuated_names_still_match(self):
        terms = distinctive_terms("asyncio gather")

        assert is_relevant(terms, "Call asyncio.gather(*tasks) to run them together.")

    def test_a_query_without_distinctive_terms_keeps_everything(self):
        sources = [page(DOCS, ON_TOPIC), page(FARM, OFF_TOPIC)]

        selection = filter_sources("what is the best", sources)

        assert selection["sources"] == sources
        assert selection["dropped"] == []
        assert selection["applied"] is False
        assert selection["note"] == ""


class TestMinimumSources:
    """Filtering never leaves the researcher with too little evidence."""

    def test_query_with_no_match_keeps_the_original_set(self):
        sources = [
            page(FARM, OFF_TOPIC),
            page("https://example.org/shoes", "Trail shoes compared."),
            page("https://example.net/kit", "Our kit list."),
        ]

        selection = filter_sources("xyzzy plugh frotz", sources)

        assert selection["sources"] == sources
        assert selection["applied"] is False
        assert len(selection["dropped"]) == 3
        assert "below the minimum of 2" in selection["note"]

    def test_too_few_relevant_pages_falls_back_to_everything(self):
        sources = [
            page(DOCS, ON_TOPIC),
            page(FARM, OFF_TOPIC),
            page("https://example.org/shoes", "Trail shoes compared."),
        ]

        selection = filter_sources("langgraph postgres checkpointer", sources)

        assert selection["sources"] == sources
        assert selection["applied"] is False
        assert "only 1 of 3 pages were on topic" in selection["note"]

    def test_enough_relevant_pages_keep_the_filter(self):
        sources = [
            page(DOCS, ON_TOPIC),
            page(GUIDE, "A Postgres checkpointer keeps LangGraph state in one table."),
            page(FARM, OFF_TOPIC),
        ]

        selection = filter_sources("langgraph postgres checkpointer", sources)

        assert selection["sources"] == sources[:2]
        assert selection["applied"] is True
        assert selection["note"] == ""

    def test_the_minimum_is_the_callers_to_set(self):
        sources = [page(DOCS, ON_TOPIC), page(FARM, OFF_TOPIC)]

        selection = filter_sources("langgraph postgres checkpointer", sources, minimum=1)

        assert selection["sources"] == [sources[0]]
        assert selection["applied"] is True


class TestRanking:
    """Documentation and long-form pages outrank content farms."""

    def test_documentation_hosts_are_primary(self):
        assert source_tier("https://docs.python.org/3/library/asyncio.html") == TIER_PRIMARY
        assert source_tier("https://developer.mozilla.org/en-US/docs/Web/API") == TIER_PRIMARY
        assert source_tier("https://www.nist.gov/publications/report") == TIER_PRIMARY

    def test_source_links_on_github_are_primary(self):
        assert source_tier("https://github.com/owner/repo/blob/main/src/x.py") == TIER_PRIMARY

    def test_reference_works_sit_between(self):
        assert source_tier("https://en.wikipedia.org/wiki/PostgreSQL") == TIER_REFERENCE

    def test_anyone_can_publish_hosts_are_aggregators(self):
        assert source_tier(FARM) == TIER_AGGREGATOR
        assert source_tier("https://someone.substack.com/p/post") == TIER_AGGREGATOR
        assert source_tier("https://www.quora.com/What-is-a-checkpointer") == TIER_AGGREGATOR

    def test_an_unknown_site_is_general(self):
        assert source_tier("https://example.org/some-post") == TIER_GENERAL

    def test_documentation_outranks_a_longer_content_farm(self):
        ranked = rank_sources(
            [
                page(FARM, ON_TOPIC * 200),
                page("https://example.org/some-post", ON_TOPIC),
                page(DOCS, ON_TOPIC),
            ]
        )

        assert [item["url"] for item in ranked] == [
            DOCS,
            "https://example.org/some-post",
            FARM,
        ]

    def test_the_longer_page_wins_within_one_tier(self):
        short = page("https://example.org/short", ON_TOPIC)
        long = page("https://example.org/long", ON_TOPIC * 20)

        assert source_rank(long) > source_rank(short)
        assert rank_sources([short, long]) == [long, short]


class TestGatherSources:
    """The whole path over fixtures: dedupe, fetch, filter, rank, report."""

    @pytest.mark.asyncio
    async def test_on_topic_pages_survive_and_the_rest_are_reported(self, monkeypatch):
        pages = {DOCS: ON_TOPIC, GUIDE: GUIDE_TEXT, FARM: OFF_TOPIC}

        async def fake_search(query: str, max_results: int = 5) -> dict[str, object]:
            return {"query": query, "engine": "fixture", "results": [result(url) for url in pages]}

        async def fake_fetch(url: str, max_chars: int = 5000) -> dict[str, object]:
            return page(url, pages[url])

        monkeypatch.setattr(tools, "web_search", fake_search)
        monkeypatch.setattr(tools, "fetch_url", fake_fetch)

        gathered = await gather_sources("langgraph postgres checkpointer")

        assert [item["url"] for item in gathered["sources"]] == [DOCS, GUIDE]
        assert gathered["dropped"] == [{"url": FARM, "reason": "mentions 0 of 3 query terms"}]
        assert gathered["note"] == ""

    @pytest.mark.asyncio
    async def test_the_drop_and_the_reason_are_logged_once(self, monkeypatch, caplog):
        pages = {DOCS: ON_TOPIC, GUIDE: GUIDE_TEXT, FARM: OFF_TOPIC}

        async def fake_search(query: str, max_results: int = 5) -> dict[str, object]:
            return {"query": query, "engine": "fixture", "results": [result(url) for url in pages]}

        async def fake_fetch(url: str, max_chars: int = 5000) -> dict[str, object]:
            return page(url, pages[url])

        monkeypatch.setattr(tools, "web_search", fake_search)
        monkeypatch.setattr(tools, "fetch_url", fake_fetch)

        with caplog.at_level(logging.INFO, logger="src.tools"):
            gathered = await gather_sources("langgraph postgres checkpointer")

        assert [item["url"] for item in gathered["sources"]] == [DOCS, GUIDE]
        assert gathered["note"] == ""
        lines = [record.getMessage() for record in caplog.records]
        assert any("opened 3, kept 2, dropped 1" in line and FARM in line for line in lines)
        assert any("mentions 0 of 3 query terms" in line for line in lines)

    @pytest.mark.asyncio
    async def test_the_fallback_is_reported_to_the_researcher(self, monkeypatch, caplog):
        pages = {DOCS: ON_TOPIC, FARM: OFF_TOPIC}

        async def fake_search(query: str, max_results: int = 5) -> dict[str, object]:
            return {"query": query, "engine": "fixture", "results": [result(url) for url in pages]}

        async def fake_fetch(url: str, max_chars: int = 5000) -> dict[str, object]:
            return page(url, pages[url])

        monkeypatch.setattr(tools, "web_search", fake_search)
        monkeypatch.setattr(tools, "fetch_url", fake_fetch)

        with caplog.at_level(logging.INFO, logger="src.tools"):
            gathered = await gather_sources("langgraph postgres checkpointer")

        assert [item["url"] for item in gathered["sources"]] == [DOCS, FARM]
        assert "below the minimum of 2" in gathered["note"]
        assert any("below the minimum of 2" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_pages_that_will_not_open_are_not_sources(self, monkeypatch):
        async def fake_search(query: str, max_results: int = 5) -> dict[str, object]:
            return {"query": query, "engine": "fixture", "results": [result(DOCS)]}

        async def fake_fetch(url: str, max_chars: int = 5000) -> dict[str, object]:
            return {"url": url, "error": "connection refused"}

        monkeypatch.setattr(tools, "web_search", fake_search)
        monkeypatch.setattr(tools, "fetch_url", fake_fetch)

        gathered = await gather_sources("langgraph postgres checkpointer")

        assert gathered["sources"] == []
        assert "nothing to read" in gathered["note"]
