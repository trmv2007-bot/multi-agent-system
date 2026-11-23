"""Build mode agents, second pass: research the material, refine, hand over.

The flow is: outline the page, research what the sections will actually need,
revise the outline against that research, write the sections in parallel, review
the assembled page for errors, fix only what the review flagged, then summarise
what was built.
"""

import json
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.activity import activity
from src.agents.base import AgentConfig, BaseAgent
from src.artifacts import SECTION_GUIDE
from src.design import (
    CAPTURED,
    DESIGN_SKILL,
    THEME_PROMPT,
    capture_reset,
    fallback_theme,
    is_default,
    parse_theme,
    subject_theme,
)
from src.graph.state import AgentRole, AgentState

logger = logging.getLogger(__name__)

DEFAULT_SECTIONS = 4
MAX_SECTIONS = 16
MAX_FIXES = 6

PLANNER_PROMPT = """You plan one web page. Break it into the sections it needs, in order.

Reply with JSON only, an array of objects:
[{"section": "hero", "brief": "what this section must say and why"}]

Rules:
- Between 2 and __MAX_SECTIONS__ sections, in the order they appear on the page.
- The first is the hero: headline, one-sentence subhead, primary call to action.
- One job per section, no overlap.
- No prose before or after the JSON."""

RESEARCH_PROMPT = """You research what a web page needs before it is written.

Rules:
- Use your tools to check real facts, real names, real numbers and current
  conventions. Never invent a statistic, price or claim.
- Keep what you find as short, quotable notes a writer can drop into copy.
- Note the sources you used, one per line at the end after "SOURCES:".
- At most 60 words per note. No essays."""

CRITIQUE_PROMPT = """You review an assembled web page and list what is wrong.

Reply with JSON only, an array of objects:
[{"section": "which section it is in", "issue": "what is wrong and how to fix it"}]

Look for: broken or duplicated copy, contradictions, empty or placeholder text,
markup that leaked into the text, missing sections from the outline, and anything
that reads like machine noise rather than a finished page.

Rules:
- At most __MAX_FIXES__ items, worst first.
- Only real defects; no stylistic preferences.
- If the page is sound, reply with []"""

ANALYSIS_PROMPT = """You report on a page the team just built.

Say, in at most 120 words and without headings: what the page contains, which
sections were rewritten during review and why, and anything you would still fix.
Plain sentences, no praise, no filler."""

DESIGNER_PROMPT = DESIGN_SKILL + chr(10) + chr(10) + THEME_PROMPT

FIX_PROMPT = """One section of a page needs a specific fix.

Return the full replacement HTML for that section, using only the existing classes
(container, hero, sub, actions, btn, eyebrow, features, card, quotes, blockquote,
stat, cta). Fix exactly what the review asked for and leave the rest as it was.
Return HTML only: no fences, no explanation."""


def parse_sections(text: str, fallback: str, limit: int) -> list[dict[str, Any]]:
    """Read a JSON list of sections out of a model reply."""
    match = re.search(r"\[.*\]", text, re.S)
    items: list[Any] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                items = parsed
        except json.JSONDecodeError:
            items = []

    sections: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            name, brief = item, item
        elif isinstance(item, dict):
            name = str(item.get("section") or item.get("name") or "section").strip()
            brief = str(item.get("brief") or item.get("task") or name).strip()
        else:
            continue
        if brief:
            sections.append({"id": len(sections), "section": name, "brief": brief})
        if len(sections) >= limit:
            break

    if sections:
        return sections
    return [{"id": 0, "section": "hero", "brief": fallback}]


def parse_fixes(text: str, limit: int) -> list[dict[str, str]]:
    """Read the review's JSON list of defects."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []

    fixes: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "").strip()
        issue = str(item.get("issue") or item.get("problem") or "").strip()
        if issue:
            fixes.append({"section": section, "issue": issue})
        if len(fixes) >= limit:
            break
    return fixes


def sources_from(text: str) -> list[str]:
    """URLs mentioned in a research note."""
    found = re.findall(r"https?://[^\s\"'<>\])}]+", text)
    seen: set[str] = set()
    unique = []
    for url in found:
        clean = url.rstrip(".,)")
        if clean not in seen:
            seen.add(clean)
            unique.append(clean)
    return unique[:8]


class BuildPlannerAgent(BaseAgent):
    """Decides which sections the page needs."""

    def __init__(
        self, llm: BaseChatModel | None = None, max_sections: int = DEFAULT_SECTIONS
    ) -> None:
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Page planner",
            description="Splits a page into sections",
            system_prompt=PLANNER_PROMPT.replace("__MAX_SECTIONS__", str(max_sections)),
        )
        super().__init__(config, llm)
        self.max_sections = max_sections

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": f"Page to plan: {state.current_task}"},
            ]
        )
        sections = parse_sections(str(response.content), state.current_task, self.max_sections)

        return {
            "plan": sections,
            "agent_timings": {"planner": time.time() - start},
            "messages": [self._message(f"Planned {len(sections)} sections")],
        }


class PageResearchAgent(BaseAgent):
    """Finds the facts and conventions the sections will need."""

    def __init__(self, llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None):
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            name="Page researcher",
            description="Researches the material the page needs",
            system_prompt=RESEARCH_PROMPT,
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        outline = "\n".join(f"- {item.get('section')}: {item.get('brief')}" for item in state.plan)
        notes = ""
        runs: list[dict[str, Any]] = []
        try:
            notes, runs = await self._with_tools_traced(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Page: {state.current_task}\n\nSections planned:\n{outline}\n\n"
                            "What does this page need to be accurate and specific?"
                        ),
                    },
                ]
            )
        except Exception:
            notes = ""

        return {
            "research_findings": [
                {
                    "source": sources_from(notes)[0] if sources_from(notes) else "llm_synthesis",
                    "content": notes.strip(),
                    "confidence": 0.8,
                    "relevance": 0.9,
                    "turn": state.turn,
                    "question": "page research",
                    "sources": sources_from(notes),
                }
            ]
            if notes.strip()
            else [],
            "agent_timings": {"page_researcher": time.time() - start},
            "messages": [self._message(f"Researched the page with {len(runs)} tool calls")],
        }


class ReplanAgent(BaseAgent):
    """Revises the outline so the sections use what the research found."""

    def __init__(self, llm: BaseChatModel | None = None, max_sections: int = DEFAULT_SECTIONS):
        config = AgentConfig(
            role=AgentRole.ANALYST,
            name="Outline reviser",
            description="Improves the outline with the research",
            system_prompt=PLANNER_PROMPT.replace("__MAX_SECTIONS__", str(max_sections)),
        )
        super().__init__(config, llm)
        self.max_sections = max_sections

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        notes = "\n".join(
            str(item.get("content", ""))
            for item in state.research_findings
            if item.get("turn") == state.turn
        )
        outline = json.dumps(
            [{"section": i.get("section"), "brief": i.get("brief")} for i in state.plan]
        )
        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Page: {state.current_task}\n\nCurrent outline: {outline}\n\n"
                        f"Research notes:\n{notes[:4000]}\n\n"
                        "Return the improved outline as JSON."
                    ),
                },
            ]
        )
        revised = parse_sections(str(response.content), state.current_task, self.max_sections)

        return {
            "plan": revised,
            "agent_timings": {"replanner": time.time() - start},
            "messages": [self._message(f"Revised the outline to {len(revised)} sections")],
        }


class DesignerAgent(BaseAgent):
    """Sets the visual direction the sections are written against."""

    def __init__(self, llm: BaseChatModel | None = None, tools: list[BaseTool] | None = None):
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Designer",
            description="Decides how the page should look",
            system_prompt=DESIGNER_PROMPT,
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        outline = "\n".join(f"- {item.get('section')}: {item.get('brief')}" for item in state.plan)
        notes = "\n".join(
            str(item.get("content", ""))
            for item in state.research_findings
            if item.get("turn") == state.turn
        )
        brief = ""
        capture_reset()
        try:
            answer, _runs = await self._with_tools_traced(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Page: {state.current_task}\n\nSections:\n{outline}\n\n"
                            f"What we found:\n{notes[:2000] or '(no research)'}"
                        ),
                    },
                ]
            )
            brief = text_only(answer).strip()
        except Exception:
            brief = ""

        # The tool call is the reliable path; the text is the fallback.
        theme = dict(CAPTURED) if CAPTURED else parse_theme(brief)
        if is_default(theme):
            # The designer answered in prose. Ask once more, bluntly.
            try:
                stricter = await self._ainvoke(
                    [
                        {"role": "system", "content": self.config.system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Page: {state.current_task}. "
                                "Return only the JSON object. No prose, no explanation, "
                                "no code fences."
                            ),
                        },
                    ]
                )
                retried = parse_theme(text_only(stricter))
                if not is_default(retried):
                    theme = retried
            except Exception as e:
                # Keep whatever we have; the subject match below covers it.
                logger.debug("designer retry failed: %s", e)
        if is_default(theme):
            theme = subject_theme(state.current_task) or fallback_theme(state.current_task)

        return {
            "context": {
                "design_brief": str(theme.get("feeling", "")),
                "design_theme": theme,
            },
            "agent_timings": {"designer": time.time() - start},
            "messages": [self._message("Set the visual direction")],
        }


def text_only(value: Any) -> str:
    """Model output as plain text."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", "")) for part in content
        )
    return str(content)


class SectionWriterAgent(BaseAgent):
    """Writes one section of the page as an HTML fragment."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        config = AgentConfig(
            role=AgentRole.WRITER,
            name="Section writer",
            description="Writes one section of the page",
            system_prompt=SECTION_GUIDE,
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        return await self.write_section(
            index=0,
            section="hero",
            brief=state.current_task,
            page_title=state.current_task,
            earlier=[],
            turn=state.turn,
        )

    async def write_section(  # noqa: PLR0913, PLR0917 - one call per planned section
        self,
        index: int,
        section: str,
        brief: str,
        page_title: str,
        earlier: list[str],
        turn: str = "",
        notes: str = "",
        fix: str = "",
        design: str = "",
    ) -> dict[str, Any]:
        """Write (or rewrite) one section and report it as a state update."""
        start = time.time()
        actor = (f"worker.{index}", f"Section {index + 1}")
        self._report("writing", actor)

        system = self.config.system_prompt
        if fix:
            system = FIX_PROMPT

        user = (
            f"Page: {page_title}\n\nYour section: {section}\nBrief: {brief}\n"
            f"Earlier sections: {', '.join(earlier) or '(none)'}\n"
            f"Research notes:\n{notes[:2000] or '(none)'}\n\n"
            f"Design direction:\n{design[:800] or '(none given)'}"
        )
        if fix:
            user += f"\n\nWhat the review said is wrong:\n{fix}"

        error: dict[str, Any] | None = None
        html = ""
        try:
            response = await self._ainvoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            html = str(response.content if hasattr(response, "content") else response)
        except Exception as e:
            error = {"agent": f"worker.{index}", "error": str(e), "details": {"section": section}}
            self._report("failed", actor)

        self._report("done" if error is None else "failed", actor)
        if error is None and html.strip():
            # Publish the fragment so the preview grows section by section
            # instead of jumping at the end of every superstep.
            activity.section(index, section, html)

        update: dict[str, Any] = {
            "subtask_results": [
                {
                    "index": index,
                    "task": f"{section}: {brief}",
                    "content": html,
                    "duration": time.time() - start,
                    "turn": turn,
                }
            ],
            "agent_timings": {f"section.{index}": time.time() - start},
            "messages": [self._message(f"Section {index + 1} ({section}) written")],
        }
        if error is not None:
            update["errors"] = [error]
        return update


class CritiqueAgent(BaseAgent):
    """Reviews the assembled page and lists concrete defects."""

    def __init__(self, llm: BaseChatModel | None = None, max_fixes: int = MAX_FIXES):
        config = AgentConfig(
            role=AgentRole.VALIDATOR,
            name="Reviewer",
            description="Reviews the assembled page",
            system_prompt=CRITIQUE_PROMPT.replace("__MAX_FIXES__", str(max_fixes)),
        )
        super().__init__(config, llm)
        self.max_fixes = max_fixes

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        outline = "\n".join(f"- {item.get('section')}" for item in state.plan)
        page = state.final_report or ""
        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Page: {state.current_task}\n\nOutline:\n{outline}\n\n"
                        f"Assembled HTML:\n{page[:9000]}"
                    ),
                },
            ]
        )
        fixes = parse_fixes(str(response.content), self.max_fixes)
        round_number = int(state.context.get("fix_round", 0)) + 1

        return {
            "context": {"fix_round": round_number},
            "validation_results": [
                {
                    "claim": f"{fix['section']}: {fix['issue']}"[:300],
                    "status": "needs_fix",
                    "severity": "medium",
                    "evidence": [],
                    "confidence": 0.7,
                    "turn": state.turn,
                    "section": fix["section"],
                }
                for fix in fixes
            ],
            "agent_timings": {"reviewer": time.time() - start},
            "messages": [self._message(f"Review found {len(fixes)} things to fix")],
        }


class AnalysisAgent(BaseAgent):
    """Says what was built and what changed on review."""

    def __init__(self, llm: BaseChatModel | None = None):
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Reporter",
            description="Reports what the build produced",
            system_prompt=ANALYSIS_PROMPT,
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        mine = [r for r in state.validation_results if r.get("turn") == state.turn]
        outline = "\n".join(f"- {item.get('section')}" for item in state.plan)
        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Page: {state.current_task}\n\nSections:\n{outline}\n\n"
                        f"Review findings: {len(mine)}\n"
                        + "\n".join(f"- {item.get('claim', '')[:140]}" for item in mine[:6])
                    ),
                },
            ]
        )

        return {
            "context": {"build_summary": str(response.content)},
            "agent_timings": {"reporter": time.time() - start},
            "messages": [self._message("Reported on the build")],
        }
