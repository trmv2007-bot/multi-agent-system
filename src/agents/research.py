"""Deep research agents: plan questions, fan out finders, chase the gaps.

One planner decides what needs answering, a wave of finders works the questions
at the same time, a gap analyst asks what is still missing, a second wave closes
those gaps, and a report agent writes the answer with its sources.
"""

import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import AgentConfig, BaseAgent
from src.graph.state import AgentRole, AgentState
from src.structured import StructuredOutputError, call_json

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = 4
DEEP_QUESTIONS = 10
MAX_QUESTIONS = 20
MAX_FOLLOW_UPS = 6

URL_RE = re.compile(r"https?://[^\s\"'<>\])}]+")

PLANNER_PROMPT = """You plan research. Break the task into the questions that must be answered.

Reply with JSON only, an array of objects:
[{"question": "a specific, answerable question", "why": "why it matters to the task"}]

Rules:
- Between 3 and __MAX__ questions, ordered by importance.
- Each must be answerable from public sources and stand alone.
- Cover the task from different angles; no near-duplicates.
- No prose outside the JSON."""

FINDER_PROMPT = """You are one researcher on a team. You have one question.

Rules:
- Use your tools. Start with gather_sources, which searches, opens the pages, and
  keeps only those on topic; use web_search plus fetch_url only to follow a
  specific link. Read before answering.
- Answer your question directly in 120-200 words.
- Keep every URL you actually used, one per line at the end after "SOURCES:".
- If the sources disagree, say so in one sentence.
- If you cannot establish something, say that plainly instead of guessing."""

GAP_PROMPT = """You review research coverage for one task.

Given the task and what the team already established, list the questions that are
still unanswered and worth answering.

Reply with JSON only, an array of objects:
[{"question": "the missing question", "why": "what it would add"}]

Rules:
- At most __MAX__ questions. Fewer is fine.
- Only genuine gaps that would change the answer.
- Do not repeat questions already answered.
- If coverage is adequate, reply with []"""

REPORT_PROMPT = """You write the final answer from a team's research.

Rules:
- Answer the task in the first sentence, then structure the rest so it can be
  skimmed: a short lead, then bullets or a numbered list.
- When you compare three or more things on the same attributes, use a small
  markdown table instead of a paragraph.
- One idea per bullet, no bullet longer than two lines.
- Do not paste the research notes together; group what belongs together and cut
  what repeats.
- At most 220 words unless the task asks for a report.
- Cite with markdown links, using only the URLs listed under each finding.
  Never invent a URL and never cite a bare domain.
- One link per claim is enough; do not stack citations.
- Where the evidence was thin, say so rather than padding.
- No headings, no "Executive Summary", no restating the question.
- Never paste HTML or code; this is a spoken-style answer."""


def questions_from(payload: Any, limit: int) -> list[dict[str, str]]:
    """Shape a parsed planner or gap reply into the questions worth asking."""
    items = payload if isinstance(payload, list) else []

    questions: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            question, why = item, ""
        elif isinstance(item, dict):
            question = str(item.get("question") or item.get("task") or "").strip()
            why = str(item.get("why") or "").strip()
        else:
            continue
        if question:
            questions.append({"question": question, "why": why})
        if len(questions) >= limit:
            break
    return questions


def sources_from(runs: list[dict[str, Any]]) -> list[str]:
    """Every URL a finder actually touched, most useful first."""
    found: list[str] = []
    for run in runs:
        url = str(run.get("arguments", {}).get("url", "")).strip()
        if url.startswith("http"):
            found.append(url)
        for match in URL_RE.findall(str(run.get("output", ""))):
            found.append(match)

    seen: set[str] = set()
    unique = []
    for url in found:
        clean = url.rstrip(".,)")
        if clean not in seen:
            seen.add(clean)
            unique.append(clean)
    return unique[:8]


class QuestionPlannerAgent(BaseAgent):
    """Turns a task into the questions worth answering."""

    def __init__(self, llm: BaseChatModel | None = None, max_questions: int = DEFAULT_QUESTIONS):
        config = AgentConfig(
            role=AgentRole.ORCHESTRATOR,
            name="Research planner",
            description="Plans the research questions",
            system_prompt=PLANNER_PROMPT.replace("__MAX__", str(max_questions)),
        )
        super().__init__(config, llm)
        self.max_questions = max_questions

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        questions = [{"question": state.current_task, "why": "the task itself"}]
        try:
            payload = await call_json(
                self._json_sender(self._actor()),
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": f"Task: {state.current_task}"},
                ],
                site="research.planner",
                expect=list,
            )
        except StructuredOutputError as error:
            logger.warning("research planner fell back to the task itself: %s", error)
        else:
            questions = questions_from(payload, self.max_questions) or questions

        plan = [
            {"id": index, "section": item["question"], "brief": item["why"], "wave": 1}
            for index, item in enumerate(questions)
        ]
        span = time.time() - start
        return {
            "plan": plan,
            "agent_timings": {"research_planner": span},
            "messages": [self._message(f"Planned {len(plan)} research questions")],
        }


class FinderAgent(BaseAgent):
    """Answers one question with tools, and keeps its sources."""

    def __init__(self, llm: BaseChatModel | None = None, tools: list[Any] | None = None):
        config = AgentConfig(
            role=AgentRole.RESEARCHER,
            name="Finder",
            description="Answers one research question",
            system_prompt=FINDER_PROMPT,
            tools=tools or [],
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        return await self.find(
            index=0, question=state.current_task, task=state.current_task, turn=state.turn
        )

    async def find(
        self, index: int, question: str, task: str, turn: str = "", wave: int = 1
    ) -> dict[str, Any]:
        """Research one question and report it as a state update."""
        start = time.time()
        actor = (f"worker.{index}", f"Finder {index + 1}")
        self._report("researching", actor)

        error: dict[str, Any] | None = None
        answer = ""
        sources: list[str] = []

        try:
            answer, runs = await self._with_tools_traced(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {
                        "role": "user",
                        "content": f"Overall task: {task}\n\nYour question: {question}",
                    },
                ],
                actor,
            )
            sources = sources_from(runs)
        except Exception as e:
            error = {"agent": f"worker.{index}", "error": str(e), "details": {"question": question}}
            self._report("failed", actor)

        self._report("done" if error is None else "failed", actor)
        answer = answer.strip() or "(no answer produced)"

        update: dict[str, Any] = {
            "research_findings": [
                {
                    "source": sources[0] if sources else "llm_synthesis",
                    "content": answer,
                    "confidence": 0.8 if sources else 0.5,
                    "relevance": 0.9,
                    "question": question,
                    "sources": sources,
                    "wave": wave,
                    "turn": turn,
                }
            ],
            "agent_timings": {f"finder.{index}": time.time() - start},
            "messages": [self._message(f"Researched: {question[:60]}")],
        }
        if error is not None:
            update["errors"] = [error]
        return update


class GapAnalystAgent(BaseAgent):
    """Works out what the first wave did not answer."""

    def __init__(self, llm: BaseChatModel | None = None, max_follow_ups: int = MAX_FOLLOW_UPS):
        config = AgentConfig(
            role=AgentRole.ANALYST,
            name="Gap analyst",
            description="Finds what the research missed",
            system_prompt=GAP_PROMPT.replace("__MAX__", str(max_follow_ups)),
        )
        super().__init__(config, llm)
        self.max_follow_ups = max_follow_ups

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        covered = "\n\n".join(
            f"Q: {item.get('question', '')}\nA: {str(item.get('content', ''))[:700]}"
            for item in state.research_findings
            if item.get("turn") == state.turn
        )
        gaps: list[dict[str, str]] = []
        try:
            payload = await call_json(
                self._json_sender(self._actor()),
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {
                        "role": "user",
                        "content": f"Task: {state.current_task}\n\nEstablished:\n{covered}",
                    },
                ],
                site="research.gap_analyst",
                expect=list,
            )
        except StructuredOutputError as error:
            logger.warning("gap analyst added no follow-ups: %s", error)
        else:
            gaps = questions_from(payload, self.max_follow_ups)

        existing = {item.get("question", "") for item in state.research_findings}
        gaps = [gap for gap in gaps if gap["question"] not in existing]

        plan = [
            {
                "id": len(state.plan) + index,
                "section": gap["question"],
                "brief": gap["why"],
                "wave": 2,
            }
            for index, gap in enumerate(gaps)
        ]
        span = time.time() - start
        return {
            "plan": list(state.plan) + plan,
            "agent_timings": {"gap_analyst": span},
            "messages": [self._message(f"Found {len(plan)} gaps to fill")],
        }


class ReportAgent(BaseAgent):
    """Writes the answer from every finding, with sources."""

    def __init__(self, llm: BaseChatModel | None = None):
        config = AgentConfig(
            role=AgentRole.WRITER,
            name="Report writer",
            description="Writes the researched answer",
            system_prompt=REPORT_PROMPT,
        )
        super().__init__(config, llm)

    async def execute(self, state: AgentState) -> dict[str, Any]:
        start = time.time()
        findings = [item for item in state.research_findings if item.get("turn") == state.turn]
        material = "\n\n".join(
            f"[{index + 1}] {item.get('question', '')}\n"
            f"{str(item.get('content', ''))[:900]}\n"
            f"Sources: {'; '.join((item.get('sources') or [])[:3]) or 'none captured'}"
            for index, item in enumerate(findings)
        )
        response = await self._ainvoke(
            [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": f"Task: {state.current_task}\n\nResearch:\n{material}",
                },
            ]
        )
        span = time.time() - start
        return {
            "final_report": str(response.content),
            "agent_timings": {"report": span},
            "messages": [self._message(f"Report written from {len(findings)} findings")],
        }
