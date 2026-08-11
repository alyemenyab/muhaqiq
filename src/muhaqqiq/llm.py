"""Reasoning providers.

The graph never talks to a model directly — it talks to a `ReasoningProvider`,
which exposes one method per graph stage and always returns a validated Pydantic
object. Two implementations ship:

* `OfflineReasoner` (in `offline.py`) — deterministic, no network, no keys.
* `OpenRouterReasoner` — real models via OpenRouter, using JSON-schema
  structured outputs, with per-stage fallback to the offline reasoner so a
  provider outage degrades the report instead of destroying the run.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings
from .offline import OfflineReasoner
from .schemas import GapReport, Report, ResearchBrief, ResearchPlan, SubAnswer
from .skills import SkillLibrary

log = logging.getLogger("muhaqqiq.llm")

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ReasoningProvider(Protocol):
    """One method per graph stage. Every one returns a validated model."""

    name: str

    def brief(self, payload: dict[str, Any]) -> ResearchBrief: ...
    def plan(self, payload: dict[str, Any]) -> ResearchPlan: ...
    def answer_subquestion(self, payload: dict[str, Any]) -> SubAnswer: ...
    def critique(self, payload: dict[str, Any]) -> GapReport: ...
    def report(self, payload: dict[str, Any]) -> Report: ...


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #
class OpenRouterReasoner:
    """Hosted models through OpenRouter, constrained to a JSON schema."""

    name = "openrouter"

    def __init__(
        self,
        settings: Settings | None = None,
        skills: SkillLibrary | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.skills = skills or SkillLibrary.load(self.settings.skills_dir)
        self._fallback = OfflineReasoner()
        self._client = client
        self.calls = 0

    # -- public stage methods -------------------------------------------- #
    def brief(self, payload: dict[str, Any]) -> ResearchBrief:
        return self._stage(
            stage="brief",
            schema=ResearchBrief,
            task=(
                "Turn the user's raw question into a research brief. "
                f"Raw question:\n{payload.get('question', '')}\n\n"
                f"Requested depth: {payload.get('depth', 'standard')}."
            ),
            payload=payload,
            fallback=self._fallback.brief,
            fast=True,
        )

    def plan(self, payload: dict[str, Any]) -> ResearchPlan:
        brief = _dump(payload["brief"])
        limit = payload.get("max_subquestions", 4)
        return self._stage(
            stage="plan",
            schema=ResearchPlan,
            task=(
                "Decompose this brief into at most "
                f"{limit} orthogonal sub-questions.\n\nBRIEF:\n{json.dumps(brief, ensure_ascii=False, indent=2)}"
            ),
            payload=payload,
            fallback=self._fallback.plan,
            fast=True,
        )

    def answer_subquestion(self, payload: dict[str, Any]) -> SubAnswer:
        sub = _dump(payload["subquestion"])
        sources = [_dump(s) for s in payload.get("sources", [])]
        rendered = "\n\n".join(
            f"[{s['id']}] {s.get('title', '')} — {s.get('publisher', '')} ({s.get('url', '')})\n"
            f"{s.get('content') or s.get('snippet', '')}"
            for s in sources
        )
        return self._stage(
            stage="research",
            schema=SubAnswer,
            task=(
                f"SUB-QUESTION ({sub['id']}): {sub['question']}\n\n"
                f"SOURCES:\n{rendered or '(none retrieved)'}\n\n"
                "Answer using only these sources. Cite with [S#] markers. "
                f"Set subquestion_id to exactly '{sub['id']}'."
            ),
            payload=payload,
            fallback=self._fallback.answer_subquestion,
        )

    def critique(self, payload: dict[str, Any]) -> GapReport:
        subanswers = [_dump(s) for s in payload.get("subanswers", [])]
        return self._stage(
            stage="critique",
            schema=GapReport,
            task=(
                "Audit these sub-answers for coverage gaps. For each gap, propose "
                "follow-up search queries keyed by sub-question id.\n\n"
                f"{json.dumps(subanswers, ensure_ascii=False, indent=2)}"
            ),
            payload=payload,
            fallback=self._fallback.critique,
            fast=True,
        )

    def report(self, payload: dict[str, Any]) -> Report:
        body = {
            "brief": _dump(payload["brief"]),
            "plan": _dump(payload["plan"]),
            "subanswers": [_dump(s) for s in payload.get("subanswers", [])],
        }
        return self._stage(
            stage="synthesize",
            schema=Report,
            task=(
                "Write the final report from these verified sub-answers. Preserve every "
                "[S#] citation marker exactly.\n\n"
                f"{json.dumps(body, ensure_ascii=False, indent=2)}"
            ),
            payload=payload,
            fallback=self._fallback.report,
        )

    # -- plumbing ---------------------------------------------------------- #
    def _stage(
        self,
        *,
        stage: str,
        schema: type[T],
        task: str,
        payload: dict[str, Any],
        fallback,
        fast: bool = False,
    ) -> T:
        system = self.skills.instructions_for(stage) or (
            "You are a rigorous research agent. Answer only with valid JSON matching the schema."
        )
        try:
            return self._structured(system=system, user=task, schema=schema, fast=fast)
        except (LLMError, ValidationError, httpx.HTTPError) as exc:
            log.warning("stage %s fell back to the offline reasoner: %s", stage, exc)
            return fallback(payload)

    def _structured(self, *, system: str, user: str, schema: type[T], fast: bool = False) -> T:
        if not self.settings.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        model = self.settings.fast_model if fast else self.settings.model
        json_schema = _openai_compatible_schema(schema)
        request = {
            "model": model,
            "temperature": self.settings.temperature,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user
                    + "\n\nRespond with a single JSON object matching the required schema. "
                    "No prose, no markdown fence.",
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": False,
                    "schema": json_schema,
                },
            },
        }
        client = self._client or httpx.Client(timeout=self.settings.request_timeout)
        close_after = self._client is None
        try:
            response = client.post(
                f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/SDAIAAcademy",
                    "X-Title": "Muhaqqiq research agent",
                },
                json=request,
            )
            response.raise_for_status()
            data = response.json()
        finally:
            if close_after:
                client.close()
        self.calls += 1
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected OpenRouter payload: {data}") from exc
        return schema.model_validate(parse_json_object(content))


def parse_json_object(content: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response."""
    content = (content or "").strip()
    if not content:
        raise LLMError("empty model response")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    fenced = _JSON_BLOCK_RE.search(content)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"could not parse JSON from model output: {exc}") from exc
    raise LLMError("model output contained no JSON object")


def _openai_compatible_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Inline $defs and drop keywords structured-output endpoints reject."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].rsplit("/", 1)[-1]
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**defs.get(ref, {}), **merged})
            return {
                k: resolve(v)
                for k, v in node.items()
                if k not in {"default", "$schema", "additionalProperties"}
            }
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(raw)


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_provider(
    settings: Settings | None = None, skills: SkillLibrary | None = None
) -> ReasoningProvider:
    settings = settings or get_settings()
    if settings.effective_llm_provider == "openrouter":
        return OpenRouterReasoner(settings=settings, skills=skills)
    return OfflineReasoner()
