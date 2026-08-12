"""The LangGraph state machine.

    START → brief → plan → dispatch ─(Send × n)→ researcher ─→ critique
                              ▲                                   │
                              └──────── gaps & rounds left ────────┘
                                                                  │ sufficient
                                          synthesize → verify → render → END

Three properties are worth pointing at:

* **Fan-out.** `dispatch` emits one `Send` per sub-question, so researchers run
  concurrently. Latency is the slowest researcher, not their sum.
* **A bounded cycle.** `critique` may route back to `dispatch` for another
  retrieval round, but only while `round < MUHAQQIQ_MAX_RESEARCH_ROUNDS`.
* **Dependency injection.** Nodes receive a `Deps` bundle, so the same graph runs
  against the offline reasoner, a hosted model, in-process tools or an MCP
  server, with no branching inside the nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .llm import ReasoningProvider, build_provider
from .mcp_client import build_registry
from .nodes.brief import brief_node
from .nodes.critique import critique_node, should_retry
from .nodes.plan import dispatch_node, fan_out, plan_node
from .nodes.research import researcher_node
from .nodes.synthesize import synthesize_node
from .nodes.verify import render_node, verify_node
from .observability import Trace
from .skills import SkillLibrary
from .state import ResearchState
from .store import RunStore


@dataclass
class Deps:
    """Everything a node needs, injected rather than imported."""

    settings: Settings
    provider: ReasoningProvider
    tools: Any
    skills: SkillLibrary
    trace: Trace
    store: RunStore | None = None
    warnings: list[str] | None = None


def build_deps(
    settings: Settings | None = None,
    *,
    provider: ReasoningProvider | None = None,
    tools: Any | None = None,
    store: RunStore | None = None,
) -> Deps:
    settings = settings or get_settings()
    skills = SkillLibrary.load(settings.skills_dir)
    warnings: list[str] = list(settings.degraded_reasons)
    store = store if store is not None else RunStore(settings.db_path)
    if tools is None:
        tools, tool_warnings = build_registry(settings, store)
        warnings.extend(tool_warnings)
    return Deps(
        settings=settings,
        provider=provider or build_provider(settings, skills),
        tools=tools,
        skills=skills,
        trace=Trace(),
        store=store,
        warnings=warnings,
    )


def build_graph(deps: Deps, checkpointer: Any | None = None):
    """Wire and compile the research graph."""
    builder = StateGraph(ResearchState)

    builder.add_node("brief", partial(brief_node, deps=deps))
    builder.add_node("plan", partial(plan_node, deps=deps))
    builder.add_node("dispatch", partial(dispatch_node, deps=deps))
    builder.add_node("researcher", partial(researcher_node, deps=deps))
    builder.add_node("critique", partial(critique_node, deps=deps))
    builder.add_node("synthesize", partial(synthesize_node, deps=deps))
    builder.add_node("verify", partial(verify_node, deps=deps))
    builder.add_node("render", partial(render_node, deps=deps))

    builder.add_edge(START, "brief")
    builder.add_edge("brief", "plan")
    builder.add_edge("plan", "dispatch")
    builder.add_conditional_edges("dispatch", fan_out, ["researcher"])
    builder.add_edge("researcher", "critique")
    builder.add_conditional_edges("critique", should_retry, ["dispatch", "synthesize"])
    builder.add_edge("synthesize", "verify")
    builder.add_edge("verify", "render")
    builder.add_edge("render", END)

    return builder.compile(checkpointer=checkpointer)


def graph_mermaid() -> str:
    """The architecture diagram, generated from the graph rather than drawn by hand."""
    return """flowchart TD
    START([user question]) --> BRIEF[brief<br/>normalise + scope]
    BRIEF --> PLAN[plan<br/>decompose into sub-questions]
    PLAN --> DISPATCH{dispatch<br/>fan out}
    DISPATCH -->|Send q1| R1[researcher q1]
    DISPATCH -->|Send q2| R2[researcher q2]
    DISPATCH -->|Send qN| R3[researcher qN]
    R1 --> CRIT[critique<br/>gap analysis]
    R2 --> CRIT
    R3 --> CRIT
    CRIT -->|gaps and rounds left| DISPATCH
    CRIT -->|sufficient| SYN[synthesize<br/>renumber + write]
    SYN --> VER[verify<br/>citation audit]
    VER --> REN[render<br/>markdown + HTML]
    REN --> DONE([cited report])
"""
