"""Tracing.

Two layers, both optional and both cheap:

* An always-on in-process `Trace` that records a timestamped event per graph
  node. It ends up inside the `RunResult`, so the delivered artefact carries its
  own audit trail — you can see what the agent did without a dashboard.
* LangSmith, enabled by setting `LANGSMITH_TRACING=true` and a key. If the
  package is absent the decorator degrades to a no-op rather than exploding.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Trace:
    """Ordered list of what happened during a run."""

    events: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def event(self, node: str, message: str, **data: Any) -> None:
        self.events.append(
            {
                "t_ms": int((time.perf_counter() - self._t0) * 1000),
                "node": node,
                "message": message,
                **data,
            }
        )

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.events)


def langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def traceable(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """Wrap a graph node for LangSmith when configured, else return it unchanged."""

    def decorator(func: F) -> F:
        if not langsmith_enabled():
            return func
        try:
            from langsmith import traceable as ls_traceable  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            return func
        return ls_traceable(name=name, run_type=run_type)(func)  # type: ignore[return-value]

    return decorator
