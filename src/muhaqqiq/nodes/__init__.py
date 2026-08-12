"""Graph nodes.

Each module here is one stage of the research pipeline. Every node is a pure
function of `(state, deps) -> state update`, which is what makes the run
checkpointable and each stage independently testable.
"""

from .brief import brief_node
from .critique import critique_node
from .plan import dispatch_node, fan_out, plan_node
from .research import researcher_node
from .synthesize import renumber_sources, synthesize_node
from .verify import render_node, verify_node

__all__ = [
    "brief_node",
    "plan_node",
    "dispatch_node",
    "fan_out",
    "researcher_node",
    "critique_node",
    "synthesize_node",
    "renumber_sources",
    "verify_node",
    "render_node",
]
