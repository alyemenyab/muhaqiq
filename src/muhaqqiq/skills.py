"""Agent Skills loader.

A "skill" here is exactly what the course taught: a folder containing a
`SKILL.md` with YAML front-matter and a body of procedural instructions. The
skills are *not* baked into the Python source — they are loaded from disk at
runtime and injected as the system prompt for whichever graph stage declares
them. That means the agent's methodology can be edited, reviewed and diffed by
a domain expert who never opens a `.py` file.

Front-matter contract:

    ---
    name: report-writing
    description: How to turn verified evidence into a readable, cited report.
    stages: [synthesize]
    ---
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    stages: tuple[str, ...]
    body: str
    path: Path

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.name} ({', '.join(self.stages) or 'unscoped'})"


@dataclass
class SkillLibrary:
    skills: list[Skill] = field(default_factory=list)
    root: Path | None = None

    @classmethod
    def load(cls, root: Path | str) -> SkillLibrary:
        root_path = Path(root)
        skills: list[Skill] = []
        if root_path.is_dir():
            for skill_file in sorted(root_path.glob("*/SKILL.md")):
                parsed = _parse_skill(skill_file)
                if parsed is not None:
                    skills.append(parsed)
        return cls(skills=skills, root=root_path)

    def get(self, name: str) -> Skill | None:
        return next((s for s in self.skills if s.name == name), None)

    def for_stage(self, stage: str) -> list[Skill]:
        return [s for s in self.skills if stage in s.stages]

    def instructions_for(self, stage: str) -> str:
        """Concatenated skill bodies for a stage, ready to use as a system prompt."""
        chosen = self.for_stage(stage)
        if not chosen:
            return ""
        blocks = [f"## Skill: {s.name}\n{s.description}\n\n{s.body.strip()}" for s in chosen]
        return (
            "You are operating with the following loaded Agent Skills. "
            "Follow them literally.\n\n" + "\n\n---\n\n".join(blocks)
        )

    def names(self) -> list[str]:
        return [s.name for s in self.skills]

    def __len__(self) -> int:
        return len(self.skills)


def _parse_skill(path: Path) -> Skill | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = FRONTMATTER_RE.match(raw)
    meta: dict[str, str] = {}
    body = raw
    if match:
        body = raw[match.end() :]
        meta = _parse_frontmatter(match.group(1))
    name = meta.get("name") or path.parent.name
    stages = tuple(_split_list(meta.get("stages", "")))
    return Skill(
        name=name,
        description=meta.get("description", ""),
        stages=stages,
        body=body,
        path=path,
    )


def _parse_frontmatter(block: str) -> dict[str, str]:
    """Minimal YAML subset: `key: value` pairs, no nesting. Keeps deps at zero."""
    out: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def _split_list(value: str) -> list[str]:
    value = value.strip().strip("[]")
    return [item.strip().strip("'\"") for item in value.split(",") if item.strip()]
