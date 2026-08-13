"""Command line interface.

    muhaqqiq research "your question"     run the agent
    muhaqqiq serve                        start the HTTP API
    muhaqqiq doctor                       show the resolved configuration
    muhaqqiq skills                       list the Agent Skills that loaded
    muhaqqiq tools                        list the tool surface
    muhaqqiq graph                        print the architecture as Mermaid
    muhaqqiq runs                         list previous runs
    muhaqqiq show RUN_ID                  re-print a stored report
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import run_research, write_outputs
from .config import get_settings
from .graph import build_deps, graph_mermaid
from .schemas import Verdict
from .skills import SkillLibrary
from .store import RunStore

app = typer.Typer(
    add_completion=False,
    help="Muhaqqiq — a multi-agent research agent that refuses to publish an uncited claim.",
)
console = Console()

VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.PASS_WITH_WARNINGS: "bold yellow",
    Verdict.FAIL: "bold red",
}


@app.command()
def research(
    question: str = typer.Argument(..., help="The research question."),
    depth: str = typer.Option("standard", "--depth", "-d", help="quick | standard | deep"),
    out: Path = typer.Option(None, "--out", "-o", help="Directory for the report files."),
    show: bool = typer.Option(True, "--show/--no-show", help="Print the report to the terminal."),
    trace: bool = typer.Option(False, "--trace", help="Print the node-by-node trace."),
    language: str = typer.Option(None, "--lang", help="Output language hint, e.g. 'ar'."),
) -> None:
    """Research a question and produce a cited report."""
    settings = get_settings()
    for warning in settings.degraded_reasons:
        console.print(f"[yellow]![/yellow] {warning}")

    with console.status("[bold]researching…", spinner="dots"):
        result = run_research(question, depth=depth, language=language, settings=settings)

    if show:
        console.print(Markdown(result.markdown))

    if trace:
        table = Table(title="Trace", show_lines=False, header_style="dim")
        table.add_column("t+ms", justify="right", style="dim")
        table.add_column("node")
        table.add_column("event")
        for event in result.events:
            table.add_row(str(event.get("t_ms", "")), event.get("node", ""), event.get("message", ""))
        console.print(table)

    verdict = result.verification.verdict
    console.print(
        Panel(
            f"[{VERDICT_STYLE[verdict]}]{verdict.value}[/] · "
            f"coverage {result.verification.citation_coverage:.0%} · "
            f"{result.verification.source_diversity} sources cited · "
            f"{result.meta.tool_calls} tool calls · {result.meta.duration_ms} ms",
            title=f"run {result.meta.run_id}",
            expand=False,
        )
    )

    paths = write_outputs(result, out or settings.output_dir)
    for label, path in paths.items():
        console.print(f"  [dim]{label:9}[/dim] {path}")

    if verdict is Verdict.FAIL:
        raise typer.Exit(code=2)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("muhaqqiq.api:app", host=host, port=port, reload=reload)


@app.command()
def doctor() -> None:
    """Show how the agent is configured and what it will actually use."""
    settings = get_settings()
    table = Table(title="Muhaqqiq configuration", header_style="dim")
    table.add_column("setting")
    table.add_column("value")
    rows = [
        ("LLM provider (requested)", settings.llm_provider),
        ("LLM provider (effective)", settings.effective_llm_provider),
        ("Model", settings.model),
        ("Search provider (requested)", settings.search_provider),
        ("Search provider (effective)", settings.effective_search_provider),
        ("Corpus directory", str(settings.corpus_dir)),
        ("Skills directory", str(settings.skills_dir)),
        ("Max sub-questions", str(settings.max_subquestions)),
        ("Max research rounds", str(settings.max_research_rounds)),
        ("Min citation coverage", f"{settings.min_citation_coverage:.0%}"),
        ("MCP tools", "on" if settings.use_mcp else "off (in-process)"),
        ("LangSmith tracing", "on" if settings.langsmith_tracing else "off"),
        ("Database", str(settings.db_path)),
    ]
    for key, value in rows:
        table.add_row(key, str(value))
    console.print(table)
    for warning in settings.degraded_reasons:
        console.print(f"[yellow]![/yellow] {warning}")
    if not settings.degraded_reasons:
        console.print("[green]✓[/green] configuration is internally consistent")


@app.command()
def skills() -> None:
    """List the Agent Skills that were loaded from disk."""
    library = SkillLibrary.load(get_settings().skills_dir)
    if not library.skills:
        console.print("[yellow]no skills found[/yellow]")
        raise typer.Exit(code=1)
    table = Table(title=f"{len(library)} skills loaded", header_style="dim")
    table.add_column("skill")
    table.add_column("stages")
    table.add_column("description")
    for skill in library.skills:
        table.add_row(skill.name, ", ".join(skill.stages) or "—", skill.description)
    console.print(table)


@app.command()
def tools() -> None:
    """List the tool surface and describe the corpus behind it."""
    deps = build_deps()
    table = Table(title="Tool surface", header_style="dim")
    table.add_column("tool")
    table.add_column("description")
    for spec in deps.tools.specs():
        table.add_row(spec["name"], spec["description"])
    console.print(table)
    console.print(deps.tools.corpus_stats())


@app.command()
def graph() -> None:
    """Print the agent graph as a Mermaid diagram."""
    console.print(graph_mermaid())


@app.command()
def runs(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """List previous runs."""
    store = RunStore(get_settings().db_path)
    rows = store.list_runs(limit)
    if not rows:
        console.print("[dim]no runs yet[/dim]")
        return
    table = Table(header_style="dim")
    table.add_column("run id")
    table.add_column("when")
    table.add_column("verdict")
    table.add_column("cov", justify="right")
    table.add_column("question")
    for row in rows:
        table.add_row(
            row["run_id"],
            row["created_at"],
            row["verdict"],
            f"{row['coverage']:.0%}",
            row["question"][:60],
        )
    console.print(table)


@app.command()
def show(run_id: str = typer.Argument(..., help="Run id from `muhaqqiq runs`.")) -> None:
    """Re-print a stored report."""
    result = RunStore(get_settings().db_path).get_run(run_id)
    if result is None:
        console.print(f"[red]no such run:[/red] {run_id}")
        raise typer.Exit(code=1)
    console.print(Markdown(result.markdown))


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
