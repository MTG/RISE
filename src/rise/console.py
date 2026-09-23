"""Shared Rich console and the small set of output primitives built on it.

Every experiment reports through these helpers rather than through ``print`` or a
console of its own, so that progress bars, section headers and result tables look
the same across the whole toolbox.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

#: One theme for the whole toolbox; semantic names only, never raw colours.
THEME = Theme(
    {
        "title": "bold cyan",
        "stage": "bold cyan",
        "metric": "bold green",
        "delta": "yellow",
        "path": "dim",
        "detail": "dim",
        "warning": "bold yellow",
        "error": "bold red",
    }
)

console = Console(theme=THEME, highlight=False)


def rule(title: str) -> None:
    """Announce a top-level stage of an experiment."""
    console.rule(f"[title]{title}[/title]")


def banner(text: str) -> None:
    """Announce a sub-stage, e.g. the rāga currently being processed."""
    console.print(Panel(f"[stage]{text}[/stage]", expand=False, border_style="cyan"))


def detail(text: str) -> None:
    """Report a secondary fact that is useful but not worth emphasising."""
    console.print(f"  [detail]{text}[/detail]")


def warn(text: str) -> None:
    console.print(f"[warning]![/warning] {text}")


def artifact(kind: str, path) -> None:
    """Report that a file has been written."""
    console.print(f"  [detail]{kind} written to[/detail] [path]{path}[/path]")


def progress(*, transient: bool = False) -> Progress:
    """A progress display with the layout used by every experiment."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def metrics_table(title: str, columns: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Render a result table with the standard column styling."""
    table = Table(title=title, title_style="title", header_style="bold")
    table.add_column(columns[0])
    for column in columns[1:]:
        table.add_column(column, justify="right", style="metric")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def parameters_table(parameters: Mapping[str, object]) -> None:
    """Echo the resolved configuration of an experiment before it starts."""
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="detail")
    table.add_column(justify="right")
    for key, value in sorted(parameters.items()):
        table.add_row(key.replace("_", " "), str(value))
    console.print(table)
