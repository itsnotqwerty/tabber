#!/usr/bin/env python3
"""Tabber — find the most likely recent location of a famous person."""
from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


@click.group()
def cli() -> None:
    """Tabber — find the most likely recent location of a famous person."""


@cli.command()
@click.argument("name")
@click.option("--verbose", "-v", is_flag=True, help="Show per-iteration OSINT details.")
@click.option(
    "--max-iter",
    "-n",
    default=None,
    type=int,
    help="Override max feedback loop iterations (default: from config, usually 3).",
)
def lookup(name: str, verbose: bool, max_iter: int | None) -> None:
    """Look up the most likely recent location of NAME."""
    from modules import identification, location_analysis

    try:
        console.print(f"\n[bold]Tabber[/bold] — looking up [cyan]{name}[/cyan]\n")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            bundle = identification.run(
                name, progress=progress, verbose=verbose, max_iter=max_iter
            )
            result = location_analysis.analyse(bundle, progress=progress)

        if result.confidence >= 0.7:
            conf_style = "green"
        elif result.confidence >= 0.4:
            conf_style = "yellow"
        else:
            conf_style = "red"

        console.print(
            Panel.fit(
                f"[bold]Location:[/bold]   {result.location}\n"
                f"[bold]Confidence:[/bold] [{conf_style}]{result.confidence:.0%}[/{conf_style}]\n"
                f"[bold]Reasoning:[/bold]  {result.reasoning}\n"
                f"[bold]Sources:[/bold]    {', '.join(result.sources)}",
                title=f"[bold]{bundle.person.name}[/bold]",
                border_style="blue",
            )
        )
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.group()
def config() -> None:
    """Manage Tabber configuration (~/.tabber/config.json)."""


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration KEY to VALUE."""
    import config as cfg_module

    cfg_module.set_key(key, value)
    click.echo(f"Set {key}.")


@config.command("show")
def config_show() -> None:
    """Show the current configuration (secrets are masked)."""
    import config as cfg_module

    cfg = cfg_module.load()
    display = cfg_module.masked(cfg)
    table = Table(title="Tabber Config", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for k, v in display.items():
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    # Allow `python tabber.py "Name"` (and `python tabber.py -v "Name"` etc.) as
    # shorthand for `python tabber.py lookup "Name"`.  Find the first non-flag
    # positional argument; if it is not a known subcommand, inject "lookup" at
    # position 1 so all flags are forwarded to the lookup command.
    _known_subcommands = {"lookup", "config"}
    _positional_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if _positional_args and _positional_args[0] not in _known_subcommands:
        sys.argv.insert(1, "lookup")
    cli()
