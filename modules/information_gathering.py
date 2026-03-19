from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg_module
from gatherers.events import EventsGatherer
from gatherers.instagram import InstagramGatherer
from gatherers.news import NewsGatherer
from gatherers.reddit import RedditGatherer
from gatherers.twitter import TwitterGatherer
from gatherers.wikipedia import WikipediaGatherer
from models import GathererResult, OSINTBundle, PersonProfile

_ALL_GATHERERS = [
    NewsGatherer(),
    WikipediaGatherer(),
    TwitterGatherer(),
    InstagramGatherer(),
    RedditGatherer(),
    EventsGatherer(),
]


def gather(
    profile: PersonProfile,
    hints: list[str],
    iteration: int = 0,
    verbose: bool = False,
    progress=None,
) -> OSINTBundle:
    cfg = cfg_module.load()
    enabled = [g for g in _ALL_GATHERERS if g.is_configured(cfg)]
    skipped = [g.name for g in _ALL_GATHERERS if not g.is_configured(cfg)]

    if skipped:
        msg = f"Skipping unconfigured sources: {', '.join(skipped)}"
        if progress is not None:
            progress.console.log(f"[dim]{msg}[/dim]")
        elif verbose:
            import click
            click.echo(f"  [info] {msg}")

    gather_task = None
    if progress is not None:
        gather_task = progress.add_task(
            f"Searching {len(enabled)} sources (iteration {iteration})",
            total=len(enabled),
        )

    results: list[GathererResult] = []

    with ThreadPoolExecutor(max_workers=max(1, len(enabled))) as executor:
        futures = {executor.submit(g.gather, profile, hints): g.name for g in enabled}
        for future in as_completed(futures):
            gname = futures[future]
            try:
                result = future.result(timeout=30)
                results.append(result)
                if progress is not None:
                    count = len(result.items)
                    progress.advance(gather_task)
                    if count:
                        progress.console.log(
                            f"  [bold]{gname}[/bold] — "
                            f"[green]{count} result{'s' if count != 1 else ''}[/green]"
                        )
                    else:
                        progress.console.log(
                            f"  [bold]{gname}[/bold] — [dim]no results[/dim]"
                        )
                elif verbose:
                    import click
                    click.echo(f"  [{gname}] {len(result.items)} items gathered")
            except Exception as exc:
                if progress is not None:
                    progress.advance(gather_task)
                    progress.console.log(
                        f"  [bold]{gname}[/bold] — [red]error: {exc}[/red]"
                    )
                else:
                    import click
                    click.echo(f"  [{gname}] error: {exc}", err=True)

    if progress is not None and gather_task is not None:
        progress.update(
            gather_task,
            description=f"[green]✓[/green] Sources searched (iteration {iteration})",
        )

    return OSINTBundle(person=profile, results=results, iteration=iteration)
