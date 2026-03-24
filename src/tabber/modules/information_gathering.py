from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tabber import config as cfg_module
from tabber.gatherers.events import EventsGatherer
from tabber.gatherers.instagram import InstagramGatherer
from tabber.gatherers.news import NewsGatherer
from tabber.gatherers.reddit import RedditGatherer
from tabber.gatherers.twitter import TwitterGatherer
from tabber.gatherers.wikipedia import WikipediaGatherer
from tabber.models import GathererResult, OSINTBundle, PersonProfile

_ALL_GATHERERS = [
    NewsGatherer(),
    WikipediaGatherer(),
    TwitterGatherer(),
    InstagramGatherer(),
    RedditGatherer(),
    EventsGatherer(),
]

# ---------------------------------------------------------------------------
# Prompt-injection screening
# ---------------------------------------------------------------------------

_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|all|the\s+above)\s+instructions?"
    r"|disregard\s+(previous|all|the\s+above)"
    r"|you\s+are\s+now"
    r"|new\s+instructions?\s*:"
    r"|system\s*:)",
    re.IGNORECASE,
)


def _screen_content(text: str) -> str:
    """Redact lines from external sources that appear to contain prompt-injection attempts."""
    clean_lines = []
    for line in text.splitlines():
        if _INJECTION_RE.search(line):
            clean_lines.append("[REDACTED]")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines)


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
                    text_len = len(result.raw_text)
                    progress.advance(gather_task)
                    if count:
                        progress.console.log(
                            f"  [bold]{gname}[/bold] — "
                            f"[green]{count} result{'s' if count != 1 else ''}[/green]"
                            f" ({text_len} chars)"
                        )
                    else:
                        progress.console.log(
                            f"  [bold]{gname}[/bold] — [dim]no results[/dim]"
                        )
                elif verbose:
                    import click

                    click.echo(
                        f"  [{gname}] {len(result.items)} items, "
                        f"{len(result.raw_text)} chars of raw_text"
                    )
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

    # Screen all gatherer output for prompt-injection patterns before storing.
    screened: list[GathererResult] = []
    for r in results:
        if r.raw_text:
            r = r.model_copy(update={"raw_text": _screen_content(r.raw_text)})
        screened.append(r)

    return OSINTBundle(person=profile, results=screened, iteration=iteration)
