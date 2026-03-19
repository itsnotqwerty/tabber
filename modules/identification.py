"""Identification module.

Responsibilities:
1. Disambiguate the raw input name → PersonProfile via LLM.
2. Drive the feedback loop:
   a. Generate search hints (LLM).
   b. Call information_gathering.gather() to collect OSINT.
   c. Ask LLM whether the bundle contains enough location signal.
   d. If yes, exit. If no and iterations remain, refine hints and repeat.
3. Return the best OSINTBundle to the caller.
"""
from __future__ import annotations

import json
from typing import Optional

import llm
from models import OSINTBundle, PersonProfile

_SYS_DISAMBIGUATE = (
    "You are an intelligence research assistant. "
    "Disambiguate a name and produce a structured profile of the most publicly "
    "prominent person with that name. Respond ONLY with a valid JSON object."
)

_SYS_HINTS = (
    "You are an intelligence research assistant. "
    "Generate targeted search query hints to locate a public figure. "
    "Respond ONLY with a valid JSON array of strings."
)

_SYS_EVALUATE = (
    "You are an intelligence analyst specialising in OSINT. "
    "Evaluate whether gathered data contains enough recent location signal. "
    "Respond ONLY with a valid JSON object."
)


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    return json.loads(text)


def _disambiguate(name: str) -> PersonProfile:
    prompt = f"""Disambiguate the name "{name}".

Return a JSON object with these exact fields:
- name: string (canonical full name)
- aliases: array of strings (other names, stage names, handles)
- known_roles: array of strings (e.g. "CEO of Tesla", "musician", "politician")
- disambiguation_notes: string (brief note on who this person is and why they are notable)

Example:
{{"name": "Elon Musk", "aliases": ["@elonmusk"], "known_roles": ["CEO of Tesla", "CEO of SpaceX", "owner of X"], "disambiguation_notes": "South African-born American entrepreneur and businessman."}}"""

    response = llm.complete(prompt, system=_SYS_DISAMBIGUATE)
    return PersonProfile(**_parse_json(response))


def _generate_hints(
    profile: PersonProfile, prior_bundle: Optional[OSINTBundle]
) -> list[str]:
    context = "(none yet)"
    if prior_bundle:
        snippets = [
            f"[{r.source_name}]: {r.raw_text[:300]}"
            for r in prior_bundle.results
            if r.raw_text
        ]
        context = "\n".join(snippets[:6]) or "(no content)"

    prompt = f"""Generate targeted search query hints to find the most recent physical location of:

Name: {profile.name}
Roles: {", ".join(profile.known_roles)}
Notes: {profile.disambiguation_notes}

Prior OSINT data:
{context}

Return a JSON array of exactly 5 short search hint strings optimised for news and social media.
Example: ["visited Texas", "appearance NYC", "conference Dubai", "spotted London", "tour Europe"]"""

    response = llm.complete(prompt, system=_SYS_HINTS)
    hints = _parse_json(response)
    return hints if isinstance(hints, list) else []


def _has_enough_signal(bundle: OSINTBundle) -> bool:
    all_text = "\n".join(r.raw_text for r in bundle.results if r.raw_text)
    if not all_text.strip():
        return False

    prompt = f"""Person: {bundle.person.name}
OSINT data (iteration {bundle.iteration}):
{all_text[:3000]}

Is there enough location information here to determine this person's most recent or current physical location with reasonable confidence?

Respond with JSON: {{"sufficient": true/false, "reason": "brief explanation"}}"""

    response = llm.complete(prompt, system=_SYS_EVALUATE)
    data = _parse_json(response)
    return bool(data.get("sufficient", False))


def _spinner_start(progress, description: str):
    """Add an indeterminate spinner task; returns task id."""
    if progress is None:
        return None
    return progress.add_task(description, total=None)


def _spinner_done(progress, task_id, description: str) -> None:
    """Mark a spinner task as completed with an updated description."""
    if progress is None or task_id is None:
        return
    progress.update(task_id, description=description, completed=1, total=1)


def run(
    name: str,
    verbose: bool = False,
    max_iter: Optional[int] = None,
    progress=None,
) -> OSINTBundle:
    from modules import information_gathering
    import config as cfg_module

    cfg = cfg_module.load()
    max_iterations: int = max_iter if max_iter is not None else int(cfg.get("max_iterations", 3))

    # --- Disambiguation ---
    task = _spinner_start(progress, f"Identifying [bold]{name}[/bold]...")
    if task is None and verbose:
        import click
        click.echo(f"[identification] Disambiguating '{name}'...")

    profile = _disambiguate(name)

    roles_str = ", ".join(profile.known_roles[:2])
    _spinner_done(
        progress, task,
        f"[green]✓[/green] [bold]{profile.name}[/bold]" + (f" ({roles_str})" if roles_str else ""),
    )
    if task is None and verbose:
        import click
        click.echo(f"[identification] → {profile.name} ({', '.join(profile.known_roles)})")

    bundle: Optional[OSINTBundle] = None

    for i in range(max_iterations):
        iter_label = f"iter {i + 1}/{max_iterations}"

        # --- Generate hints ---
        htask = _spinner_start(progress, f"Generating search hints ({iter_label})...")
        if htask is None and verbose:
            import click
            click.echo(f"\n[loop {i + 1}/{max_iterations}] Generating query hints...")

        hints = _generate_hints(profile, bundle)

        _spinner_done(progress, htask, f"[green]✓[/green] Search hints ready ({iter_label})")
        if htask is None and verbose:
            import click
            click.echo(f"  hints: {hints}")
            click.echo(f"[loop {i + 1}/{max_iterations}] Gathering OSINT...")
        elif verbose and progress is not None:
            progress.console.log(f"  hints: {hints}")

        # --- Gather ---
        bundle = information_gathering.gather(
            profile, hints, iteration=i + 1, verbose=verbose, progress=progress
        )

        # --- Evaluate signal ---
        etask = _spinner_start(progress, f"Evaluating signal ({iter_label})...")
        if etask is None and verbose:
            import click
            click.echo(f"[loop {i + 1}/{max_iterations}] Evaluating signal...")

        sufficient = _has_enough_signal(bundle)

        if sufficient:
            _spinner_done(
                progress, etask,
                f"[green]✓[/green] Sufficient location signal found ({iter_label})",
            )
            if etask is None and verbose:
                import click
                click.echo(f"[identification] Sufficient signal at iteration {i + 1}. Exiting loop.")
            break
        else:
            _spinner_done(
                progress, etask,
                f"[yellow]↻[/yellow] Signal insufficient — refining ({iter_label})",
            )
            if etask is None and verbose:
                import click
                click.echo("[identification] Signal insufficient — refining...")

    # Return last bundle (may be None only if max_iterations == 0).
    if bundle is None:
        bundle = information_gathering.gather(
            profile, [], iteration=0, verbose=verbose, progress=progress
        )

    return bundle
