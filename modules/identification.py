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
from models import OSINTBundle, PersonProfile, SignalEvaluation

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
    "Score how confident you are that the gathered data is sufficient to determine "
    "where this person is or has recently been. Known residences, upcoming events, "
    "recent travel mentions, and confirmed public appearances all contribute positively. "
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


def _generate_verification_hints(
    profile: PersonProfile, evaluation: SignalEvaluation
) -> list[str]:
    prompt = f"""You are verifying a suspected location for an intelligence target.

Person: {profile.name}
Suspected location signal: {evaluation.reason}

Generate 5 targeted search query strings to VERIFY or CONFIRM the suspected location
with additional independent evidence. Focus on:
- Direct sightings or confirmed appearances at that specific place
- Recent news datelines from that location
- Official schedules, announcements, or events tied to that place
- Confirmed locations of events the person is known to be attending

Return a JSON array of exactly 5 short search hint strings.
Example: ["confirmed Austin March 2026", "Gigafactory Texas sighting", "SpaceX Boca Chica appearance", "Superbowl 2026 location", "Elon Musk conference Texas"]"""

    response = llm.complete(prompt, system=_SYS_HINTS)
    hints = _parse_json(response)
    return hints if isinstance(hints, list) else []


def _evaluate_signal(
    bundle: OSINTBundle, verbose: bool = False, progress=None
) -> SignalEvaluation:
    def _log(msg: str) -> None:
        if not verbose:
            return
        if progress is not None:
            progress.console.log(msg)
        else:
            import click

            click.echo(msg)

    all_text = "\n".join(r.raw_text for r in bundle.results if r.raw_text)

    _log(
        f"[signal-eval] all_text length: {len(all_text)} chars"
        + (f", preview: {all_text[:200]!r}" if all_text else "")
    )

    if not all_text.strip():
        _log("[signal-eval] Early return confidence=0 — no gathered text at all.")
        return SignalEvaluation(confidence=0.0, reason="No gathered text")

    prompt = f"""Person: {bundle.person.name}
OSINT data (iteration {bundle.iteration}):
{all_text[:6000]}

Score your confidence (0.0–1.0) that this data is sufficient to determine where
this person is or has recently been. Known residences, upcoming scheduled events,
recent travel mentions, and confirmed public appearances all count.

Respond with JSON: {{"confidence": <float 0.0-1.0>, "reason": "brief explanation"}}
Example: {{"confidence": 0.87, "reason": "Multiple recent articles place the subject in London."}}"""

    response = llm.complete(prompt, system=_SYS_EVALUATE)

    _log(f"[signal-eval] LLM raw response: {response!r}")

    try:
        data = _parse_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        _log(f"[signal-eval] JSON parse failed ({exc}); raw: {response!r}")
        # Substantial text present — assume workable confidence rather than burn another iteration.
        assumed = 0.85 if len(all_text) > 500 else 0.0
        return SignalEvaluation(
            confidence=assumed,
            reason=f"LLM response unparseable; confidence assumed from text volume ({len(all_text)} chars)",
        )

    if "confidence" not in data:
        _log(
            f"[signal-eval] 'confidence' key missing from response; "
            f"got keys: {list(data.keys())}"
        )
        return SignalEvaluation(
            confidence=0.0,
            reason=f"Unexpected LLM response keys: {list(data.keys())}",
        )

    raw_conf = float(data["confidence"])
    # Normalise in case the LLM returns 0–100 despite instructions.
    if raw_conf > 1.0:
        raw_conf /= 100.0
    confidence = max(0.0, min(1.0, raw_conf))

    return SignalEvaluation(confidence=confidence, reason=str(data.get("reason", "")))


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
    max_iterations: int = (
        max_iter if max_iter is not None else int(cfg.get("max_iterations", 3))
    )

    # --- Disambiguation ---
    task = _spinner_start(progress, f"Identifying [bold]{name}[/bold]...")
    if task is None and verbose:
        import click

        click.echo(f"[identification] Disambiguating '{name}'...")

    profile = _disambiguate(name)

    roles_str = ", ".join(profile.known_roles[:2])
    _spinner_done(
        progress,
        task,
        f"[green]✓[/green] [bold]{profile.name}[/bold]"
        + (f" ({roles_str})" if roles_str else ""),
    )
    if task is None and verbose:
        import click

        click.echo(
            f"[identification] → {profile.name} ({', '.join(profile.known_roles)})"
        )

    bundle: Optional[OSINTBundle] = None

    for i in range(max_iterations):
        # iter_label = f"iter {i + 1}/{max_iterations}"

        # --- Generate hints ---
        htask = _spinner_start(progress, f"Generating search hints...")
        if htask is None and verbose:
            import click

            click.echo(f"\n[loop {i + 1}/{max_iterations}] Generating query hints...")

        hints = _generate_hints(profile, bundle)

        _spinner_done(progress, htask, f"[green]✓[/green] Search hints ready")
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
        etask = _spinner_start(progress, f"Evaluating signal...")
        if etask is None and verbose:
            import click

            click.echo(f"[loop {i + 1}/{max_iterations}] Evaluating signal...")

        evaluation = _evaluate_signal(bundle, verbose=verbose, progress=progress)
        bundle.signal_evaluation = evaluation
        conf_pct = f"{evaluation.confidence:.0%}"

        if evaluation.confidence >= 0.80:
            _spinner_done(
                progress,
                etask,
                f"[green]✓[/green] Signal confidence {conf_pct} — sufficient",
            )
            if etask is None and verbose:
                import click

                click.echo(
                    f"[identification] Signal confidence {conf_pct} at iteration {i + 1}. Verifying..."
                )

            # --- Verification pass ---
            vtask = _spinner_start(progress, "Verifying suspected location...")
            if vtask is None and verbose:
                import click

                click.echo("[identification] Running verification gather...")

            verification_hints = _generate_verification_hints(profile, evaluation)
            if verbose and progress is not None:
                progress.console.log(f"  verification hints: {verification_hints}")
            elif vtask is None and verbose:
                import click

                click.echo(f"  verification hints: {verification_hints}")

            verification_bundle = information_gathering.gather(
                profile,
                verification_hints,
                iteration=i + 2,
                verbose=verbose,
                progress=progress,
            )
            bundle.results.extend(verification_bundle.results)

            evaluation = _evaluate_signal(bundle, verbose=verbose, progress=progress)
            bundle.signal_evaluation = evaluation

            if evaluation.confidence >= 0.85:
                _spinner_done(progress, vtask, "[green]✓[/green] Location verified")
                if vtask is None and verbose:
                    import click

                    click.echo("[identification] Verification complete. Exiting loop.")
                break
        else:
            _spinner_done(
                progress,
                etask,
                f"[yellow]↻[/yellow] Signal confidence {conf_pct} — refining",
            )
            if etask is None and verbose:
                import click

                click.echo(
                    f"[identification] Signal confidence {conf_pct} — refining..."
                )

    # Return last bundle (may be None only if max_iterations == 0).
    if bundle is None:
        bundle = information_gathering.gather(
            profile, [], iteration=0, verbose=verbose, progress=progress
        )

    return bundle
