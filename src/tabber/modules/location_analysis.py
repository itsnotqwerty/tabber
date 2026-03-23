"""Location Analysis module.

Takes the final OSINTBundle from the identification feedback loop and makes a
single LLM call to synthesise all gathered data into a LocationResult.
"""

from __future__ import annotations

from tabber import llm
from tabber.models import LocationResult, OSINTBundle

_SYSTEM = (
    "You are an expert intelligence analyst specialising in open-source intelligence "
    "and location analysis. Synthesise OSINT data to determine the most likely current "
    "or most recent physical location of a public figure. Be precise and evidence-based. "
    "Respond ONLY with a valid JSON object."
)


def analyse(bundle: OSINTBundle, progress=None) -> LocationResult:
    task = None
    if progress is not None:
        task = progress.add_task("Analysing location...", total=None)

    snippets = [
        f"[{r.source_name.upper()}]\n{r.raw_text[:500]}"
        for r in bundle.results
        if r.raw_text
    ]
    osint_text = "\n\n".join(snippets) if snippets else "(no data gathered)"
    source_names = [r.source_name for r in bundle.results]

    prompt = f"""Analyse the following OSINT data about {bundle.person.name} and determine their most likely current or most recent physical location.

Person profile:
- Name: {bundle.person.name}
- Known roles: {", ".join(bundle.person.known_roles)}
- Notes: {bundle.person.disambiguation_notes}

OSINT data:
{osint_text}

Return a JSON object with these exact fields:
- location: string (city and country, or more specific if known — e.g. "Austin, Texas, USA", "Deceased" if they have died, or "Unknown" if there is no location information or they are missing)
- confidence: float 0.0–1.0 (how confident you are; be conservative)
- reasoning: string (concise explanation referencing specific evidence from the data)
- sources: array of strings (source names that contributed; choose from {source_names})

Example:
{{"location": "Austin, Texas, USA", "confidence": 0.82, "reasoning": "Multiple recent news articles place the subject at Gigafactory Texas.", "sources": ["news", "twitter"]}}"""

    result = llm.complete(prompt, system=_SYSTEM, response_format=LocationResult)

    if progress is not None and task is not None:
        progress.update(
            task,
            description="[green]✓[/green] Location analysis complete",
            completed=1,
            total=1,
        )

    return result
