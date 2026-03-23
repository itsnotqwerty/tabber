from __future__ import annotations

import requests

from gatherers.base import BaseGatherer
from models import GathererResult, PersonProfile

_HEADERS = {"User-Agent": "tabber/1.0 (OSINT research tool)"}
_TIMEOUT = 10


class WikipediaGatherer(BaseGatherer):
    name = "wikipedia"

    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult:
        items: list[dict] = []
        raw_parts: list[str] = []

        # Wikipedia REST summary
        try:
            slug = profile.name.replace(" ", "_")
            resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            if resp.ok:
                data = resp.json()
                extract = data.get("extract", "")
                items.append({"source": "wikipedia_summary", "text": extract})
                raw_parts.append(extract)
        except Exception:
            pass

        # Wikidata: look up residence / country of citizenship
        try:
            search_resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": profile.name,
                    "language": "en",
                    "format": "json",
                    "limit": 1,
                },
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            if search_resp.ok:
                results = search_resp.json().get("search", [])
                if results:
                    qid = results[0]["id"]
                    # Fetch only specific properties to avoid huge payloads
                    props_resp = requests.get(
                        "https://www.wikidata.org/w/api.php",
                        params={
                            "action": "wbgetclaims",
                            "entity": qid,
                            "property": "P551|P27|P19|P937",  # residence|citizenship|birth place|work location
                            "format": "json",
                        },
                        headers=_HEADERS,
                        timeout=_TIMEOUT,
                    )
                    if props_resp.ok:
                        claims = props_resp.json().get("claims", {})
                        label_map = {
                            "P551": "residence",
                            "P27": "country_of_citizenship",
                            "P19": "place_of_birth",
                            "P937": "work_location",
                        }
                        for prop, label in label_map.items():
                            if prop in claims:
                                val = (
                                    claims[prop][0]
                                    .get("mainsnak", {})
                                    .get("datavalue", {})
                                    .get("value", {})
                                )
                                if isinstance(val, dict) and "id" in val:
                                    items.append(
                                        {
                                            "source": f"wikidata_{label}",
                                            "qid": val["id"],
                                        }
                                    )
                                    raw_parts.append(f"wikidata {label}: {val['id']}")
        except Exception:
            pass

        return GathererResult(
            source_name=self.name,
            items=items,
            raw_text="\n".join(raw_parts),
        )
