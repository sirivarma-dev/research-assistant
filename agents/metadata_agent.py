"""
METADATA AGENT
Pulls structured bibliographic metadata out of a paper:
  title, authors, abstract, publication year, keywords, DOI.

It works in two steps:
  1. Ask the LLM to read the paper's opening pages and return the metadata as
     strict JSON (papers put all of this near the top).
  2. Look the title up on CrossRef (a free scholarly API) to fill in / verify
     the DOI and year, which are often missing from the raw text.
"""
import json
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

from agents.base_agent import BaseAgent


class MetadataAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MetadataAgent",
            system_prompt=(
                "You extract bibliographic metadata from the opening of a research "
                "paper. Return ONLY a JSON object with EXACTLY these keys:\n"
                '  "title": string,\n'
                '  "authors": array of author name strings,\n'
                '  "abstract": string (the abstract text, or "" if none),\n'
                '  "year": string (4-digit publication year, or ""),\n'
                '  "keywords": array of keyword strings,\n'
                '  "doi": string (the DOI if printed in the text, else "").\n'
                "Use \"\" or [] when a field is not present. Output JSON only - no "
                "markdown, no commentary."
            ),
        )

    # ---------------------------------------------------------------- helpers --
    @staticmethod
    def _parse_json(raw):
        """Best-effort: strip code fences and grab the JSON object."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            # drop a leading 'json' language tag if present
            if raw.lower().startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        return json.loads(raw)

    @staticmethod
    def _crossref_lookup(title):
        """Look a title up on CrossRef. Returns {doi, year, authors} or {}."""
        if not title:
            return {}
        try:
            params = urllib.parse.urlencode(
                {"query.bibliographic": title, "rows": 5}
            )
            url = f"https://api.crossref.org/works?{params}"
            # CrossRef asks for a contact in the User-Agent ("polite pool").
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ResearchAssistant/1.0 (mailto:apps@automatr.tech)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            items = data.get("message", {}).get("items", [])
            if not items:
                return {}

            # CrossRef returns the CLOSEST matches even when the paper isn't in
            # its database (wrong DOI), and titles aren't unique (derivative
            # works share them). So: keep only candidates whose title is genuinely
            # similar, then prefer the most-cited one (the canonical paper).
            target = title.lower().strip()

            def title_sim(item):
                matched = (item.get("title") or [""])[0]
                return SequenceMatcher(None, target, matched.lower().strip()).ratio()

            # Strong threshold: a wrong DOI is worse than no DOI, so we'd rather
            # return nothing than a near-miss derivative work.
            good = [it for it in items if title_sim(it) >= 0.85]
            if not good:
                return {}
            # Among strong matches, prefer the most-cited (canonical) one.
            item = max(good, key=lambda it: it.get("is-referenced-by-count", 0))

            out = {}
            if item.get("DOI"):
                out["doi"] = item["DOI"]
            # Year lives under issued -> date-parts -> [[year, ...]]
            parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
            if parts and parts[0]:
                out["year"] = str(parts[0])
            authors = [
                " ".join(filter(None, [a.get("given"), a.get("family")]))
                for a in item.get("author", [])
            ]
            if authors:
                out["authors"] = [a for a in authors if a]
            return out
        except Exception:
            return {}

    # ----------------------------------------------------------------- public --
    def extract(self, text):
        """Return a metadata dict. Always returns the full set of keys."""
        meta = {
            "title": "", "authors": [], "abstract": "",
            "year": "", "keywords": [], "doi": "",
        }

        # The metadata is near the front; first ~6000 chars is plenty.
        try:
            raw = self.run(f"Paper text:\n\n{text[:6000]}", temperature=0.0)
            parsed = self._parse_json(raw)
            for key in meta:
                if key in parsed and parsed[key]:
                    meta[key] = parsed[key]
        except Exception:
            # If the LLM/JSON step fails, we still return the empty skeleton.
            pass

        # Fill in DOI / year (and authors as a fallback) from CrossRef.
        # A DOI printed in the paper itself (found by the LLM) is the most
        # trustworthy, so only fall back to CrossRef when we don't already have one.
        cr = self._crossref_lookup(meta.get("title", ""))
        if cr.get("doi") and not meta.get("doi"):
            meta["doi"] = cr["doi"]
        if cr.get("year") and not meta.get("year"):
            meta["year"] = cr["year"]
        if cr.get("authors") and not meta.get("authors"):
            meta["authors"] = cr["authors"]

        return meta
