"""
CITATION AGENT
Looks a paper up on Semantic Scholar (a free scholarly graph API) to surface:
  - how many times it has been cited (and "influential" citations),
  - the most influential papers that cite it,
  - the key papers it references.

This is a data-only agent (no LLM): it queries the API and shapes the result.
No API key is required for basic use.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

S2_BASE = "https://api.semanticscholar.org/graph/v1"
_HEADERS = {"User-Agent": "ResearchAssistant/1.0 (mailto:apps@automatr.tech)"}


class RateLimited(Exception):
    """Raised when Semantic Scholar returns HTTP 429 after retries."""


def _get(url, tries=3):
    headers = dict(_HEADERS)
    # Optional free API key for higher rate limits (set SEMANTIC_SCHOLAR_API_KEY).
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key

    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt < tries - 1:
                    time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s, ...
                    continue
                raise RateLimited()
            raise


class CitationAgent:
    def analyze(self, title, top_n=8):
        """Return a citation report dict for the paper with this title."""
        empty = {"found": False, "title": title}
        if not title:
            return empty

        try:
            # 1. Find the paper's Semantic Scholar id.
            params = urllib.parse.urlencode({
                "query": title,
                "limit": 1,
                "fields": "paperId,title,year,citationCount,"
                          "influentialCitationCount,referenceCount,url",
            })
            search = _get(f"{S2_BASE}/paper/search?{params}")
            items = search.get("data", [])
            if not items:
                return empty
            paper = items[0]
            pid = paper["paperId"]

            # 2. Pull citing papers. The API returns an arbitrary page (not the
            # globally most-cited), so we rank by the per-citation 'isInfluential'
            # flag first, then by the citing paper's own citation count.
            cparams = urllib.parse.urlencode({
                "fields": "title,year,citationCount,isInfluential", "limit": 100,
            })
            citations = _get(f"{S2_BASE}/paper/{pid}/citations?{cparams}")
            citing = []
            for c in citations.get("data", []):
                paper_obj = c.get("citingPaper", {})
                if paper_obj.get("title"):
                    paper_obj["isInfluential"] = c.get("isInfluential", False)
                    citing.append(paper_obj)
            citing.sort(
                key=lambda c: (c.get("isInfluential", False), c.get("citationCount") or 0),
                reverse=True,
            )

            # 3. Pull the key references (most-cited ones it builds on).
            references = _get(f"{S2_BASE}/paper/{pid}/references?{cparams}")
            refs = [r.get("citedPaper", {}) for r in references.get("data", [])]
            refs = [r for r in refs if r.get("title")]
            refs.sort(key=lambda r: r.get("citationCount") or 0, reverse=True)

            return {
                "found": True,
                "title": paper.get("title", title),
                "year": paper.get("year"),
                "url": paper.get("url"),
                "citation_count": paper.get("citationCount", 0),
                "influential_citation_count": paper.get("influentialCitationCount", 0),
                "reference_count": paper.get("referenceCount", 0),
                "top_citing": citing[:top_n],
                "top_references": refs[:top_n],
            }
        except RateLimited:
            return {"found": False, "title": title, "error": "rate_limited"}
        except Exception:
            return empty
