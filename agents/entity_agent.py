"""
ENTITY AGENT
Extracts the graph-worthy entities from a paper so we can build a knowledge
graph: the concepts, methods/algorithms, and datasets it involves.

Authors come straight from the metadata, so this agent focuses on the three
things that aren't already structured: concepts, methods, datasets.
"""
import json

from agents.base_agent import BaseAgent


class EntityAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="EntityAgent",
            system_prompt=(
                "You extract key entities from a research paper for a knowledge "
                "graph. Return ONLY a JSON object with EXACTLY these keys, each a "
                "list of SHORT canonical names (max 6 items each, no duplicates):\n"
                '  "concepts": main topics/ideas,\n'
                '  "methods": methods, algorithms, models, or techniques,\n'
                '  "datasets": datasets or data sources used.\n'
                "Use [] if none. Keep names concise (e.g. 'CNN', 'ImageNet', "
                "'transfer learning'). Output JSON only."
            ),
        )

    @staticmethod
    def _parse_json(raw):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        return json.loads(raw)

    def extract(self, record):
        """Return {'concepts': [...], 'methods': [...], 'datasets': [...]}."""
        result = {"concepts": [], "methods": [], "datasets": []}
        meta = record.get("metadata", {}) or {}
        analysis = record.get("analysis", {}) or {}

        # Seed concepts with any keywords we already have.
        kws = meta.get("keywords") or []
        if isinstance(kws, list):
            result["concepts"] = list(kws)[:6]

        # Build a compact description for the LLM.
        text = (
            f"Title: {meta.get('title', record.get('name', ''))}\n"
            f"Keywords: {', '.join(result['concepts'])}\n"
            f"Objectives: {analysis.get('objectives', '')}\n"
            f"Methodology: {analysis.get('methodology', '')}\n"
            f"Datasets/Tools: {analysis.get('datasets_tools', '')}\n"
        )
        try:
            parsed = self._parse_json(self.run(text, temperature=0.0))
            for key in result:
                vals = parsed.get(key)
                if isinstance(vals, list) and vals:
                    # Merge + de-dupe, keep order, cap at 6.
                    merged = result[key] + [str(v) for v in vals]
                    seen, out = set(), []
                    for v in merged:
                        k = v.strip().lower()
                        if k and k not in seen:
                            seen.add(k)
                            out.append(v.strip())
                    result[key] = out[:6]
        except Exception:
            pass
        return result
