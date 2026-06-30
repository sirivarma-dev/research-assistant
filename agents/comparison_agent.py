"""
COMPARISON AGENT
Compares two or more papers side by side using the structured analysis we
already extracted for each (objectives, methodology, datasets, evaluation,
limitations, future work). Produces a markdown comparison table plus a short
narrative of the key similarities and differences.
"""
from agents.base_agent import BaseAgent


def _paper_brief(record):
    """Condense a library record into a compact text block for the prompt."""
    meta = record.get("metadata", {}) or {}
    analysis = record.get("analysis", {}) or {}
    authors = meta.get("authors") or []
    if isinstance(authors, list):
        authors = ", ".join(authors)
    title = meta.get("title") or record.get("name", "Untitled")
    lines = [f"TITLE: {title}", f"AUTHORS: {authors}", f"YEAR: {meta.get('year', '')}"]
    for key in ("objectives", "methodology", "datasets_tools",
                "evaluation", "limitations", "future_work"):
        lines.append(f"{key.upper()}: {analysis.get(key, 'N/A')}")
    return "\n".join(lines)


class ComparisonAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ComparisonAgent",
            system_prompt=(
                "You are a research analyst comparing scientific papers. Using ONLY "
                "the provided details for each paper, produce:\n"
                "1. A markdown comparison TABLE with one column per paper and rows "
                "for: Objective, Methodology, Datasets/Tools, Evaluation/Results, "
                "Limitations, Future Work. Keep cells short.\n"
                "2. A brief '### Key Similarities' section (bullet points).\n"
                "3. A brief '### Key Differences' section (bullet points).\n"
                "Be factual and grounded in the provided text. Do not invent details."
            ),
        )

    def compare(self, records):
        if len(records) < 2:
            return "Select at least two papers to compare."
        blocks = []
        for i, rec in enumerate(records, 1):
            blocks.append(f"=== PAPER {i} ===\n{_paper_brief(rec)}")
        prompt = "Compare these papers:\n\n" + "\n\n".join(blocks)
        return self.run(prompt, temperature=0.3)
