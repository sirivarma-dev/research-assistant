"""
GAP AGENT
Looks across one or more analyzed papers and identifies:
  - research gaps and unexplored areas,
  - recommended future research directions.

It reuses the structured analysis (limitations + future work are especially
useful here) plus objectives and methodology.
"""
from agents.base_agent import BaseAgent
from agents.comparison_agent import _paper_brief


class GapAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="GapAgent",
            system_prompt=(
                "You are a senior researcher identifying where a field can go next. "
                "Using ONLY the provided paper details, identify research gaps WITH "
                "EVIDENCE. For EACH gap, output this markdown structure:\n\n"
                "### Gap N: <short title>\n"
                "- **What's missing:** the unexplored area / unanswered question / "
                "unaddressed weakness.\n"
                "- **Supporting papers:** name the specific paper(s) by TITLE that "
                "show evidence of this gap, and quote or paraphrase the exact "
                "limitation or future-work statement that points to it.\n"
                "- **Why it's a gap:** explain the reasoning — why this is genuinely "
                "open / unsolved given what the papers report.\n\n"
                "After all gaps, add:\n"
                "### Recommended Future Directions\n"
                "- concrete, actionable directions that would address the gaps above.\n\n"
                "Ground EVERY point in the provided text (especially the stated "
                "limitations and future work). Never invent papers or findings; if "
                "you cannot tie a gap to evidence in the papers, do not list it."
            ),
        )

    def find_gaps(self, records):
        if not records:
            return "Select at least one paper to analyze for research gaps."
        blocks = []
        for i, rec in enumerate(records, 1):
            blocks.append(f"=== PAPER {i} ===\n{_paper_brief(rec)}")
        prompt = (
            "Identify research gaps and future directions across these papers:\n\n"
            + "\n\n".join(blocks)
        )
        return self.run(prompt, temperature=0.4)
