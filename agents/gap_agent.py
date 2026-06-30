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
                "Using ONLY the provided paper details, produce markdown with:\n"
                "### Research Gaps\n"
                "- unexplored areas, unanswered questions, and weaknesses shared or "
                "left open by the paper(s).\n"
                "### Recommended Future Directions\n"
                "- concrete, actionable research directions that would address those "
                "gaps.\n"
                "Ground every point in the provided text. Pay special attention to "
                "the stated limitations and future work. Do not invent findings."
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
