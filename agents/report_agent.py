"""
REPORT AGENT
Generates long-form research reports from a set of analyzed papers:
  - literature_review      : a narrative literature review
  - survey                 : a structured survey report of the area
  - gap_analysis           : a research-gap analysis report
  - methodology_comparison : a report comparing the papers' methodologies

Each report is grounded in the structured analysis we already extracted per
paper, and returned as markdown.
"""
from agents.base_agent import BaseAgent
from agents.comparison_agent import _paper_brief

# report_type -> (UI label, instruction for the model)
REPORTS = {
    "literature_review": (
        "📚 Literature Review",
        "Write a cohesive LITERATURE REVIEW. Synthesize the papers into flowing "
        "prose (not just a list): introduce the area, group related work by theme, "
        "discuss how the papers relate, and end with a short synthesis. Use "
        "markdown headings.",
    ),
    "survey": (
        "🗂️ Survey Report",
        "Write a structured SURVEY REPORT of this research area with these "
        "sections: ## Introduction, ## Taxonomy/Themes, ## Methods Overview, "
        "## Datasets & Evaluation, ## Open Challenges, ## Conclusion.",
    ),
    "gap_analysis": (
        "🔍 Research Gap Analysis",
        "Write a RESEARCH GAP ANALYSIS report with sections: ## Current State, "
        "## Identified Gaps, ## Unexplored Areas, ## Recommended Future Work. "
        "Ground gaps in the papers' stated limitations and future work.",
    ),
    "methodology_comparison": (
        "🧪 Methodology Comparison",
        "Write a METHODOLOGY COMPARISON report. Include a markdown table comparing "
        "the methods/algorithms, datasets, and evaluation of each paper, then "
        "sections: ## Strengths & Weaknesses, ## When to Use Which, ## Conclusion.",
    ),
}


class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ReportAgent",
            system_prompt=(
                "You are an expert scientific writer. Produce a well-structured, "
                "factual research report in markdown, grounded ONLY in the provided "
                "paper details. Do not invent papers, results, or citations. Refer "
                "to papers by their titles."
            ),
        )

    def generate(self, report_type, records):
        if report_type not in REPORTS:
            raise ValueError(f"Unknown report type: {report_type}")
        if not records:
            return "Select at least one paper to generate a report."

        _, instruction = REPORTS[report_type]
        blocks = []
        for i, rec in enumerate(records, 1):
            blocks.append(f"=== PAPER {i} ===\n{_paper_brief(rec)}")
        prompt = (
            f"{instruction}\n\nBase the report on these papers:\n\n"
            + "\n\n".join(blocks)
        )
        return self.run(prompt, temperature=0.4)
