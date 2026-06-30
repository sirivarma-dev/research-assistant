"""
SUMMARIZER AGENT
Produces a clean, structured summary of a paper's text.
"""
from agents.base_agent import BaseAgent


class SummarizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SummarizerAgent",
            system_prompt=(
                "You are an expert at summarizing scientific papers for students. "
                "Produce a clear summary using these headings:\n"
                "**Problem** - what the paper tackles\n"
                "**Method** - the approach or technique used\n"
                "**Key Findings** - the main results\n"
                "**Limitations** - weaknesses or open questions\n"
                "Keep each section short and easy to understand."
            ),
        )

    def summarize(self, text):
        # Keep within the model's input limit by trimming very long papers
        text = text[:12000]
        return self.run(f"Summarize this research paper:\n\n{text}")
