"""
GUARD AGENT
The "bouncer" of the system. Before any question is answered, this agent decides
whether the question belongs to the allowed domain (scientific / academic
research). If not, the orchestrator returns a polite refusal message instead of
answering.

It replies with a single word - ALLOW or BLOCK - which we parse.
"""
from agents.base_agent import BaseAgent


class GuardAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="GuardAgent",
            system_prompt=(
                "You are a strict domain filter for a SCIENTIFIC RESEARCH assistant. "
                "Decide whether the user's message belongs to the allowed domain.\n\n"
                "ALLOWED: any question or request about research papers, scientific or "
                "academic topics, methods, experiments, results, datasets, theories, "
                "literature reviews, or finding/summarizing related research. This "
                "covers ANY field of study - computer science, physics, biology, "
                "medicine, engineering, economics, social science, and so on. It "
                "ALSO covers basic questions about an uploaded paper itself - its "
                "title, authors, abstract, references, conclusions, or which "
                "section discusses what. When in doubt, ALLOW.\n\n"
                "NOT ALLOWED: casual chit-chat, standalone greetings, general trivia "
                "unrelated to research, personal or life advice, jokes, creative "
                "writing, shopping, cooking, coding help unrelated to research, or "
                "anything off-topic.\n\n"
                "Reply with EXACTLY one word: ALLOW or BLOCK. Nothing else."
            ),
        )

    def is_allowed(self, message):
        try:
            verdict = self.run(message, temperature=0.0).upper()
        except Exception:
            # If the check itself errors out, don't break the app - allow it.
            return True
        # Block only when the model clearly says BLOCK.
        return "BLOCK" not in verdict
