"""
The shared "brain" that every agent uses to talk to the Groq AI model.
Each agent gives itself a name and a system prompt (its job description),
then calls .run() to get an answer.
"""
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL


class BaseAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt
        self.client = Groq(api_key=GROQ_API_KEY)

    def run(self, user_message, temperature=0.3, system_prompt=None):
        # Allow a one-off system prompt override (defaults to this agent's own).
        system = system_prompt or self.system_prompt
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
