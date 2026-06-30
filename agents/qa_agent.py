"""
Q&A AGENT
Answers your questions. When the question is about the uploaded paper it pulls
the most relevant chunks from the vector store and answers from them (RAG). When
"general questions" are allowed, it can also answer everyday questions from its
own knowledge instead of refusing them.
"""
from agents.base_agent import BaseAgent


class QAAgent(BaseAgent):
    def __init__(self, store):
        super().__init__(
            name="QAAgent",
            # Strict mode: answer only from the paper (used when general Q&A is off).
            system_prompt=(
                "You are a careful research assistant. Answer the question using "
                "ONLY the provided context from a research paper. If the answer is "
                "not in the context, say: 'I couldn't find that in the paper.' "
                "Be clear and concise."
            ),
        )
        self.store = store

    # Used when general questions are allowed but no paper is loaded.
    GENERAL_SYSTEM = (
        "You are a helpful, knowledgeable assistant. Answer the user's question "
        "clearly and accurately using your own knowledge. Be concise."
    )

    # Used when a paper IS loaded and general questions are allowed: prefer the
    # paper, but fall back to general knowledge when the paper doesn't cover it.
    FLEX_SYSTEM = (
        "You are a helpful assistant for research. You are given context from an "
        "uploaded paper. If that context answers the question, answer from it and "
        "stay grounded in the paper. If the question is general or not covered by "
        "the paper, answer it from your own knowledge and briefly note that this "
        "is general information, not from the uploaded paper. Be clear and concise."
    )

    # Words that signal a "whole paper" question. These need the full text,
    # because the answer (references, author list, etc.) is spread across the
    # document and won't fit in just a few retrieved chunks.
    GLOBAL_HINTS = (
        "reference", "bibliography", "citation", "cited",
        "author", "who wrote", "title of", "project title", "paper title",
        "summarize", "summary", "overview", "entire", "whole",
        "list all", "all the", "abstract", "conclusion",
    )

    # Words that specifically signal a request for the reference / citation list.
    REFERENCE_HINTS = ("reference", "bibliography", "citation", "cited", "works cited")

    def _needs_full_paper(self, question):
        q = question.lower()
        return any(hint in q for hint in self.GLOBAL_HINTS)

    def _needs_references(self, question):
        q = question.lower()
        return any(hint in q for hint in self.REFERENCE_HINTS)

    def answer(self, question, allow_general=True):
        has_paper = self.store.count() > 0

        # No paper loaded.
        if not has_paper:
            if allow_general:
                # Answer as a normal general-purpose assistant.
                return self.run(question, system_prompt=self.GENERAL_SYSTEM)
            return "No paper has been uploaded yet. Please upload a paper first."

        # A paper is loaded: gather the most useful context from it.
        if self._needs_references(question):
            # Reference question: jump straight to the reference section.
            context = self.store.get_references_section()
        elif self._needs_full_paper(question):
            # Whole-document question: give the model the entire paper.
            context = self.store.get_full_text()
        else:
            # Focused question: retrieve only the most relevant chunks (RAG).
            context = "\n\n---\n\n".join(self.store.search(question, n_results=6))

        if allow_general:
            # Prefer the paper, but allow general-knowledge answers as a fallback.
            prompt = (
                f"Context from the uploaded paper:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            return self.run(prompt, system_prompt=self.FLEX_SYSTEM)

        # Strict mode: answer only from the paper.
        prompt = (
            f"Context from the paper:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return self.run(prompt)
