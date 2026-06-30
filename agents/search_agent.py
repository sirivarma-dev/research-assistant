"""
SEARCH AGENT
Given a project title or some keywords, it (1) asks the AI to turn your topic
into a clean search query, then (2) searches arXiv (a free, huge database of
scientific papers) and returns the most relevant ones.
"""
import arxiv

from agents.base_agent import BaseAgent


class SearchAgent(BaseAgent):
    def __init__(self, store):
        # The shared VectorStore gives us the local embedding model used to
        # re-rank arXiv results by meaning (see search_papers below).
        self.store = store
        super().__init__(
            name="SearchAgent",
            system_prompt=(
                "You convert a user's research topic into a short arXiv search "
                "query of 3 to 6 important keywords. Reply with ONLY the query, "
                "no quotes, no explanation."
            ),
        )
        self.client_arxiv = arxiv.Client()

    def refine_query(self, topic):
        try:
            return self.run(f"Topic: {topic}")
        except Exception:
            # If the AI call fails for any reason, just search the raw topic
            return topic

    def search_papers(self, topic, max_results=5):
        query = self.refine_query(topic)

        # Pull a LARGER candidate pool than we need. arXiv only matches keywords,
        # so its own ordering is unreliable - we just want raw candidates here and
        # will re-rank them by meaning afterwards.
        pool_size = max(max_results * 6, 30)
        search = arxiv.Search(
            query=query,
            max_results=pool_size,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        candidates = []
        for result in self.client_arxiv.results(search):
            summary = (result.summary or "").replace("\n", " ")
            # Skip withdrawn papers - they show up as useless "this paper has
            # been withdrawn" entries.
            if "withdrawn" in summary.lower():
                continue
            candidates.append(
                {
                    "title": result.title,
                    "authors": ", ".join(a.name for a in result.authors[:3]),
                    "summary": summary,
                    "url": result.pdf_url,
                    "published": result.published.strftime("%Y-%m-%d"),
                }
            )

        if not candidates:
            return [], query

        # Re-rank by MEANING using the user's full original sentence (not the
        # stripped-down keyword query), so results match intent, not just words.
        try:
            documents = [f"{c['title']}. {c['summary']}" for c in candidates]
            ranked = self.store.rerank(topic, documents, top_k=max_results)
            papers = [candidates[i] for i, _score in ranked]
        except Exception:
            # If embedding/re-ranking fails, fall back to arXiv's own order.
            papers = candidates[:max_results]

        return papers, query
