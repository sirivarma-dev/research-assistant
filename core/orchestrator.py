"""
ORCHESTRATOR
This is the "manager" of the multi-agent system. It creates each agent once,
shares a single memory (vector store) between the Reader and Q&A agents, and
exposes simple methods that the user interface (app.py) calls.
"""
from config import OUT_OF_DOMAIN_MESSAGE
from core.vector_store import VectorStore
from core.library import PaperLibrary
from core.graph import build_graph_html
from agents.search_agent import SearchAgent
from agents.reader_agent import ReaderAgent
from agents.qa_agent import QAAgent
from agents.summarizer_agent import SummarizerAgent
from agents.guard_agent import GuardAgent
from agents.metadata_agent import MetadataAgent
from agents.analysis_agent import AnalysisAgent
from agents.comparison_agent import ComparisonAgent
from agents.gap_agent import GapAgent
from agents.citation_agent import CitationAgent
from agents.entity_agent import EntityAgent
from agents.report_agent import ReportAgent


class Orchestrator:
    def __init__(self):
        self.store = VectorStore()                 # shared memory
        self.library = PaperLibrary()              # persistent history of analyzed papers
        self.search_agent = SearchAgent(self.store)  # uses store's embedder to re-rank
        self.reader_agent = ReaderAgent(self.store)
        self.qa_agent = QAAgent(self.store)
        self.summarizer_agent = SummarizerAgent()
        self.guard_agent = GuardAgent()           # domain bouncer
        self.metadata_agent = MetadataAgent()     # title/authors/DOI/etc.
        self.analysis_agent = AnalysisAgent()     # objectives/methods/results/...
        self.comparison_agent = ComparisonAgent() # side-by-side comparison
        self.gap_agent = GapAgent()               # research gaps + future directions
        self.citation_agent = CitationAgent()     # Semantic Scholar citation analysis
        self.entity_agent = EntityAgent()         # concepts/methods/datasets for the graph
        self.report_agent = ReportAgent()         # literature review / survey / etc.

    # --- Upload & analyze a paper ---
    def ingest_paper(self, pdf_path, source_name):
        return self.reader_agent.read_pdf(pdf_path, source_name)

    def summarize_text(self, text):
        return self.summarizer_agent.summarize(text)

    def extract_metadata(self, text):
        return self.metadata_agent.extract(text)

    def analyze_paper(self, text):
        return self.analysis_agent.analyze(text)

    # --- Paper library / history ---
    def save_to_library(self, name, stats, metadata, analysis, summary):
        return self.library.add(name, stats, metadata, analysis, summary)

    def get_library(self):
        return self.library.all()

    def get_papers(self, names):
        return self.library.get_many(names)

    def remove_from_library(self, name):
        self.library.remove(name)

    def clear_library(self):
        self.library.clear()

    # --- Multi-paper analysis ---
    def compare_papers(self, records):
        return self.comparison_agent.compare(records)

    def find_research_gaps(self, records):
        return self.gap_agent.find_gaps(records)

    def analyze_citations(self, title, name=None):
        report = self.citation_agent.analyze(title)
        # Cache counts on the library record so the dashboard can use them.
        if name and report.get("found"):
            self.library.set_citations(
                name, report.get("citation_count", 0),
                report.get("influential_citation_count", 0),
            )
        return report

    def generate_report(self, report_type, records):
        return self.report_agent.generate(report_type, records)

    def knowledge_graph(self, records):
        """Build an interactive knowledge-graph HTML for the given papers."""
        papers = []
        for rec in records:
            meta = rec.get("metadata", {}) or {}
            # Extract entities once and cache them on the library record.
            entities = rec.get("entities")
            if not entities:
                entities = self.entity_agent.extract(rec)
                self.library.set_entities(rec["name"], entities)
            papers.append({
                "title": meta.get("title") or rec["name"],
                "authors": meta.get("authors") or [],
                "entities": entities,
            })
        return build_graph_html(papers)

    # --- Ask questions about the uploaded paper ---
    def ask(self, question, allow_general=True):
        # When general questions are allowed, skip the research-only guard.
        if not allow_general and not self.guard_agent.is_allowed(question):
            return OUT_OF_DOMAIN_MESSAGE
        return self.qa_agent.answer(question, allow_general=allow_general)

    # Reusable check the UI can call (e.g. for the search box).
    def is_in_domain(self, text):
        return self.guard_agent.is_allowed(text)

    # --- Find related papers from a title/keywords ---
    def find_papers(self, topic, max_results=5):
        return self.search_agent.search_papers(topic, max_results)

    def summarize_abstract(self, abstract):
        return self.summarizer_agent.summarize(abstract)
