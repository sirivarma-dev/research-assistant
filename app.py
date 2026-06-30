"""
THE APP (run this file).
An interactive web interface with three tabs:
  1. Analyze a Paper   - upload a PDF, get it indexed + summarized
  2. Ask Questions     - chat with the uploaded paper
  3. Find Related Papers - enter a title/keywords, get relevant arXiv papers

Run it with:   streamlit run app.py
"""
import os
import re
import urllib.request

from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from core.orchestrator import Orchestrator
from agents.analysis_agent import SECTIONS as ANALYSIS_SECTIONS
from agents.report_agent import REPORTS
from utils.pdf_export import markdown_to_pdf
from config import UPLOAD_DIR, GROQ_API_KEY, GROQ_MODEL, OUT_OF_DOMAIN_MESSAGE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load the SVG logo once so we can drop it anywhere in the page.
@st.cache_data
def load_logo():
    try:
        with open(os.path.join(BASE_DIR, "assets", "logo.svg"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


LOGO_SVG = load_logo()

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- A little CSS to make things look nicer ----------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Tighten the top padding */
        .block-container { padding-top: 2rem; }

        /* Gradient app title */
        .app-title {
            font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem;
            background: linear-gradient(90deg, #7c5cff, #4ab1ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .app-sub { color: #9aa0aa; margin-top: 0; margin-bottom: 1rem; }

        /* Card used for summaries and paper results */
        .card {
            background: #191c24; border: 1px solid #2a2e3a;
            border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 1rem;
        }
        .paper-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 .2rem 0; }
        .paper-meta { color: #9aa0aa; font-size: .85rem; margin-bottom: .5rem; }

        /* Status pill in the sidebar */
        .pill {
            display: inline-block; padding: .2rem .6rem; border-radius: 999px;
            font-size: .8rem; font-weight: 600;
        }
        .pill-on  { background: rgba(46,204,113,.15); color: #2ecc71; }
        .pill-off { background: rgba(231,76,60,.15);  color: #e74c3c; }

        /* Make buttons full-width and rounded */
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px; font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# Streamlit treats bare keys as shortcuts (C = clear cache, R = rerun) and does
# not ignore them when Ctrl/Cmd is held - so Ctrl+C to copy text also pops the
# "Clear cache?" dialog. This stops Streamlit from seeing any modified keypress,
# while leaving native browser copy/paste/reload working.
def disable_modifier_hotkeys():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        doc.addEventListener("keydown", function (e) {
            if (e.ctrlKey || e.metaKey || e.altKey) {
                // Capture phase + stopPropagation: the event never reaches
                // Streamlit's shortcut handler, but the browser's default
                // (copy/paste/reload) still happens because we don't preventDefault.
                e.stopPropagation();
            }
        }, true);
        </script>
        """,
        height=0,
    )


disable_modifier_hotkeys()


# Build the multi-agent system only once (cached across reruns)
@st.cache_resource
def get_orchestrator():
    return Orchestrator()


# Friendly check so beginners see a clear message instead of a crash
if not GROQ_API_KEY:
    st.error(
        "No Groq API key found. Create a file named '.env' in the project folder "
        "with this line:\n\nGROQ_API_KEY=your_key_here\n\n"
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

orch = get_orchestrator()


# Fetch a paper's PDF bytes so we can offer a real download button.
# Cached so repeated reruns don't re-download the same file.
@st.cache_data(show_spinner=False)
def fetch_pdf_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


# Cache AI abstract summaries so they aren't regenerated on every rerun.
@st.cache_data(show_spinner=False)
def ai_abstract_summary(abstract):
    return orch.summarize_abstract(abstract)


# Turn a paper title into a safe-ish file name for the download.
def safe_filename(title):
    name = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return (name[:80] or "paper") + ".pdf"


# ---------------- Session state defaults ----------------
# These survive across tab switches and reruns.
defaults = {
    "messages": [],        # chat history
    "paper_name": None,    # name of the currently loaded paper
    "n_chunks": 0,         # how many chunks were indexed
    "paper_stats": None,   # length info: Short/Medium/Long + word/page counts
    "summary": None,       # the generated summary (so it persists)
    "paper_meta": None,    # extracted metadata: title/authors/abstract/year/...
    "paper_analysis": None,# deep analysis: objectives/methods/results/...
    "max_results": 5,      # how many related papers to fetch
    "allow_general": True, # answer general questions, not just paper questions
    "search_results": None,# last arXiv search: {"papers": [...], "query": str}
    "comparison_result": None,  # last side-by-side comparison (markdown)
    "gap_result": None,         # last research-gap analysis (markdown)
    "graph_html": None,         # last knowledge-graph HTML
    "report_result": None,      # last generated report: {"type":.., "title":.., "md":..}
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem;">
            <div style="width:46px;">{LOGO_SVG}</div>
            <div>
                <div style="font-weight:800; font-size:1.1rem; line-height:1.1;">Research Assistant</div>
                <div style="color:#9aa0aa; font-size:.78rem;">Multi-agent · AI powered</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Live status of the loaded paper
    st.markdown("#### 📄 Current paper")
    if st.session_state.paper_name:
        st.markdown(
            f"<span class='pill pill-on'>● Loaded</span>",
            unsafe_allow_html=True,
        )
        st.write(f"**{st.session_state.paper_name}**")
        stats = st.session_state.paper_stats
        if stats:
            st.write(f"{stats['emoji']} **{stats['label']}** paper")
            st.caption(
                f"~{stats['words']:,} words · {stats['pages']} pages · "
                f"~{stats['read_min']} min read"
            )
        st.metric("Chunks indexed", st.session_state.n_chunks)
    else:
        st.markdown(
            "<span class='pill pill-off'>● None</span>",
            unsafe_allow_html=True,
        )
        st.caption("Upload a paper in the first tab.")

    st.divider()

    # Settings
    st.markdown("#### ⚙️ Settings")
    st.session_state.max_results = st.slider(
        "Related papers to fetch", min_value=3, max_value=10,
        value=st.session_state.max_results,
    )

    st.session_state.allow_general = st.toggle(
        "🌐 Answer general questions",
        value=st.session_state.allow_general,
        help="On: also answers everyday questions. Off: only answers from the "
             "uploaded paper and refuses off-topic questions.",
    )

    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat cleared", icon="🧹")

    st.divider()
    with st.expander("🤖 AI agents"):
        st.markdown(
            "This system runs a team of specialized agents:\n"
            "- **GuardAgent** — keeps it on research topics\n"
            "- **ReaderAgent** — extracts & indexes documents\n"
            "- **MetadataAgent** — title / authors / DOI\n"
            "- **AnalysisAgent** — objectives, methods, results\n"
            "- **SummarizerAgent** — structured summaries\n"
            "- **SearchAgent** — semantic arXiv retrieval\n"
            "- **ComparisonAgent** — side-by-side comparison\n"
            "- **GapAgent** — research gaps & directions\n"
            "- **CitationAgent** — citation analysis\n"
            "- **EntityAgent** — knowledge-graph entities\n"
            "- **ReportAgent** — literature reviews & reports\n\n"
            "Watch them run live in the status box when you analyze a paper."
        )

    with st.expander("ℹ️ How it works"):
        st.markdown(
            "- **Analyze**: reads your PDF, splits it into chunks, and stores "
            "them in a vector database.\n"
            "- **Ask**: finds the most relevant parts of the paper and answers "
            "from them (RAG).\n"
            "- **Find**: turns your topic into keywords and searches arXiv."
        )
    st.caption(f"Model: `{GROQ_MODEL}`")


# ---------------- MAIN HEADER ----------------
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:.1rem;">
        <div style="width:64px; flex:none;">{LOGO_SVG}</div>
        <div>
            <div class='app-title'>Autonomous Research Assistant</div>
            <p class='app-sub'>Upload a paper, ask questions about it, or discover related research.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Analyze a Paper", "💬 Ask Questions", "🔎 Find Related Papers",
     "📚 Library & Compare", "📊 Dashboard"]
)

# ---------------- TAB 1: Analyze a Paper ----------------
with tab1:
    st.subheader("Upload a research paper")
    uploaded = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt", "md", "pptx", "png", "jpg", "jpeg", "tiff", "bmp"],
        label_visibility="collapsed",
        help="PDF, Word (DOCX), text, PowerPoint (PPTX), or a scanned image. "
             "Scanned PDFs/images use OCR (needs Tesseract installed).",
    )

    if uploaded is not None:
        if st.button("✨ Analyze paper", type="primary"):
            save_path = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            try:
                # Animated, multi-step status instead of a plain spinner
                with st.status("Analyzing your paper...", expanded=True) as status:
                    st.write("🟢 **ReaderAgent** → extracting text & indexing chunks...")
                    n_chunks, text, stats = orch.ingest_paper(save_path, uploaded.name)
                    st.write(
                        f"📏 Detected a **{stats['label']}** paper "
                        f"(~{stats['words']:,} words, {stats['pages']} pages)."
                    )

                    st.write("🟢 **MetadataAgent** → title, authors, year, DOI...")
                    meta = orch.extract_metadata(text)

                    st.write("🟢 **AnalysisAgent** → objectives, methods, results, limitations...")
                    analysis = orch.analyze_paper(text)

                    st.write("🟢 **SummarizerAgent** → writing a structured summary...")
                    summary = orch.summarize_text(text)

                    status.update(
                        label=f"Done! Indexed {n_chunks} chunks.",
                        state="complete", expanded=False,
                    )

                # Save to session so it persists across tabs
                st.session_state.paper_name = uploaded.name
                st.session_state.n_chunks = n_chunks
                st.session_state.paper_stats = stats
                st.session_state.paper_meta = meta
                st.session_state.paper_analysis = analysis
                st.session_state.summary = summary
                # Save into the persistent library (history + multi-paper features)
                orch.save_to_library(uploaded.name, stats, meta, analysis, summary)
                st.toast("Paper analyzed!", icon="✅")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # Warn for long papers: whole-document answers may be trimmed.
    if st.session_state.paper_stats and st.session_state.paper_stats["label"] == "Long":
        st.warning(
            "🔴 This is a **long** paper. Whole-document answers (e.g. full "
            "summaries) use up to ~30,000 characters, so the very end may be "
            "trimmed. Focused questions and references still work fully."
        )

    # Show extracted metadata (title, authors, year, DOI, keywords, abstract)
    meta = st.session_state.paper_meta
    if meta and any(meta.values()):
        st.divider()
        st.subheader("🏷️ Metadata")
        authors = meta.get("authors") or []
        if isinstance(authors, list):
            authors = ", ".join(authors)
        keywords = meta.get("keywords") or []
        if isinstance(keywords, list):
            keywords = ", ".join(keywords)
        doi = meta.get("doi") or ""
        doi_md = f"[{doi}](https://doi.org/{doi})" if doi else "—"

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"**Title:** {meta.get('title') or '—'}")
            st.markdown(f"**Authors:** {authors or '—'}")
            st.markdown(f"**Year:** {meta.get('year') or '—'}")
        with m2:
            st.markdown(f"**DOI:** {doi_md}")
            st.markdown(f"**Keywords:** {keywords or '—'}")
        if meta.get("abstract"):
            with st.expander("📄 Abstract"):
                st.write(meta["abstract"])

    # Show the summary (persists even after switching tabs)
    if st.session_state.summary:
        st.divider()
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.subheader("📋 Summary")
            st.caption(f"For: {st.session_state.paper_name}")
        with col_b:
            st.download_button(
                "⬇️ Download",
                data=st.session_state.summary,
                file_name=f"summary_{st.session_state.paper_name}.md",
                use_container_width=True,
            )
        st.markdown(f"<div class='card'>{st.session_state.summary}</div>", unsafe_allow_html=True)

    # Show the deep analysis (objectives, methodology, results, etc.)
    analysis = st.session_state.paper_analysis
    if analysis:
        st.divider()
        st.subheader("🔬 Detailed Analysis")
        for key, label in ANALYSIS_SECTIONS.items():
            with st.expander(label):
                st.markdown(analysis.get(key, "Not reported in the paper."))

# ---------------- TAB 2: Ask Questions ----------------
with tab2:
    st.subheader("Ask anything about the uploaded paper")

    if st.session_state.paper_name:
        st.caption(f"Chatting with: **{st.session_state.paper_name}**")
    elif st.session_state.allow_general:
        st.caption("💬 General mode — ask anything, or upload a paper for paper-specific answers.")
    else:
        st.info("👈 Upload and analyze a paper first (in the **Analyze a Paper** tab).")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("e.g. What method did the authors use?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = orch.ask(question, allow_general=st.session_state.allow_general)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- TAB 3: Find Related Papers ----------------
with tab3:
    st.subheader("Find related research from a title or keywords")
    topic = st.text_input("Enter a project title or keywords", label_visibility="collapsed",
                          placeholder="e.g. multi-agent systems for literature review")

    if topic and st.button("🔎 Search arXiv", type="primary"):
        if not orch.is_in_domain(topic):
            st.session_state.search_results = None
            st.warning(OUT_OF_DOMAIN_MESSAGE)
        else:
            with st.spinner("Searching arXiv for related papers..."):
                papers, used_query = orch.find_papers(
                    topic, max_results=st.session_state.max_results
                )
            # Persist so the list survives reruns (e.g. clicking a download button).
            st.session_state.search_results = {"papers": papers, "query": used_query}

    # Render the most recent results (persisted across reruns).
    results = st.session_state.search_results
    if results:
        papers = results["papers"]
        used_query = results["query"]
        st.caption(f"🔑 Search query used: `{used_query}`  ·  {len(papers)} results")

        if not papers:
            st.warning("No papers found. Try different keywords.")

        for i, p in enumerate(papers):
            st.markdown(
                f"""
                <div class='card'>
                    <div class='paper-title'>{p['title']}</div>
                    <div class='paper-meta'>👤 {p['authors']} &nbsp;·&nbsp; 📅 {p['published']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(p["summary"][:400] + "...")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"[📄 Open PDF]({p['url']})")
            with c2:
                try:
                    pdf_bytes = fetch_pdf_bytes(p["url"])
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=safe_filename(p["title"]),
                        mime="application/pdf",
                        key=f"dl_{i}_{p['url']}",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("⚠️ Download unavailable")
            with st.expander("🤖 AI summary of this paper's abstract"):
                st.markdown(ai_abstract_summary(p["summary"]))
            st.divider()

# ---------------- TAB 4: Library & Compare ----------------
with tab4:
    st.subheader("📚 Your analyzed papers")
    library = orch.get_library()

    if not library:
        st.info("No papers yet. Analyze a paper in the **Analyze a Paper** tab and "
                "it will appear here.")
    else:
        st.caption(f"{len(library)} paper(s) in your library. "
                   "Select two or more to compare, or any to find research gaps.")

        # Multi-select papers (label -> file name)
        labels = {}
        for rec in library:
            meta = rec.get("metadata", {}) or {}
            title = meta.get("title") or rec["name"]
            year = meta.get("year") or ""
            label = f"{title}" + (f" ({year})" if year else "") + f"  ·  {rec['name']}"
            labels[label] = rec["name"]

        chosen_labels = st.multiselect(
            "Select papers", options=list(labels.keys()),
            placeholder="Choose papers to compare / analyze",
        )
        chosen_names = [labels[l] for l in chosen_labels]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            do_compare = st.button("⚖️ Compare", use_container_width=True,
                                   disabled=len(chosen_names) < 2)
        with col2:
            do_gaps = st.button("🔍 Research gaps", use_container_width=True,
                                disabled=len(chosen_names) < 1)
        with col3:
            do_graph = st.button("🕸️ Knowledge graph", use_container_width=True,
                                 disabled=len(chosen_names) < 1)
        with col4:
            if st.button("🗑️ Clear library", use_container_width=True):
                orch.clear_library()
                st.session_state.comparison_result = None
                st.session_state.gap_result = None
                st.session_state.graph_html = None
                st.rerun()

        if do_compare:
            with st.spinner("Comparing papers..."):
                records = orch.get_papers(chosen_names)
                st.session_state.comparison_result = orch.compare_papers(records)
            st.session_state.gap_result = None

        if do_gaps:
            with st.spinner("Identifying research gaps and future directions..."):
                records = orch.get_papers(chosen_names)
                st.session_state.gap_result = orch.find_research_gaps(records)
            st.session_state.comparison_result = None

        if do_graph:
            with st.spinner("Building knowledge graph..."):
                records = orch.get_papers(chosen_names)
                st.session_state.graph_html = orch.knowledge_graph(records)

        if st.session_state.comparison_result:
            st.divider()
            st.subheader("⚖️ Side-by-side comparison")
            st.markdown(st.session_state.comparison_result)

        if st.session_state.gap_result:
            st.divider()
            st.subheader("🔍 Research gaps & future directions")
            st.markdown(st.session_state.gap_result)

        if st.session_state.graph_html:
            st.divider()
            st.subheader("🕸️ Knowledge graph")
            st.caption("🟣 papers · 🔵 authors · 🟢 concepts · 🟠 methods · 🔴 datasets. "
                       "Drag nodes; shared entities link papers together.")
            components.html(st.session_state.graph_html, height=620, scrolling=True)

        # ---- Report generation + PDF export ----
        st.divider()
        st.subheader("📝 Generate a report")
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            report_label_to_type = {label: rtype for rtype, (label, _) in REPORTS.items()}
            report_label = st.selectbox(
                "Report type", options=list(report_label_to_type.keys()),
                label_visibility="collapsed",
            )
        with rc2:
            make_report = st.button("Generate", use_container_width=True,
                                    disabled=len(chosen_names) < 1)

        if make_report:
            rtype = report_label_to_type[report_label]
            with st.spinner(f"Writing {report_label}..."):
                records = orch.get_papers(chosen_names)
                md = orch.generate_report(rtype, records)
            st.session_state.report_result = {
                "type": rtype, "title": report_label, "md": md,
            }

        rep = st.session_state.report_result
        if rep:
            st.markdown(f"#### {rep['title']}")
            st.markdown(rep["md"])
            try:
                pdf_bytes = markdown_to_pdf(rep["title"], rep["md"])
                st.download_button(
                    "⬇️ Download report as PDF",
                    data=pdf_bytes,
                    file_name=safe_filename(rep["title"]).replace(".pdf", "") + "_report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.caption(f"⚠️ PDF export failed: {e}")

        # The library list itself (history), each removable.
        st.divider()
        st.markdown("#### History")
        for rec in library:
            meta = rec.get("metadata", {}) or {}
            title = meta.get("title") or rec["name"]
            with st.expander(f"📄 {title}  ·  added {rec.get('added', '')}"):
                authors = meta.get("authors") or []
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                st.markdown(f"**File:** {rec['name']}")
                st.markdown(f"**Authors:** {authors or '—'}  ·  **Year:** {meta.get('year') or '—'}")
                if rec.get("summary"):
                    st.markdown(rec["summary"])

                # Citation analysis (Semantic Scholar)
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("📈 Citation analysis", key=f"cite_{rec['name']}",
                                 use_container_width=True):
                        with st.spinner("Looking up citations on Semantic Scholar..."):
                            report = orch.analyze_citations(title, name=rec["name"])
                        st.session_state[f"cite_result_{rec['name']}"] = report
                with cc2:
                    if st.button("Remove from library", key=f"rm_{rec['name']}",
                                 use_container_width=True):
                        orch.remove_from_library(rec["name"])
                        st.rerun()

                report = st.session_state.get(f"cite_result_{rec['name']}")
                if report:
                    if not report.get("found"):
                        if report.get("error") == "rate_limited":
                            st.warning("Semantic Scholar is rate-limited right now. "
                                       "Try again in a minute, or add a free "
                                       "SEMANTIC_SCHOLAR_API_KEY to your .env.")
                        else:
                            st.info("This paper wasn't found on Semantic Scholar "
                                    "(common for non-indexed/student papers).")
                    else:
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Citations", report.get("citation_count", 0))
                        mc2.metric("Influential", report.get("influential_citation_count", 0))
                        mc3.metric("References", report.get("reference_count", 0))
                        if report.get("top_citing"):
                            st.markdown("**Notable papers citing this** (★ = influential):")
                            for c in report["top_citing"]:
                                star = "★ " if c.get("isInfluential") else ""
                                st.markdown(
                                    f"- {star}{c.get('title')} "
                                    f"({c.get('year') or 'n.d.'}, "
                                    f"{c.get('citationCount') or 0} citations)"
                                )
                        if report.get("top_references"):
                            st.markdown("**Key references it builds on:**")
                            for r in report["top_references"]:
                                st.markdown(
                                    f"- {r.get('title')} "
                                    f"({r.get('year') or 'n.d.'}, "
                                    f"{r.get('citationCount') or 0} citations)"
                                )

# ---------------- TAB 5: Dashboard ----------------
with tab5:
    st.subheader("📊 Research Dashboard")
    library = orch.get_library()

    if not library:
        st.info("Analyze some papers first — the dashboard visualizes your library.")
    else:
        # Headline metrics
        total = len(library)
        cited = [p for p in library if p.get("citation_count") is not None]
        total_citations = sum(p.get("citation_count", 0) for p in cited)
        d1, d2, d3 = st.columns(3)
        d1.metric("Papers in library", total)
        d2.metric("With citation data", len(cited))
        d3.metric("Total citations", f"{total_citations:,}")

        # Helper: gather a flat list of entity values across the library.
        def gather(entity_key):
            counter = Counter()
            for p in library:
                for v in (p.get("entities", {}) or {}).get(entity_key, []):
                    counter[v] += 1
            return counter

        # --- Publication-year trend ---
        years = [str(p.get("metadata", {}).get("year") or "").strip()
                 for p in library]
        years = [y for y in years if y.isdigit()]
        if years:
            year_counts = Counter(years)
            ordered = sorted(year_counts.items())
            fig = px.bar(
                x=[y for y, _ in ordered], y=[c for _, c in ordered],
                labels={"x": "Year", "y": "Papers"},
                title="📅 Papers by publication year",
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Citation insights (only for papers with citation data) ---
        if cited:
            names = [p.get("metadata", {}).get("title") or p["name"] for p in cited]
            fig = px.bar(
                x=names, y=[p.get("citation_count", 0) for p in cited],
                labels={"x": "Paper", "y": "Citations"},
                title="📈 Citation counts",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("💡 Run *Citation analysis* on papers (Library tab) to populate "
                       "citation insights here.")

        # --- Top methods / concepts / datasets ---
        for key, title in [("methods", "🟠 Top methods & algorithms"),
                           ("concepts", "🟢 Top concepts"),
                           ("datasets", "🔴 Top datasets")]:
            counts = gather(key)
            if counts:
                top = counts.most_common(10)
                fig = px.bar(
                    x=[c for _, c in top], y=[name for name, _ in top],
                    orientation="h", labels={"x": "Papers", "y": ""},
                    title=title,
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        if not any(gather(k) for k in ("methods", "concepts", "datasets")):
            st.caption("💡 Build a *Knowledge graph* (Library tab) to extract methods, "
                       "concepts, and datasets — they'll be charted here.")
