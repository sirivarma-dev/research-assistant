"""
KNOWLEDGE GRAPH BUILDER
Turns a set of analyzed papers (plus their extracted entities) into an
interactive graph that connects papers to their authors, concepts, methods, and
datasets. Shared entities (e.g. two papers using the same dataset) naturally
link the papers together.

Returns a self-contained HTML string that the app embeds with
streamlit.components.v1.html.
"""
import networkx as nx
from pyvis.network import Network

# Colors per node type (match the app's purple/blue theme).
NODE_STYLE = {
    "paper":   {"color": "#7c5cff", "size": 26},
    "author":  {"color": "#4ab1ff", "size": 16},
    "concept": {"color": "#2ecc71", "size": 16},
    "method":  {"color": "#f39c12", "size": 16},
    "dataset": {"color": "#e74c3c", "size": 16},
}


def _add_node(graph, node_id, label, ntype):
    if node_id not in graph:
        style = NODE_STYLE[ntype]
        graph.add_node(
            node_id, label=label, color=style["color"], size=style["size"],
            title=f"{ntype.title()}: {label}", group=ntype,
        )


def build_graph_html(papers):
    """
    `papers` is a list of dicts, each like:
      {"title": str, "authors": [..], "entities": {"concepts":[],"methods":[],"datasets":[]}}
    """
    g = nx.Graph()

    for p in papers:
        title = p.get("title") or "Untitled"
        paper_id = f"paper::{title}"
        _add_node(g, paper_id, title, "paper")

        for author in (p.get("authors") or [])[:4]:
            if author:
                nid = f"author::{author.lower()}"
                _add_node(g, nid, author, "author")
                g.add_edge(paper_id, nid)

        entities = p.get("entities", {}) or {}
        for ntype in ("concept", "method", "dataset"):
            for name in entities.get(ntype + "s", []):
                if name:
                    nid = f"{ntype}::{name.lower()}"
                    _add_node(g, nid, name, ntype)
                    g.add_edge(paper_id, nid)

    net = Network(
        height="600px", width="100%",
        bgcolor="#0e1117", font_color="#e6e6e6",
        notebook=False, cdn_resources="in_line",
    )
    net.from_nx(g)
    net.barnes_hut(gravity=-8000, spring_length=120)

    # generate_html() returns the HTML string directly, avoiding pyvis's
    # save_graph() which writes with the OS default encoding (breaks on Windows).
    return net.generate_html(notebook=False)
