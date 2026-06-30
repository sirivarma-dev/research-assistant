"""
PAPER LIBRARY
A small persistent store of every paper that has been analyzed, so the user can:
  - keep a history of past analyses (survives app restarts),
  - select several papers to compare or find gaps across,
  - revisit a paper's metadata / analysis / summary later.

Each record is a plain dict saved to data/library.json:
  {
    "name": file name (unique key),
    "added": ISO date string,
    "stats": length stats dict,
    "metadata": {title, authors, abstract, year, keywords, doi},
    "analysis": {objectives, methodology, ...},
    "summary": markdown string,
  }
"""
import json
import os
from datetime import datetime

from config import BASE_DIR

LIBRARY_PATH = os.path.join(BASE_DIR, "data", "library.json")


class PaperLibrary:
    def __init__(self, path=LIBRARY_PATH):
        self.path = path
        self.papers = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.papers, f, indent=2, ensure_ascii=False)

    def add(self, name, stats, metadata, analysis, summary):
        """Add (or replace, by name) an analyzed paper."""
        record = {
            "name": name,
            "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stats": stats,
            "metadata": metadata,
            "analysis": analysis,
            "summary": summary,
        }
        # De-duplicate by name: a re-analyzed paper replaces the old record.
        self.papers = [p for p in self.papers if p.get("name") != name]
        self.papers.append(record)
        self._save()
        return record

    def all(self):
        """All records, newest first."""
        return list(reversed(self.papers))

    def get_many(self, names):
        """Records whose name is in `names`, in library order."""
        wanted = set(names)
        return [p for p in self.papers if p.get("name") in wanted]

    def set_entities(self, name, entities):
        """Cache the extracted graph entities on a record so we don't redo it."""
        for p in self.papers:
            if p.get("name") == name:
                p["entities"] = entities
                self._save()
                return

    def set_citations(self, name, citation_count, influential_count):
        """Cache citation counts on a record (used by the dashboard)."""
        for p in self.papers:
            if p.get("name") == name:
                p["citation_count"] = citation_count
                p["influential_count"] = influential_count
                self._save()
                return

    def remove(self, name):
        self.papers = [p for p in self.papers if p.get("name") != name]
        self._save()

    def clear(self):
        self.papers = []
        self._save()

    def count(self):
        return len(self.papers)
