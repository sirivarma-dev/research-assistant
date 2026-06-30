"""
Export a markdown-ish report to a PDF (bytes), using fpdf2.

This is a lightweight renderer - it handles headings (#, ##, ###), bullet lists
(-, *), simple tables (| a | b |), and paragraphs. It is not a full markdown
engine, but produces a clean, readable PDF for the reports this app generates.

fpdf2's core fonts are latin-1 only, so we transliterate common Unicode
characters (smart quotes, dashes, arrows, emoji) to ASCII before writing.
"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Common Unicode -> ASCII replacements so core fonts don't choke.
_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "→": "->",
    "•": "-", " ": " ", "−": "-", "×": "x",
    "≥": ">=", "≤": "<=",
}


def _ascii(text):
    for uni, asc in _REPLACEMENTS.items():
        text = text.replace(uni, asc)
    # Drop anything still outside latin-1 (e.g. emoji).
    return text.encode("latin-1", "ignore").decode("latin-1")


class _PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def markdown_to_pdf(title, markdown_text):
    """Return PDF bytes for the given title + markdown body."""
    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def write(h, text, x=None):
        # Always reset the cursor to the left margin after a full-width cell,
        # otherwise the next cell starts at the right edge (fpdf2 quirk).
        if x is not None:
            pdf.set_x(x)
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(40, 40, 40)
    write(10, _ascii(title))
    pdf.ln(2)

    pdf.set_text_color(20, 20, 20)

    for raw_line in markdown_text.splitlines():
        line = _ascii(raw_line.rstrip())
        if not line.strip():
            pdf.ln(3)
            continue

        # Table row -> render cells separated by spaces (simple, readable).
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Skip markdown table separator rows (---|---).
            if all(set(c) <= set("-: ") for c in cells):
                continue
            pdf.set_font("Helvetica", "", 9)
            write(5, "  |  ".join(cells))
            continue

        # Headings
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            write(7, line[4:])
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(1)
            write(8, line[3:])
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.ln(1)
            write(9, line[2:])
        # Bullets
        elif line.lstrip().startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 10)
            text = line.lstrip()[2:]
            write(6, f"- {text}", x=pdf.l_margin + 4)
        else:
            # Strip basic emphasis markers for cleaner output.
            text = line.replace("**", "").replace("`", "")
            pdf.set_font("Helvetica", "", 10)
            write(6, text)

    # fpdf2 returns a bytearray; Streamlit's download_button wants bytes.
    return bytes(pdf.output())
