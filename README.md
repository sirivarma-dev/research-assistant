# Autonomous Research Assistant

A multi-agent research assistant built with Streamlit. Upload papers, get
structured analysis, search related research semantically, compare papers, build
knowledge graphs, and generate reports.

## Features
- Upload **PDF, DOCX, TXT, PPTX, and scanned images** (OCR)
- Extract **metadata** (title, authors, abstract, year, keywords, DOI)
- **Deep analysis**: objectives, methodology, datasets/tools, evaluation,
  limitations, future work
- **Semantic search** of arXiv (understands meaning, not just keywords)
- **RAG Q&A** grounded in the uploaded paper
- **Compare** papers, detect **research gaps**, build an interactive
  **knowledge graph**, and run **citation analysis**
- Generate **literature review / survey / gap / methodology** reports and
  **export to PDF**
- **Dashboard** with research trends and citation insights

## Run locally
```bash
python -m venv venv
venv/Scripts/activate          # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```
Create a `.env` file (copy `.env.example`) and add your free Groq key:
```
GROQ_API_KEY=your_key_here
```
Then:
```bash
streamlit run app.py
```

OCR (for scanned documents) also needs the free **Tesseract** program:
https://github.com/UB-Mannheim/tesseract/wiki

## Models / services (all free)
- **Groq – Llama 3.3 70B** for the LLM agents
- **ChromaDB** local embeddings for semantic search
- **arXiv**, **Semantic Scholar**, **CrossRef** APIs
- **Tesseract** for OCR

## Deploy
Designed for **Streamlit Community Cloud**. Add `GROQ_API_KEY` in the app's
**Secrets**; `packages.txt` installs Tesseract for OCR.
