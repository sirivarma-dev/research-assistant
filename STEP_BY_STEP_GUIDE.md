# Autonomous Multi-Agent Research Assistant — Setup Guide (Windows + VS Code + Groq)

This guide takes you from an empty machine to a working app. Follow it top to bottom.
Total time: about 15–20 minutes (most of it is downloads).

---

## What you are building

A web app with **four cooperating AI agents** managed by an **Orchestrator**:

| Agent | Job |
|---|---|
| **Reader Agent** | Reads an uploaded PDF, splits it into chunks, stores them in memory |
| **Q&A Agent** | Answers your questions using only the paper's content (this is "RAG") |
| **Summarizer Agent** | Writes a structured summary of a paper |
| **Search Agent** | Takes your title/keywords and finds related papers on arXiv |

It does exactly the three things you asked for:
1. Upload a paper → it studies it and summarizes it.
2. Ask questions → it answers from that paper.
3. Give a title/keywords → it finds related papers and summarizes them.

---

## The folder structure (so you know where everything lives)

```
research-assistant/
├── app.py                  <- the main file you RUN
├── config.py               <- your API key + which AI model to use
├── requirements.txt        <- list of libraries to install
├── .env                    <- YOU create this (holds your secret key)
├── .env.example            <- a template for the .env file
├── agents/
│   ├── base_agent.py       <- shared code to talk to the Groq AI
│   ├── reader_agent.py     <- reads the PDF
│   ├── qa_agent.py         <- answers questions
│   ├── summarizer_agent.py <- summarizes
│   └── search_agent.py     <- searches arXiv
├── core/
│   ├── orchestrator.py     <- the manager that connects all agents
│   └── vector_store.py     <- the assistant's memory (ChromaDB)
├── utils/
│   └── pdf_utils.py        <- pulls text out of PDFs
└── data/
    ├── uploads/            <- your uploaded PDFs get saved here
    └── chroma_db/          <- the memory database lives here
```

You do NOT need to type any of this code yourself — every file already contains
the finished code. You only need to (a) install the tools, (b) add your key,
(c) run one command.

---

## STEP 1 — Install Python (if you don't have it)

1. Go to https://www.python.org/downloads/
2. Download **Python 3.12** (or 3.11). *Avoid the very newest 3.13+ for now —
   some libraries don't have ready installers for it yet.*
3. Run the installer. **IMPORTANT: tick the box "Add Python to PATH"** at the
   bottom of the first screen, then click Install.
4. To confirm, open the Start menu, type `cmd`, press Enter, and run:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`.

## STEP 2 — Install VS Code

1. Get it from https://code.visualstudio.com/ and install.
2. Open VS Code → click the **Extensions** icon on the left (four squares) →
   search **"Python"** (by Microsoft) → click **Install**.

## STEP 3 — Open the project in VS Code

1. Put the `research-assistant` folder somewhere easy, e.g. your Desktop.
2. In VS Code: **File → Open Folder…** → select the `research-assistant` folder.
3. You should now see all the files listed on the left.

## STEP 4 — Open the terminal inside VS Code

- Top menu: **Terminal → New Terminal**.
- A panel opens at the bottom. This is where you type commands.
- Make sure it says it's inside your `research-assistant` folder.

## STEP 5 — Create and activate a virtual environment

A "virtual environment" keeps this project's libraries separate from the rest of
your computer. In the terminal, run these two commands **one at a time**:

```
python -m venv venv
```
```
venv\Scripts\activate
```

After the second command you should see `(venv)` at the start of the line.

> **If you get a red error** like *"running scripts is disabled on this system"*,
> PowerShell is blocking the script. Easiest fix: click the small dropdown arrow
> (˅) next to the `+` in the terminal panel, choose **Command Prompt**, then run
> `venv\Scripts\activate` again. (Command Prompt doesn't have that restriction.)

## STEP 6 — Install the libraries

With `(venv)` showing, run:

```
pip install -r requirements.txt
```

This installs Streamlit, the Groq client, the arXiv search library, the PDF
reader, and ChromaDB. It takes a few minutes. Wait until it finishes.

## STEP 7 — Get your FREE Groq API key

1. Go to https://console.groq.com/keys and sign up (no credit card needed).
2. Click **Create API Key**, give it any name, and **copy** the key
   (it starts with `gsk_...`). You only see it once, so copy it now.

## STEP 8 — Add your key to the project

1. In VS Code's file list, right-click → **New File** → name it exactly `.env`
   (yes, starting with a dot, no name before it).
2. Put this one line inside it, pasting your real key:
   ```
   GROQ_API_KEY=gsk_your_real_key_here
   ```
3. Save the file (Ctrl+S).

*(There's a `.env.example` file showing the format if you get stuck.)*

## STEP 9 — Run the app!

In the terminal (still showing `(venv)`), run:

```
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.
The **first time only**, it downloads a small embedding model (needs internet,
takes ~1 minute). After that it's fast.

To stop the app later: click in the terminal and press **Ctrl+C**.
To start it again next time: activate the venv (Step 5, second command) and run
`streamlit run app.py` again.

---

## How to use it

**Tab 1 — Analyze a Paper:** Click "Choose a PDF", pick a research paper, click
**Analyze paper**. You get a summary, and the paper is now in memory.

**Tab 2 — Ask Questions:** Type questions like *"What dataset did they use?"* or
*"What were the main results?"* It answers from the paper you uploaded.

**Tab 3 — Find Related Papers:** Type your project title or keywords (e.g.
*"multi-agent reinforcement learning for traffic control"*) and click **Search
arXiv**. You get a list of related papers, each with a link and an AI summary.

---

## Troubleshooting

- **"No Groq API key found"** → Your `.env` file is missing, misnamed, or the key
  line is wrong. It must be named `.env` and contain `GROQ_API_KEY=gsk_...`.

- **"model has been decommissioned" / model error** → Groq retires models often.
  Open `config.py`, find the `GROQ_MODEL = ...` line, and change it to a current
  model from https://console.groq.com/docs/models (e.g.
  `meta-llama/llama-4-scout-17b-16e-instruct` or `openai/gpt-oss-120b`). Save and
  restart the app.

- **"No readable text found in this PDF"** → The PDF is a scanned image, not real
  text. This version can't read scanned pages (that needs OCR). Try a normal,
  text-based PDF.

- **`pip install` fails on ChromaDB** → Make sure you're on Python 3.11 or 3.12,
  not 3.13+. Re-create the venv with the right Python version if needed.

- **Rate limit / "too many requests"** → Groq's free tier allows ~30 requests per
  minute. Just wait a minute and try again.

- **Want to clear the memory and start fresh?** → Stop the app, delete everything
  inside the `data/chroma_db` folder, and run the app again.
