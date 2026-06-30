"""
Central configuration for the Research Assistant.
Loads your Groq API key from the .env file and defines the model + folder paths.
"""
import os
from dotenv import load_dotenv

# Load variables from the .env file in this folder
load_dotenv()

# ---- Groq settings ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Semantic Scholar (optional) ----
# Citation analysis works without a key, but the free keyless API is heavily
# rate-limited. Get a free key at https://www.semanticscholar.org/product/api
# and add SEMANTIC_SCHOLAR_API_KEY=... to your .env for higher limits.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# The AI model that powers every agent.
# This is the model name as Groq expects it.
#
# If you ever get an error like "model has been decommissioned", just swap the
# line below for one of the current models from https://console.groq.com/docs/models
# Good free choices (as of 2026):
#   "llama-3.3-70b-versatile"                       <-- default, strong & general
#   "meta-llama/llama-4-scout-17b-16e-instruct"     <-- newer Llama 4
#   "openai/gpt-oss-120b"                            <-- powerful reasoning
#   "llama-3.1-8b-instant"                           <-- fastest / lightest
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---- Domain restriction ----
# The message shown when a user asks something outside the research domain.
# Edit this wording to whatever you like.
OUT_OF_DOMAIN_MESSAGE = (
    "⚠️ I'm a research assistant and can only help with scientific papers and "
    "research topics. I can't answer that. Please ask a question about the "
    "uploaded paper, or give me a research topic or keywords to explore."
)

# ---- Folder paths (created automatically) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
