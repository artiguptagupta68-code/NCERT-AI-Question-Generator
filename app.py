# ===============================
# NCERT + UPSC Exam-Ready Generator + Chatbot (RAG)
# ===============================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# -------------------------------
# LOAD EMBEDDING MODEL (ONCE)
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# -------------------------------
# UTILITIES
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            extract_to = zfile.parent / zfile.stem
            extract_to.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(extract_to)
        except:
            pass


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""


def clean_text(text):
    text = re.sub(r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*",
                  " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 80:
            texts.append(t)
    return texts


def semantic_chunks(text):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]


# -------------------------------
# LOAD + CHUNK + EMBED (CACHED)
# -------------------------------
@st.cache_data(show_spinner=True)
def prepare_corpus():
    texts = load_all_texts()
    chunks = []
    for t in texts:
        chunks.extend(semantic_chunks(t))

    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return chunks, np.array(embeddings)

chunks, chunk_embeddings = [], np.array([])

if os.path.exists(EXTRACT_DIR):
    chunks, chunk_embeddings = prepare_corpus()

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded successfully")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Preamble, Governor)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# STATS
# -------------------------------
st.write(f"📄 PDFs detected: {len(list(Path(EXTRACT_DIR).rglob('*.pdf')))}")
st.write(f"🧩 Total chunks extracted: {len(chunks)}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📝 Subjective (NCERT)", "🧠 MCQs (NCERT + UPSC)", "💬 NCERT Chatbot"]
)

# ======================================================
# 🧠 TAB 3: NCERT CHATBOT (RETRIEVAL + CHAT)
# ======================================================
with tab3:
    st.subheader("💬 Ask Anything from NCERT (Chatbot)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Ask a question strictly from NCERT...")

    if query and len(chunks) > 0:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        # 🔍 RETRIEVAL
        q_emb = embedder.encode([query])
        scores = cosine_similarity(q_emb, chunk_embeddings)[0]
        top_idx = scores.argsort()[-5:][::-1]
        retrieved = [chunks[i] for i in top_idx]

        # 🧠 ANSWER (NCERT ONLY)
        answer = "📘 **NCERT-based answer:**\n\n"
        for i, ch in enumerate(retrieved, 1):
            answer += f"{i}. {ch}\n\n"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
