# ============================================
# NCERT + UPSC Exam-Ready Generator (RAG Based)
# ============================================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------
# CONFIG
# --------------------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["constitution", "preamble", "parliament", "president", "governor", "judiciary"],
    "Economics": ["economy", "growth", "inflation", "poverty"],
    "Sociology": ["society", "culture", "social"],
    "Psychology": ["behaviour", "learning", "cognition"],
    "Business Studies": ["management", "planning", "organisation"]
}

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# --------------------------------------------
# DATA LOADING
# --------------------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            out = zfile.parent / zfile.stem
            out.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(out)
        except:
            pass


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""


def clean_text(text):
    text = re.sub(r"(reprint|isbn|contents|page\s+\d+).*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, size=3):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sentences[i:i+size]) for i in range(0, len(sentences), size)]


# --------------------------------------------
# LOAD + EMBEDDINGS
# --------------------------------------------
@st.cache_resource
def load_embeddings():
    chunks, subjects = [], []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = clean_text(read_pdf(str(pdf)))
        if len(text.split()) < 100:
            continue

        for ch in chunk_text(text):
            if 20 <= len(ch.split()) <= 120:
                chunks.append(ch)
                subjects.append(pdf.name.lower())

    embeddings = EMBED_MODEL.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, subjects, embeddings, index


# --------------------------------------------
# RETRIEVAL
# --------------------------------------------
def retrieve_chunks(query, subject, top_k=5):
    q_emb = EMBED_MODEL.encode([query])
    D, I = index.search(q_emb, top_k * 3)

    results = []
    for idx in I[0]:
        if subject.lower() in subjects[idx]:
            results.append(chunks[idx])
        if len(results) >= top_k:
            break

    return results


# --------------------------------------------
# GENERATORS
# --------------------------------------------
def ncert_subjective(topic, level="NCERT"):
    if level == "NCERT":
        return [
            f"Explain the term {topic}.",
            f"Describe the main features of {topic}.",
            f"Why is {topic} important?"
        ]
    else:
        return [
            f"Analyse the significance of {topic} in the Indian constitutional framework.",
            f"Discuss {topic} in the context of Indian democracy."
        ]


def chatbot_answer(query, subject):
    retrieved = retrieve_chunks(query, subject)

    if not retrieved:
        return "❌ No relevant NCERT content found for this topic."

    answer = " ".join(retrieved)
    return answer


# --------------------------------------------
# SIDEBAR
# --------------------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", list(SUBJECTS.keys()))
topic = st.text_input("Topic / Question")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# --------------------------------------------
# LOAD DATA
# --------------------------------------------
if os.path.exists(EXTRACT_DIR):
    chunks, subjects, embeddings, index = load_embeddings()
    st.info(f"📄 Chunks indexed: {len(chunks)}")

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Subjective", "🧠 MCQs / AR", "💬 Ask NCERT"])

# --------------------------------------------
# SUBJECTIVE
# --------------------------------------------
with tab1:
    level = st.radio("Question Standard", ["NCERT", "UPSC"])
    if st.button("Generate Subjective"):
        qs = ncert_subjective(topic, level)
        for i, q in enumerate(qs[:num_q], 1):
            st.write(f"{i}. {q}")

# --------------------------------------------
# MCQs / AR (Concept-based)
# --------------------------------------------
with tab2:
    if st.button("Generate MCQs / AR"):
        retrieved = retrieve_chunks(topic, subject)

        if not retrieved:
            st.error("No NCERT content found.")
        else:
            fact = retrieved[0]
            st.write(f"**Assertion (A):** {fact}")
            st.write(f"**Reason (R):** {retrieved[1] if len(retrieved) > 1 else fact}")
            st.write("a) Both A and R are true and R is the correct explanation of A")
            st.write("b) Both A and R are true but R is not the correct explanation of A")
            st.write("c) A is true but R is false")
            st.write("d) A is false but R is true")

# --------------------------------------------
# CHATBOT (STRICT NCERT)
# --------------------------------------------
with tab3:
    q = st.text_input("Ask anything strictly from NCERT:")
    if st.button("Ask"):
        ans = chatbot_answer(q, subject)
        st.markdown("### 📘 NCERT-based answer:")
        st.write(ans)
