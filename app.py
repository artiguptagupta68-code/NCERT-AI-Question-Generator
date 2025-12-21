# ============================================
# NCERT + UPSC Exam-Ready Generator (RAG-based)
# SUBJECT-LEVEL FILTERING ENABLED ✅
# ============================================

import os, zipfile, re, random
from pathlib import Path

import streamlit as st
import gdown
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# CONFIG
# --------------------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45

# --------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# --------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --------------------------------------------
# UTILITIES
# --------------------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except:
            pass


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""


def clean_text(text):
    text = re.sub(
        r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------
# SUBJECT-LEVEL LOADING ✅
# --------------------------------------------
def load_subject_texts(subject):
    texts = []
    subject_key = subject.lower().replace(" ", "")

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if subject_key in str(pdf).lower():
            t = clean_text(read_pdf(str(pdf)))
            if len(t.split()) > 50:
                texts.append(t)

    return texts


def semantic_chunks(text):
    sentences = re.split(r"(?<=[.])\s+", text)
    return [
        " ".join(sentences[i:i+3])
        for i in range(0, len(sentences), 3)
        if len(sentences[i:i+3]) > 0
    ]


def is_conceptual(sentence):
    s = sentence.lower()
    skip = ["chapter", "unit", "page", "contents", "glossary", "figure", "table"]
    return not any(k in s for k in skip) and 8 <= len(s.split()) <= 60


# --------------------------------------------
# EMBEDDINGS & RETRIEVAL
# --------------------------------------------
@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)


def retrieve_relevant_chunks(chunks, embeddings, query, standard="NCERT", top_k=20):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(query_vec, embeddings)[0]

    threshold = SIMILARITY_THRESHOLD_UPSC if standard == "UPSC" else SIMILARITY_THRESHOLD_NCERT

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)

    results = []
    for ch, score in ranked:
        if score >= threshold and is_conceptual(ch):
            results.append(ch)
        if len(results) >= top_k:
            break

    return results


def count_possible_questions(chunks):
    return len([
        s for ch in chunks for s in re.split(r"[.;]", ch) if is_conceptual(s)
    ])

# --------------------------------------------
# SUBJECTIVE QUESTIONS
# --------------------------------------------
def generate_subjective(topic, n, standard="NCERT"):
    if standard == "NCERT":
        templates = [
            f"Explain the concept of {topic}.",
            f"Describe the main features of {topic}.",
            f"Write a short note on {topic}.",
            f"Why is {topic} important in the Indian context?",
            f"Explain {topic} with suitable examples.",
        ]
    else:
        templates = [
            f"Analyse the constitutional significance of {topic}.",
            f"Discuss the role of {topic} in India’s governance framework.",
            f"Critically examine the relevance of {topic} in contemporary India.",
            f"Evaluate the functioning of {topic}.",
            f"Discuss challenges associated with {topic}.",
        ]
    return templates[:n]

# --------------------------------------------
# FLASHCARDS
# --------------------------------------------
def generate_flashcards(chunks, topic, n):
    cards = []
    sentences = [
        s.strip()
        for ch in chunks
        for s in re.split(r"[.;]", ch)
        if is_conceptual(s)
    ]
    random.shuffle(sentences)

    for s in sentences[:n]:
        cards.append({
            "q": f"What does NCERT say about {topic}?",
            "a": s
        })
    return cards

# --------------------------------------------
# SIDEBAR
# --------------------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Governor, Inflation)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# --------------------------------------------
# LOAD SUBJECT CONTENT ✅
# --------------------------------------------
texts, chunks = [], []

if os.path.exists(EXTRACT_DIR):
    texts = load_subject_texts(subject)
    for t in texts:
        chunks.extend(semantic_chunks(t))

chunk_embeddings = embed_chunks(chunks) if chunks else []

st.write(f"📄 Subject PDFs loaded: {len(texts)}")
st.write(f"🧩 Subject chunks: {len(chunks)}")

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Subjective", "🧠 MCQs", "💬 Ask NCERT / UPSC", "🧠 Flashcards"]
)

# --------------------------------------------
# SUBJECTIVE TAB
# --------------------------------------------
with tab1:
    standard = st.radio("Standard", ["NCERT", "UPSC"])
    if st.button("Generate Subjective"):
        retrieved = retrieve_relevant_chunks(chunks, chunk_embeddings, topic, standard)
        if not retrieved:
            st.error("No NCERT content found.")
        else:
            qs = generate_subjective(topic, min(num_q, len(retrieved)), standard)
            for i, q in enumerate(qs, 1):
                st.write(f"{i}. {q}")

# --------------------------------------------
# CHATBOT TAB
# --------------------------------------------
with tab3:
    mode = st.radio("Answer Style", ["NCERT", "UPSC"], horizontal=True)
    q = st.text_input("Ask strictly from NCERT")

    if st.button("Ask"):
        retrieved = retrieve_relevant_chunks(chunks, chunk_embeddings, q, mode, top_k=6)
        if not retrieved:
            st.error("Answer not found in NCERT.")
        else:
            st.markdown("### 📘 NCERT-based Answer")
            for r in retrieved:
                st.write(r)

# --------------------------------------------
# FLASHCARDS TAB
# --------------------------------------------
with tab4:
    if st.button("Generate Flashcards"):
        retrieved = retrieve_relevant_chunks(chunks, chunk_embeddings, topic, "NCERT", top_k=10)
        cards = generate_flashcards(retrieved, topic, min(num_q, len(retrieved)))
        for i, c in enumerate(cards, 1):
            with st.expander(f"Flashcard {i}"):
                st.markdown(f"**Q:** {c['q']}")
                st.markdown(f"**A:** {c['a']}")
