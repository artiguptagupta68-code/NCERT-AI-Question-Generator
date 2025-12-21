# ============================================
# NCERT Exam-Ready Generator (RAG-based)
# Subject-filtered + Chatbot + Flashcards
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

SUBJECT_KEYWORDS = {
    "Polity": ["polity", "political", "constitution", "civics"],
    "Economics": ["economics", "economic"],
    "Sociology": ["sociology", "society"],
    "Psychology": ["psychology"],
    "Business Studies": ["business", "management"],
}

SIMILARITY_THRESHOLD = 0.35

# --------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------
st.set_page_config(page_title="NCERT RAG Generator", layout="wide")
st.title("📘 NCERT Exam-Ready Question Generator")

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
        r"(activity|exercise|project|editor|reprint|copyright|isbn).*",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def pdf_matches_subject(pdf_path, subject):
    path = pdf_path.lower()
    return any(k in path for k in SUBJECT_KEYWORDS.get(subject, []))


def load_subject_texts(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if pdf_matches_subject(str(pdf), subject):
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
    skip = ["chapter", "unit", "page", "figure", "table"]
    return not any(k in s for k in skip) and 8 <= len(s.split()) <= 60


@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)


def retrieve_relevant_chunks(chunks, embeddings, query, top_k=10):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_vec, embeddings)[0]

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)

    results = []
    for ch, score in ranked:
        if score >= SIMILARITY_THRESHOLD and is_conceptual(ch):
            results.append(ch)
        if len(results) >= top_k:
            break

    return results


# --------------------------------------------
# GENERATORS
# --------------------------------------------
def generate_subjective(topic, n):
    templates = [
        f"Explain the concept of {topic}.",
        f"Describe the main features of {topic}.",
        f"Write a short note on {topic}.",
        f"Explain {topic} with suitable examples.",
        f"Why is {topic} important?"
    ]
    return templates[:n]


def generate_mcqs(chunks, topic, n):
    sentences = [
        s for ch in chunks
        for s in re.split(r"[.;]", ch)
        if is_conceptual(s)
    ]
    random.shuffle(sentences)

    mcqs = []
    for s in sentences:
        distractors = random.sample(sentences, min(3, len(sentences)))
        options = [s] + distractors
        random.shuffle(options)

        mcqs.append({
            "q": f"Which statement best explains {topic}?",
            "options": options,
            "answer": options.index(s)
        })

        if len(mcqs) >= n:
            break

    return mcqs


def generate_flashcards(chunks, topic, n):
    cards = []
    for ch in chunks[:n]:
        cards.append({
            "q": f"What does NCERT say about {topic}?",
            "a": ch
        })
    return cards


# --------------------------------------------
# SIDEBAR
# --------------------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic")
num_q = st.number_input("Number", 1, 10, 5)

# --------------------------------------------
# LOAD SUBJECT DATA
# --------------------------------------------
texts, chunks = [], []

if os.path.exists(EXTRACT_DIR):
    texts = load_subject_texts(subject)
    for t in texts:
        chunks.extend(semantic_chunks(t))

embeddings = embed_chunks(chunks) if chunks else []

st.write(f"📄 PDFs used: {len(texts)}")
st.write(f"🧩 Chunks created: {len(chunks)}")

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Subjective", "🧠 MCQs", "💬 Ask NCERT", "🧠 Flashcards"]
)

# --------------------------------------------
# SUBJECTIVE
# --------------------------------------------
with tab1:
    if st.button("Generate Subjective"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic)
        qs = generate_subjective(topic, min(num_q, len(rel)))
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

# --------------------------------------------
# MCQs
# --------------------------------------------
with tab2:
    if st.button("Generate MCQs"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic)
        mcqs = generate_mcqs(rel, topic, num_q)
        for i, m in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {m['q']}**")
            for j, opt in enumerate(m["options"]):
                st.write(f"{chr(97+j)}) {opt}")
            st.write(f"✅ Answer: {chr(97+m['answer'])}")
            st.write("---")

# --------------------------------------------
# CHATBOT
# --------------------------------------------
with tab3:
    q = st.text_input("Ask from NCERT")
    if st.button("Ask"):
        rel = retrieve_relevant_chunks(chunks, embeddings, q, top_k=6)
        if not rel:
            st.error("Not found in NCERT")
        else:
            for r in rel:
                st.write(r)

# --------------------------------------------
# FLASHCARDS
# --------------------------------------------
with tab4:
    if st.button("Generate Flashcards"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic)
        cards = generate_flashcards(rel, topic, num_q)
        for i, c in enumerate(cards, 1):
            with st.expander(f"Flashcard {i}"):
                st.write("**Q:**", c["q"])
                st.write("**A:**", c["a"])
