# ============================================
# NCERT + UPSC Exam-Ready Generator (RAG-based)
# (Original logic preserved + subject filtering + fixed flashcards)
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
    "Business Studies": ["business", "management"]
}

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
# UTILITIES (UNCHANGED)
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
    p = pdf_path.lower()
    return any(k in p for k in SUBJECT_KEYWORDS.get(subject, []))


def load_all_texts(subject):
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
    skip = ["chapter", "unit", "page", "contents", "figure", "table"]
    return not any(k in s for k in skip) and 8 <= len(s.split()) <= 60


@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)


def retrieve_relevant_chunks(chunks, embeddings, query, standard="NCERT", top_k=10):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_vec, embeddings)[0]

    threshold = SIMILARITY_THRESHOLD_UPSC if standard == "UPSC" else SIMILARITY_THRESHOLD_NCERT

    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)

    results = []
    for ch, score in ranked:
        if score >= threshold and is_conceptual(ch):
            results.append(ch)
        if len(results) >= top_k:
            break

    return results


# --------------------------------------------
# ORIGINAL QUESTION GENERATORS (UNCHANGED)
# --------------------------------------------
def generate_subjective(topic, n, standard):
    if standard == "NCERT":
        templates = [
            f"Explain the concept of {topic}.",
            f"Describe the main features of {topic}.",
            f"Write a short note on {topic}.",
            f"Explain {topic} with suitable examples.",
        ]
    else:
        templates = [
            f"Analyse the significance of {topic} in the Indian context.",
            f"Critically examine {topic}.",
            f"Discuss the relevance of {topic} in contemporary India.",
        ]
    return templates[:n]


def generate_ncert_mcqs(chunks, topic, n):
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


# --------------------------------------------
# FLASHCARDS (FIXED – BULLET BASED)
# --------------------------------------------
def generate_flashcards(chunks, topic, n):
    cards = []
    for ch in chunks[:n]:
        bullets = re.split(r"[.;]", ch)
        bullets = [b.strip() for b in bullets if is_conceptual(b)]

        cards.append({
            "title": topic,
            "points": bullets[:5]
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
num_q = st.number_input("Number of Questions", 1, 10, 5)

# --------------------------------------------
# LOAD DATA (SUBJECT FILTER ONLY)
# --------------------------------------------
texts, chunks = [], []

if os.path.exists(EXTRACT_DIR):
    texts = load_all_texts(subject)
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
    standard = st.radio(
    "Standard",
    ["NCERT", "UPSC"],
    horizontal=True,
    key="subjective_standard"
)
    if st.button("Generate Subjective Questions"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, standard)
        qs = generate_subjective(topic, min(num_q, len(rel)), standard)
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

# --------------------------------------------
# MCQs
# --------------------------------------------
with tab2:
    standard = st.radio(
    "Standard",
    ["NCERT", "UPSC"],
    horizontal=True,
    key="mcq_standard"
)

    if st.button("Generate MCQs"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, standard)
        mcqs = generate_ncert_mcqs(rel, topic, num_q)
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
    standard = st.radio(
    "Answer Style",
    ["NCERT", "UPSC"],
    horizontal=True,
    key="chatbot_standard"
)

    q = st.text_input("Ask strictly from NCERT")
    if st.button("Ask"):
        rel = retrieve_relevant_chunks(chunks, embeddings, q, standard, top_k=6)
        if not rel:
            st.error("❌ Not found in NCERT textbooks")
        else:
            st.markdown("### 📘 NCERT-based answer:")
            for r in rel:
                st.write(r)

with tab4:
    st.subheader("📚 NCERT Flashcards (Concept Revision)")

    flashcard_mode = st.radio(
        "Flashcard Depth",
        ["NCERT", "UPSC"],
        horizontal=True,
        key="flashcard_depth"
    )

    if not topic.strip():
        st.info("Enter a topic above to generate flashcards.")
    elif not chunks:
        st.error("NCERT PDFs not loaded.")
    else:
        retrieved = retrieve_relevant_chunks(
            chunks,
            chunk_embeddings,   # ✅ this is NOW a numpy array
            topic,
            standard=flashcard_mode,
            top_k=10
        )

        if not retrieved:
            st.warning("❌ No NCERT content found for this topic.")
        else:
            st.markdown("### 🧠 Quick Revision Cards (Bullet Format)")

            for i, para in enumerate(retrieved[:num_q], 1):
                bullets = re.split(r"[.;]", para)
                bullets = [b.strip() for b in bullets if len(b.split()) >= 6]

                st.markdown(f"**📌 Flashcard {i}**")
                for b in bullets[:5]:
                    st.markdown(f"- {b}")
