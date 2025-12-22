# ============================================
# NCERT + UPSC Exam-Ready Generator (RAG-based)
# Streamlit-safe | Original logic preserved
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
    "Polity": ["constitution", "civics", "political"],
    "Economics": ["economics", "economic"],
    "Sociology": ["society", "sociology"],
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
# EMBEDDING MODEL
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
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|exercise|project|isbn|copyright).*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

def pdf_matches_subject(path, subject):
    return any(k in path.lower() for k in SUBJECT_KEYWORDS[subject])

def load_all_texts(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if pdf_matches_subject(str(pdf), subject):
            t = clean_text(read_pdf(str(pdf)))
            if len(t.split()) > 50:
                texts.append(t)
    return texts

def semantic_chunks(text):
    sents = re.split(r"(?<=[.])\s+", text)
    return [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]

def is_conceptual(s):
    skip = ["chapter", "unit", "page", "contents", "figure", "table"]
    return 8 <= len(s.split()) <= 60 and not any(k in s.lower() for k in skip)

@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

def retrieve_relevant_chunks(chunks, embeddings, query, standard="NCERT", top_k=10):
    q_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_vec, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if standard == "UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold and is_conceptual(c)][:top_k]

# --------------------------------------------
# GENERATORS
# --------------------------------------------
def generate_subjective(topic, n, standard):
    if standard == "NCERT":
        qs = [
            f"Explain the concept of {topic}.",
            f"Describe the main features of {topic}.",
            f"Write a short note on {topic}.",
            f"Explain {topic} with examples."
        ]
    else:
        qs = [
            f"Analyse the significance of {topic}.",
            f"Critically examine {topic}.",
            f"Discuss the relevance of {topic} in contemporary India."
        ]
    return qs[:n]

def generate_ncert_mcqs(chunks, topic, n):
    sents = [s for ch in chunks for s in re.split(r"[.;]", ch) if is_conceptual(s)]
    random.shuffle(sents)
    mcqs = []
    for s in sents[:n]:
        opts = random.sample(sents, min(4, len(sents)))
        if s not in opts:
            opts[0] = s
        random.shuffle(opts)
        mcqs.append({
            "q": f"Which statement best explains {topic}?",
            "options": opts,
            "answer": opts.index(s)
        })
    return mcqs

def normalize_text(s):
    s = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().capitalize()

def generate_flashcards(chunks, topic, mode, max_cards):
    cards = []
    for ch in chunks:
        sents = [normalize_text(s) for s in re.split(r"[.;]", ch) if is_conceptual(s)]
        if len(sents) < 2:
            continue
        if mode == "NCERT":
            para = f"{sents[0]}. {sents[1]}."
        else:
            para = (
                f"{sents[0]}. "
                f"This idea has constitutional and democratic significance. "
                f"It is frequently used in UPSC answers to interpret governance and rights."
            )
        cards.append({"title": topic.capitalize(), "content": para})
        if len(cards) >= max_cards:
            break
    return cards

# --------------------------------------------
# ACTIVE LEARNING (NEW)
# --------------------------------------------
def infer_topic_from_context(chunks):
    words = []
    for ch in chunks:
        words += re.findall(r"\b[A-Z][a-z]{4,}\b", ch)
    return max(set(words), key=words.count) if words else "Concept"

def generate_fill_blanks(sentence):
    words = sentence.split()
    key_words = [w for w in words if len(w) > 6]
    if not key_words:
        return sentence
    target = random.choice(key_words)
    return sentence.replace(target, "_" * len(target))

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
# LOAD DATA
# --------------------------------------------
texts, chunks = [], []
if os.path.exists(EXTRACT_DIR):
    texts = load_all_texts(subject)
    for t in texts:
        chunks.extend(semantic_chunks(t))

embeddings = embed_chunks(chunks) if chunks else np.array([])

# --------------------------------------------
# TABS (FIXED)
# --------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 Subjective", "🧠 MCQs", "💬 Ask NCERT", "🧠 Flashcards", "📝 Active Learning"]
)

# SUBJECTIVE
with tab1:
    std = st.radio("Standard", ["NCERT", "UPSC"], key="sub_std", horizontal=True)
    if st.button("Generate Subjective"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, std)
        for i, q in enumerate(generate_subjective(topic, min(num_q, len(rel)), std), 1):
            st.write(f"{i}. {q}")

# MCQs
with tab2:
    std = st.radio("Standard", ["NCERT", "UPSC"], key="mcq_std", horizontal=True)
    if st.button("Generate MCQs"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, std)
        for i, m in enumerate(generate_ncert_mcqs(rel, topic, num_q), 1):
            st.write(f"**Q{i}. {m['q']}**")
            for j, o in enumerate(m["options"]):
                st.write(f"{chr(97+j)}) {o}")
            st.write(f"✅ Answer: {chr(97+m['answer'])}")
            st.write("---")

# CHATBOT
with tab3:
    std = st.radio("Answer Style", ["NCERT", "UPSC"], key="chat_std", horizontal=True)
    q = st.text_input("Ask strictly from NCERT")
    if st.button("Ask"):
        rel = retrieve_relevant_chunks(chunks, embeddings, q, std, 6)
        for r in rel:
            st.write(r)

# FLASHCARDS
with tab4:
    std = st.radio("Flashcard Depth", ["NCERT", "UPSC"], key="flash_std", horizontal=True)
    rel = retrieve_relevant_chunks(chunks, embeddings, topic, std, 10)
    for i, c in enumerate(generate_flashcards(rel, topic, std, num_q), 1):
        st.markdown(f"### 📌 Flashcard {i}: {c['title']}")
        st.write(c["content"])

# ACTIVE LEARNING
with tab5:
    st.subheader("Active Learning")
    rel = retrieve_relevant_chunks(chunks, embeddings, topic, "NCERT", 5)

    if rel:
        inferred = infer_topic_from_context(rel)
        st.markdown(f"**Inferred Topic:** {inferred}")

        st.markdown("### Fill in the blanks")
        for ch in rel:
            sent = normalize_text(ch)
            st.write(generate_fill_blanks(sent))
