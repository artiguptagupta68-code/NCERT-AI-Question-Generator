# ==========================================
# NCERT Exam-Ready Generator + Chatbot (RAG)
# ==========================================

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
NCERT_FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT Exam Generator (RAG)", layout="wide")
st.title("📘 NCERT Exam-Ready Generator + Chatbot (RAG)")

# -------------------------------
# DOWNLOAD & EXTRACT
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={NCERT_FILE_ID}", ZIP_PATH, quiet=False)

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

# -------------------------------
# PDF PROCESSING
# -------------------------------
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|exercise|project|glossary|isbn|copyright).*",
                  " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for s in sentences:
        if 15 <= len(s.split()) <= 60:
            chunks.append(s.strip())
    return chunks

# -------------------------------
# BUILD NCERT VECTOR STORE
# -------------------------------
@st.cache_resource
def build_vector_store():
    chunks = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = clean_text(read_pdf(str(pdf)))
        chunks.extend(chunk_text(text))

    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return chunks, np.array(embeddings)

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve_chunks(query, chunks, embeddings, k=6):
    q_emb = embedder.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_idx]

# -------------------------------
# QUESTION GENERATORS (NCERT ONLY)
# -------------------------------
def generate_subjective(topic, context):
    return [
        f"Explain the concept of {topic} with reference to NCERT.",
        f"Describe the key features of {topic}.",
        f"Discuss the significance of {topic} as explained in NCERT textbooks.",
        f"Illustrate {topic} with suitable examples from NCERT.",
        f"Why is {topic} important in the Indian context?"
    ]

def generate_mcqs(context, topic, n):
    mcqs = []
    used = set()
    random.shuffle(context)

    for sentence in context:
        if sentence in used:
            continue
        used.add(sentence)

        distractors = random.sample(context, min(3, len(context)))
        options = [sentence] + distractors
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best explains **{topic}**?",
            "options": options,
            "answer": options.index(sentence)
        })

        if len(mcqs) >= n:
            break
    return mcqs

def generate_assertion_reason(context, n):
    qs = []
    random.shuffle(context)
    for i in range(min(n, len(context)-1)):
        qs.append({
            "A": context[i],
            "R": context[i+1],
            "answer": "a"
        })
    return qs

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD DATA
# -------------------------------
chunks, embeddings = [], []

if os.path.exists(EXTRACT_DIR):
    chunks, embeddings = build_vector_store()

st.write(f"📄 PDFs detected: {len(list(Path(EXTRACT_DIR).rglob('*.pdf')))}")
st.write(f"🧩 Total NCERT chunks: {len(chunks)}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Subjective", "🧠 MCQs / AR", "💬 Ask NCERT (Chatbot)"])

# -------------------------------
# SUBJECTIVE
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        retrieved = retrieve_chunks(topic, chunks, embeddings)
        if not retrieved:
            st.error("No NCERT content found.")
        else:
            qs = generate_subjective(topic, retrieved)[:num_q]
            for i, q in enumerate(qs, 1):
                st.write(f"{i}. {q}")

# -------------------------------
# MCQs / ASSERTION-REASON
# -------------------------------
with tab2:
    q_type = st.radio("Question Type", ["MCQs", "Assertion-Reason"])

    if st.button("Generate Questions"):
        retrieved = retrieve_chunks(topic, chunks, embeddings)
        if not retrieved:
            st.error("No NCERT content found.")
        else:
            if q_type == "MCQs":
                mcqs = generate_mcqs(retrieved, topic, num_q)
                for i, m in enumerate(mcqs, 1):
                    st.write(f"**Q{i}. {m['q']}**")
                    for j, opt in enumerate(m["options"]):
                        st.write(f"{chr(97+j)}) {opt}")
                    st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                    st.write("---")
            else:
                ar = generate_assertion_reason(retrieved, num_q)
                for i, q in enumerate(ar, 1):
                    st.write(f"**Assertion (A):** {q['A']}")
                    st.write(f"**Reason (R):** {q['R']}")
                    st.write("a) Both A and R are true and R explains A")
                    st.write("b) Both true but R not explanation")
                    st.write("c) A true, R false")
                    st.write("d) A false, R true")
                    st.write("✅ Answer: a")
                    st.write("---")

# -------------------------------
# CHATBOT (NCERT-ONLY)
# -------------------------------
with tab3:
    user_q = st.text_input("Ask anything strictly from NCERT:")

    if st.button("Get Answer"):
        retrieved = retrieve_chunks(user_q, chunks, embeddings)
        if not retrieved:
            st.error("No NCERT content found.")
        else:
            st.markdown("📘 **NCERT-based answer:**")
            st.write(" ".join(retrieved))
