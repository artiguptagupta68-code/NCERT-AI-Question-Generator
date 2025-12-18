# app.py
import os
import zipfile
from pathlib import Path
import re
import random

import streamlit as st
import gdown
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert_subjects.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "rights", "judiciary", "parliament"],
    "Sociology": ["caste", "class", "gender", "society"],
    "Psychology": ["emotion", "learning", "memory", "motivation"],
    "Business Studies": ["management", "planning", "marketing"],
    "Economics": ["gdp", "growth", "inflation", "poverty"]
}

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config("UPSC NCERT MCQ Generator", layout="wide")
st.title("📘 UPSC NCERT MCQ Generator")

# =========================
# SESSION STATE
# =========================
if "ncert_loaded" not in st.session_state:
    st.session_state.ncert_loaded = False

if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = []

# =========================
# UTILITIES
# =========================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

def load_all_pdf_text():
    texts = []
    pdf_count = 0
    page_count = 0

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        pdf_count += 1
        try:
            reader = PdfReader(str(pdf))
            for page in reader.pages:
                txt = page.extract_text()
                if txt and len(txt.split()) > 20:
                    texts.append(txt)
                    page_count += 1
        except:
            continue

    return texts, pdf_count, page_count

def semantic_chunk(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i:i+4])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks

def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [
        c for c in chunks
        if topic in c.lower() or any(k in c.lower() for k in keys)
    ]

def generate_upsc_mcqs(chunks, topic, n):
    mcqs = []
    random.shuffle(chunks)

    for c in chunks:
        if len(mcqs) >= n:
            break

        mcqs.append({
            "q": f"With reference to {topic}, consider the following statements:",
            "options": [
                "It plays a significant role in governance",
                "It has constitutional backing",
                "It is irrelevant in modern democracy",
                "It violates fundamental rights"
            ],
            "answer": 0
        })

    return mcqs

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📂 NCERT Setup")

    if st.button("Load NCERT Content", key="load_btn"):
        with st.spinner("Downloading & extracting NCERT PDFs..."):
            download_and_extract()
            texts, pdfs, pages = load_all_pdf_text()

        st.session_state.pdf_texts = texts
        st.session_state.ncert_loaded = True

        st.success("NCERT Loaded Successfully")
        st.info(f"PDFs found: {pdfs}")
        st.info(f"Pages with text: {pages}")

# =========================
# MAIN UI
# =========================
subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic (e.g., Constitution, GDP, Emotion)")
num_q = st.slider("Number of MCQs", 5, 20, 10)

if st.button("Generate UPSC MCQs", key="gen_btn"):
    if not st.session_state.ncert_loaded:
        st.error("Please load NCERT content first")
        st.stop()

    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()

    texts = st.session_state.pdf_texts
    if not texts:
        st.error("NCERT loaded but no readable text found")
        st.stop()

    chunks = []
    for t in texts:
        chunks.extend(semantic_chunk(t))

    relevant = boolean_filter(chunks, topic, subject)
    if len(relevant) < 5:
        relevant = chunks[:15]

    mcqs = generate_upsc_mcqs(relevant, topic, num_q)

    st.success(f"Generated {len(mcqs)} UPSC-style MCQs")

    for i, m in enumerate(mcqs, 1):
        st.write(f"**Q{i}. {m['q']}**")
        for idx, opt in enumerate(m["options"]):
            st.write(f"{chr(97+idx)}) {opt}")
        st.write("✅ **Answer:** a")
        st.write("---")
