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
st.set_page_config("NCERT UPSC MCQ Generator", layout="wide")
st.title("📘 UPSC NCERT MCQ Generator")

# =========================
# UTILITIES
# =========================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            if p.extract_text():
                text += p.extract_text() + "\n"
        return text
    except:
        return ""

def semantic_chunk(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i:i+4])
        if len(chunk.split()) > 25:
            chunks.append(chunk)
    return chunks

def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [
        c for c in chunks
        if topic in c.lower() or any(k in c.lower() for k in keys)
    ]

def generate_mcqs(chunks, topic, n):
    mcqs, used = [], set()
    random.shuffle(chunks)

    for c in chunks:
        if len(mcqs) >= n:
            break

        q = f"With reference to {topic}, consider the following statements:"
        if q in used:
            continue

        options = [
            "It plays a crucial role in its respective field",
            "It has no relevance in modern society",
            "It contradicts constitutional principles",
            "It is unrelated to governance"
        ]

        mcqs.append({
            "q": q,
            "options": options,
            "answer": 0
        })
        used.add(q)

    return mcqs

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📂 Data Setup")
    if st.button("Load NCERT Content", key="load_ncert_btn"):
        download_and_extract()
        st.success("NCERT content loaded")

# =========================
# MAIN UI
# =========================
subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic (e.g., Constitution, GDP, Emotion)")
num_q = st.slider("Number of MCQs", 5, 20, 10)

if st.button("Generate UPSC MCQs", key="generate_btn"):
    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()

    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = read_pdf(str(pdf))
        if len(t.split()) > 100:
            texts.append(t)

    if not texts:
        st.error("No readable NCERT text found")
        st.stop()

    chunks = []
    for t in texts:
        chunks.extend(semantic_chunk(t))

    relevant = boolean_filter(chunks, topic, subject)
    if len(relevant) < 5:
        relevant = chunks[:10]

    mcqs = generate_mcqs(relevant, topic, num_q)

    st.success(f"Generated {len(mcqs)} UPSC-style MCQs")

    for i, m in enumerate(mcqs, 1):
        st.write(f"**Q{i}. {m['q']}**")
        for idx, opt in enumerate(m["options"]):
            st.write(f"{chr(97+idx)}) {opt}")
        st.write(f"✅ **Answer:** a")
        st.write("---")
