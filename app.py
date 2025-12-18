# multiple question generation for class 11,12
# app.py
import os
import zipfile
import re
import random
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"   # Correct NCERT ZIP file ID
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "writ", "rights", "judiciary", "parliament", "emergency"],
    "Sociology": ["society", "social", "caste", "class", "gender", "movement"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "planning", "organising", "leadership", "marketing"],
    "Economics": ["economy", "growth", "gdp", "poverty", "inflation", "development"]
}

QUESTION_PATTERNS = {
    "definition": [
        "What is meant by {c}?",
        "Define {c}."
    ],
    "purpose": [
        "Why do societies need {c}?",
        "Explain the necessity of {c}."
    ],
    "features": [
        "Describe the main features of {c}.",
        "Explain any two important features of {c}."
    ],
    "role": [
        "Explain the role of {c} in a democratic system.",
        "How does {c} limit the powers of the government?"
    ],
    "analysis": [
        "Examine the relationship between {c} and democracy.",
        "Analyse the importance of {c} in modern states."
    ],
    "application": [
        "How does {c} protect the rights of citizens?",
        "Illustrate with examples the working of {c}."
    ],
    "comparision": [
        "Compare between {c}.",
        "According to you, which one is better and why?"
    ]
}

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT Question Generator", layout="wide")
st.title("📘 NCERT Question Generator (Class XI–XII)")

# =========================
# UTILITIES
# =========================
def download_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"  # Correct download URL
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
            st.write("Files extracted:")
for path in Path(EXTRACT_DIR).rglob("*"):
    st.write(path)


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# LOAD CONTENT
# =========================
def load_texts(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        raw = clean_text(read_pdf(str(pdf)))
        if len(raw.split()) < 50:  # reduce minimum words
            continue
        texts.append(raw)  # load all PDFs
    return texts



def chunk_text(text):
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if 60 <= len(p.split()) <= 200]

# =========================
# RELEVANCE CHECK
# =========================
def topic_relevant(chunk, topic):
    t_words = set(topic.lower().split())
    c_words = set(chunk.lower().split())
    if not t_words:
        return False
    return len(t_words & c_words) / len(t_words) >= 0.4

# =========================
# QUESTION LOGIC
# =========================
def select_dimensions(n):
    order = ["definition", "purpose", "features", "role", "analysis", "application"]
    return order[:n]

def generate_questions(topic, n):
    questions = []
    used = set()
    dimensions = select_dimensions(n)

    for dim in dimensions:
        template = random.choice(QUESTION_PATTERNS[dim])
        q = template.format(c=topic)
        if q not in used:
            used.add(q)
            questions.append(q)

    while len(questions) < n:
        fallback = f"Explain the significance of {topic}."
        if fallback not in used:
            questions.append(fallback)
            used.add(fallback)

    return questions

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if st.button("📥 Load NCERT Content"):
        download_zip()
        extract_zip()
        st.success("NCERT content loaded")

# =========================
# UI INPUT
# =========================
subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic / Chapter (e.g. Constitution, Writ, GDP)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# =========================
# GENERATE
# =========================
if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    texts = load_texts(subject)
    if not texts:
        st.error(f"No readable NCERT content found for {subject}.")
        st.stop()

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    relevant = [c for c in chunks if topic_relevant(c, topic)]

    if not relevant:
        st.info("Topic heading not found exactly. Using closest NCERT discussion.")

    questions = generate_questions(topic, num_q)

    st.success(f"Generated {len(questions)} NCERT-style questions")
    for i, q in enumerate(questions, 1):
        st.write(f"{i}. {q}")
