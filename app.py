# multiple choice question generation for class 11,12
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

# Keywords for relevance filtering (optional)
SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "writ", "rights", "judiciary", "parliament", "emergency"],
    "Sociology": ["society", "social", "caste", "class", "gender", "movement"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "planning", "organising", "leadership", "marketing"],
    "Economics": ["economy", "growth", "gdp", "poverty", "inflation", "development"]
}

MCQ_TEMPLATES = [
    {
        "q": "What is {c}?",
        "options": [
            "A basic concept in the subject",
            "An unrelated term",
            "A random historical fact",
            "A recent development in technology"
        ],
        "answer": 0
    },
    {
        "q": "Which statement best describes {c}?",
        "options": [
            "It is central to governance and society",
            "It is purely theoretical with no application",
            "It is irrelevant today",
            "It is only used in exams"
        ],
        "answer": 0
    },
    {
        "q": "The main features of {c} include:",
        "options": [
            "Governance and regulation",
            "Ignoring legal frameworks",
            "Random social events",
            "Unrelated historical facts"
        ],
        "answer": 0
    }
]

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT MCQ Generator", layout="wide")
st.title("📘 NCERT MCQ Generator (Class XI–XII)")

# =========================
# UTILITIES
# =========================
def download_zip():
    """Download the NCERT ZIP from Google Drive if it doesn't exist"""
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    """Extract the main ZIP and all nested class-level ZIPs"""
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
    # Extract main ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    # Extract nested ZIPs
    extract_nested_zips(EXTRACT_DIR)

def extract_nested_zips(base_dir):
    """Recursively extract all ZIPs inside a folder"""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_dir = os.path.join(root, Path(file).stem)
                os.makedirs(nested_extract_dir, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, "r") as nz:
                    nz.extractall(nested_extract_dir)

def read_pdf(path):
    """Read text from PDF safely"""
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
        if len(raw.split()) < 50:
            continue
        texts.append(raw)
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
    return len(t_words & c_words) / len(t_words) >= 0.3  # relaxed threshold

# =========================
# MCQ GENERATION
# =========================
def generate_mcqs(topic, num_q):
    mcqs = []
    used = set()
    for _ in range(num_q):
        template = random.choice(MCQ_TEMPLATES)
        question = template["q"].format(c=topic)
        if question in used:
            continue
        mcqs.append({
            "question": question,
            "options": template["options"],
            "answer": template["answer"]
        })
        used.add(question)
    return mcqs

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
topic = st.text_input("Enter Topic / Chapter (e.g., Constitution, GDP, Emotion)")
num_q = st.number_input("Number of MCQs", 1, 10, 5)

# =========================
# GENERATE MCQS
# =========================
if st.button("Generate MCQs"):
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    texts = load_texts(subject)
    if not texts:
        st.error(f"No readable NCERT content found for {subject}. Make sure ZIP is loaded and contains PDFs.")
        st.stop()

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    relevant = [c for c in chunks if topic_relevant(c, topic)]
    if not relevant:
        st.info("Topic heading not found exactly. Using closest NCERT discussion.")

    mcqs = generate_mcqs(topic, num_q)

    st.success(f"Generated {len(mcqs)} NCERT-style MCQs")
    for i, mcq in enumerate(mcqs, 1):
        st.write(f"**Q{i}. {mcq['question']}**")
        for idx, opt in enumerate(mcq["options"]):
            st.write(f"{chr(97+idx)}) {opt}")
        st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
        st.write("---")
