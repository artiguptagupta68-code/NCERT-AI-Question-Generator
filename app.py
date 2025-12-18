# NCERT Question Generator with Subjective & MCQs
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
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
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

GENERIC_PATTERNS = [
    "Explain the concept of {c}.",
    "Describe the importance of {c}.",
    "Discuss the role of {c} in society.",
    "Why is {c} significant?"
]

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
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    extract_nested_zips(EXTRACT_DIR)

def extract_nested_zips(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_dir = os.path.join(root, Path(file).stem)
                os.makedirs(nested_extract_dir, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, "r") as nz:
                    nz.extractall(nested_extract_dir)

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
# SEMANTIC CHUNKING
# =========================
def semantic_chunk(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i:i+4])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks

# =========================
# BOOLEAN FILTER
# =========================
def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [c for c in chunks if topic in c.lower() or any(k in c.lower() for k in keys)]

# =========================
# QUESTION VALIDATION
# =========================
def is_valid_question(q):
    return len(q.split()) > 5

# =========================
# SUBJECTIVE QUESTION GENERATION
# =========================
def generate_subjective(chunks, topic, n):
    questions, seen = [], set()
    random.shuffle(chunks)
    for c in chunks:
        q = random.choice(GENERIC_PATTERNS).format(c=topic)
        if q not in seen and is_valid_question(q):
            seen.add(q)
            questions.append(q)
        if len(questions) >= n:
            break
    if not questions:
        questions = [f"Explain the concept of {topic}." for _ in range(n)]
    return questions

# =========================
# MCQ GENERATION
# =========================
def generate_mcqs(topic, n):
    mcqs = []
    used = set()
    random.shuffle(MCQ_TEMPLATES)
    for t in MCQ_TEMPLATES:
        if len(mcqs) >= n:
            break
        question = t["q"].format(c=topic)
        if question in used:
            continue
        mcqs.append({"question": question, "options": t["options"], "answer": t["answer"]})
        used.add(question)
    while len(mcqs) < n:
        mcqs.append({"question": f"What is {topic}?", "options": ["Option1","Option2","Option3","Option4"], "answer": 0})
    return mcqs

# =========================
# LOAD PDF TEXTS
# =========================
def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

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
topic = st.text_input("Enter Topic / Chapter (e.g. Constitution, GDP, Emotion)")
num_q = st.number_input("Number of Questions", 1, 10, 5)
level = st.radio("Select Level", ["NCERT Level", "UPSC Level"])

tab1, tab2 = st.tabs(["Subjective Questions", "MCQs"])

# =========================
# GENERATE SUBJECTIVE
# =========================
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.warning("Enter a topic")
            st.stop()

        texts = load_all_texts()
        if not texts:
            st.error("No readable NCERT PDFs found")
            st.stop()

        chunks = []
        for t in texts:
            chunks.extend(semantic_chunk(t))

        relevant = boolean_filter(chunks, topic, subject)
        if len(relevant) < 5:
            relevant = chunks[:15]

        questions = generate_subjective(relevant, topic, num_q)

        st.success(f"Generated {len(questions)} Subjective Questions ({level})")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")

# =========================
# GENERATE MCQs
# =========================
with tab2:
    if st.button("Generate MCQs"):
        if not topic.strip():
            st.warning("Enter a topic")
            st.stop()

        texts = load_all_texts()
        if not texts:
            st.error("No readable NCERT PDFs found")
            st.stop()

        chunks = []
        for t in texts:
            chunks.extend(semantic_chunk(t))

        relevant = boolean_filter(chunks, topic, subject)
        if len(relevant) < 5:
            relevant = chunks[:15]

        mcqs = generate_mcqs(topic, num_q)

        st.success(f"Generated {len(mcqs)} MCQs ({level})")
        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            st.write("---")
