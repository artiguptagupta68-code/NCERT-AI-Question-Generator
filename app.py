# app.py — NCERT & UPSC Exam-Ready Question Generator

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
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert_books.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "preamble", "rights", "judiciary", "parliament", "federal"],
    "Sociology": ["society", "caste", "class", "gender", "movement"],
    "Psychology": ["behaviour", "learning", "memory", "emotion"],
    "Business Studies": ["management", "planning", "organising", "leadership"],
    "Economics": ["economy", "growth", "gdp", "inflation", "poverty"]
}

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT & UPSC Question Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# =========================
# UTILS
# =========================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(acknowledgement|contributors|isbn|reprint).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 200:
            texts.append(t)
    return texts

def chunk_text(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i:i+4])
        if 40 <= len(chunk.split()) <= 120:
            chunks.append(chunk)
    return chunks

# =========================
# SUBJECTIVE QUESTIONS
# =========================
SUBJECTIVE_PATTERNS = [
    "Define {c}.",
    "Explain the concept of {c}.",
    "Describe the main features of {c}.",
    "Why is {c} important? Explain.",
    "Discuss the significance of {c}."
]

def extract_concepts(chunk, topic):
    concepts = re.findall(r"\b[A-Z][a-zA-Z ]{3,40}\b", chunk)
    if topic.lower() not in " ".join(concepts).lower():
        concepts.append(topic)
    return list(set(concepts))[:3]

def generate_subjective(chunks, topic, n):
    questions, seen = [], set()
    random.shuffle(chunks)
    for ch in chunks:
        for c in extract_concepts(ch, topic):
            q = random.choice(SUBJECTIVE_PATTERNS).format(c=c)
            if q not in seen:
                seen.add(q)
                questions.append(q)
            if len(questions) >= n:
                return questions
    return [f"Explain the concept of {topic}." for _ in range(n)]

# =========================
# NCERT-STYLE MCQs
# =========================
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    random.shuffle(chunks)

    for ch in chunks:
        if topic.lower() not in ch.lower():
            continue

        statements = re.split(r';|\. ', ch)
        statements = [s.strip() for s in statements if 8 <= len(s.split()) <= 25]

        if len(statements) < 4:
            continue

        options = random.sample(statements, 4)
        correct = options[0]

        mcqs.append({
            "question": f"Which of the following best describes {topic}?",
            "options": options,
            "answer": options.index(correct)
        })

        if len(mcqs) >= n:
            break

    return mcqs

# =========================
# ASSERTION–REASON (UPSC)
# =========================
ASSERTION_REASON_OPTIONS = [
    "Both A and R are true and R is the correct explanation of A",
    "Both A and R are true but R is not the correct explanation of A",
    "A is true but R is false",
    "A is false but R is true"
]

ASSERTION_REASON_BANK = {
    "preamble": [
        {
            "A": "The Preamble declares India to be a sovereign, socialist, secular and democratic republic.",
            "R": "The Preamble reflects the objectives and philosophy of the Constitution.",
            "ans": 0
        },
        {
            "A": "The Preamble is a part of the Indian Constitution.",
            "R": "It can be amended under Article 368.",
            "ans": 1
        }
    ],
    "fundamental rights": [
        {
            "A": "Fundamental Rights are justiciable in nature.",
            "R": "They are enforceable through writs issued by constitutional courts.",
            "ans": 0
        }
    ]
}

def generate_assertion_reason(topic, n):
    key = topic.lower()
    if key not in ASSERTION_REASON_BANK:
        return []
    pool = ASSERTION_REASON_BANK[key]
    random.shuffle(pool)
    return pool[:n]

# =========================
# UI
# =========================
with st.sidebar:
    if st.button("Load NCERT Content"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic / Chapter")
num_q = st.number_input("Number of Questions", 1, 20, 5)

q_type = st.radio(
    "Select Question Type",
    ["Subjective", "NCERT MCQs", "Assertion–Reason (UPSC)"]
)

if st.button("Generate Questions"):
    texts = load_all_texts()
    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    relevant = [c for c in chunks if topic.lower() in c.lower()]
    if not relevant:
        relevant = chunks[:20]

    if q_type == "Subjective":
        qs = generate_subjective(relevant, topic, num_q)
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

    elif q_type == "NCERT MCQs":
        mcqs = generate_ncert_mcqs(relevant, topic, num_q)
        for i, m in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {m['question']}**")
            for idx, opt in enumerate(m["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + m['answer'])}")
            st.write("---")

    else:
        ars = generate_assertion_reason(topic, num_q)
        if not ars:
            st.warning("Assertion–Reason not available for this topic.")
        for i, q in enumerate(ars, 1):
            st.write(f"**Q{i}.**")
            st.write(f"**Assertion (A):** {q['A']}")
            st.write(f"**Reason (R):** {q['R']}")
            for idx, opt in enumerate(ASSERTION_REASON_OPTIONS):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + q['ans'])}")
            st.write("---")
