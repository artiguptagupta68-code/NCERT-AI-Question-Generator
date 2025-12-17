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
# NCERT Subject ZIP from Drive
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert_subjects.zip"
EXTRACT_DIR = "ncert_subjects_data"

SUBJECTS = [
    "Polity",
    "Sociology",
    "Psychology",
    "Business Studies",
    "Economics"
]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "rights", "freedom", "emergency", "judiciary", "parliament"],
    "Sociology": ["society", "social", "movement", "caste", "class", "gender"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "leadership", "planning", "controlling", "marketing"],
    "Economics": ["economy", "gdp", "growth", "inflation", "poverty", "development"]
}

# =========================
# STREAMLIT
# =========================
st.set_page_config(page_title="NCERT → UPSC Q Generator", layout="wide")
st.title("📘 NCERT → UPSC-style Question Generator (Offline)")

# =========================
# UTILITIES
# =========================
def download_ncert_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_all_zips(base_dir):
    """ Recursively extract nested ZIP files """
    for _ in range(5):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".zip"):
                    p = os.path.join(root, file)
                    try:
                        with zipfile.ZipFile(p, "r") as z:
                            z.extractall(root)
                        os.remove(p)
                    except:
                        pass

def extract_zip():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    extract_all_zips(EXTRACT_DIR)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        st.warning(f"Failed to read PDF: {os.path.basename(path)} — {e}")
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# LOAD ALL TEXTS
# =========================

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        try:
            reader = PdfReader(str(pdf))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            text = clean_text(text)
            if len(text.strip()) > 20:  # lower threshold to include more text
                texts.append(text)
        except Exception as e:
            st.warning(f"Failed to read PDF {pdf.name}: {e}")
    return texts
all_pdfs = list(Path(EXTRACT_DIR).rglob("*.pdf"))
st.text(f"Total PDFs found: {len(all_pdfs)}")


def chunk_text(text):
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if 60 <= len(p.split()) <= 200]

# =========================
# SEMANTIC FILTER
# =========================
def relevant_chunks(chunks, topic, subject):
    topic_lower = topic.lower()
    matched = [c for c in chunks if topic_lower in c.lower()]
    if len(matched) >= 3:
        return matched
    # fallback: subject keyword match
    kws = SUBJECT_KEYWORDS.get(subject, [])
    fallback = [c for c in chunks if any(k in c.lower() for k in kws)]
    return fallback if fallback else chunks[:10]

# =========================
# QUESTION GENERATION
# =========================
GENERIC_SUBJECTIVE = [
    "Explain the concept of {}.",
    "Discuss {} in detail.",
    "Why is {} important in context of {}?",
    "Describe the main aspects of {}.",
    "Analyse the role of {}.",
    "Evaluate the significance of {}."
]

def generate_subjective_questions(chunks, topic, subject, n):
    questions = []
    seen = set()
    for ch in chunks:
        sent = re.split(r"(?<=[.?!])\s+", ch)
        for s in sent:
            if topic.lower() in s.lower() and len(s.split()) > 8:
                q = random.choice(GENERIC_SUBJECTIVE).format(topic, subject)
                if q not in seen:
                    seen.add(q)
                    questions.append(q)
                if len(questions) >= n:
                    return questions
    # fallback
    while len(questions) < n:
        questions.append(random.choice(GENERIC_SUBJECTIVE).format(topic, subject))
    return questions

MCQ_PATTERNS = [
    "Which of the following best describes {}?",
    "What is the primary meaning of {}?",
    "Which statement about {} is correct?",
    "What does {} refer to in the context of {}?"
]

def generate_mcqs(chunks, topic, subject, n):
    mcqs = []
    seen = set()
    for c in chunks:
        key_phrase = topic
        for pattern in MCQ_PATTERNS:
            q = pattern.format(key_phrase, subject)
            if q not in seen:
                options = ["Option A", "Option B", "Option C", "Option D"]
                answer = random.randint(0, 3)
                mcqs.append({"question": q, "options": options, "answer": answer})
                seen.add(q)
            if len(mcqs) >= n:
                return mcqs
    # fallback
    while len(mcqs) < n:
        q = f"What is {topic}? (Answer with reason)"
        mcqs.append({"question": q, "options": ["A", "B", "C", "D"], "answer": 0})
    return mcqs

# =========================
# UI
# =========================
with st.sidebar:
    st.header("NCERT Content")
    if st.button("Download & Load NCERT ZIP"):
        download_ncert_zip()
        extract_zip()
        st.success("NCERT content extracted!")

subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Chapter / Topic (e.g., Constitution)")
num_q = st.number_input("Number of questions", 1, 20, 5)
q_type = st.radio("Question type", ["Subjective", "MCQ"])

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Enter topic first")
        st.stop()

    all_texts = load_all_texts()
    if not all_texts:
        st.error("No readable NCERT text found after extraction.")
        st.stop()

    chunks = []
    for t in all_texts:
        chunks.extend(chunk_text(t))

    relevant = relevant_chunks(chunks, topic, subject)

    if q_type == "Subjective":
        questions = generate_subjective_questions(relevant, topic, subject, num_q)
        st.success(f"Generated {len(questions)} subjective questions")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
    else:
        mcqs = generate_mcqs(relevant, topic, subject, num_q)
        st.success(f"Generated {len(mcqs)} MCQs")
        for i, m in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {m['question']}**")
            for idx, opt in enumerate(m["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + m['answer'])}")
            st.write("---")
