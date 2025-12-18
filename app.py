# NCERT Question Generator with Subjective & MCQs (NCERT + UPSC)
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
    "Polity": ["constitution", "federal", "parliament", "judiciary", "union", "state"],
    "Sociology": ["society", "caste", "class", "gender"],
    "Psychology": ["behaviour", "learning", "emotion"],
    "Business Studies": ["management", "planning", "marketing"],
    "Economics": ["economy", "gdp", "growth", "inflation"]
}

# =========================
# MCQ TEMPLATES
# =========================

NCERT_MCQ_TEMPLATES = [
    {
        "q": "What is meant by {c}?",
        "options": [
            "A system of government with division of powers",
            "A unitary form of administration",
            "A judicial principle",
            "An economic arrangement"
        ],
        "answer": 0
    },
    {
        "q": "Which of the following best describes {c}?",
        "options": [
            "Power shared between Centre and States",
            "Power concentrated at Centre",
            "Power exercised by judiciary",
            "Power exercised by military"
        ],
        "answer": 0
    }
]

UPSC_MCQ_TEMPLATES = [
    {
        "q": "With reference to {c}, consider the following statements:",
        "statements": [
            "The Constitution divides powers between different levels of government.",
            "The Centre has overriding powers in certain situations."
        ],
        "options": [
            "1 only",
            "2 only",
            "Both 1 and 2",
            "Neither 1 nor 2"
        ],
        "answer": 2
    }
]

ASSERTION_REASON = [
    {
        "assertion": "India is described as a quasi-federal State.",
        "reason": "The Constitution provides for a strong Centre with residuary powers.",
        "answer": 0
    }
]

ASSERTION_OPTIONS = [
    "Both A and R are true and R is the correct explanation of A",
    "Both A and R are true but R is not the correct explanation of A",
    "A is true but R is false",
    "A is false but R is true"
]

SUBJECTIVE_NCERT = [
    "Explain the concept of {c}.",
    "Describe the main features of {c}."
]

SUBJECTIVE_UPSC = [
    "Discuss the significance of {c} in the Indian constitutional system.",
    "Examine the challenges associated with {c}."
]

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT–UPSC Question Generator", layout="wide")
st.title("📘 NCERT–UPSC Question Generator (Class XI–XII)")

# =========================
# UTILITIES
# =========================
def download_zip():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    for root, _, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.endswith(".zip"):
                p = os.path.join(root, f)
                with zipfile.ZipFile(p, "r") as nz:
                    nz.extractall(os.path.join(root, Path(f).stem))

def read_pdf(path):
    try:
        r = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in r.pages)
    except:
        return ""

def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def semantic_chunk(text):
    s = re.split(r'(?<=[.?!])\s+', text)
    return [" ".join(s[i:i+4]) for i in range(0, len(s), 4) if len(s[i:i+4]) > 2]

def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [c for c in chunks if topic in c.lower() or any(k in c.lower() for k in keys)]

def load_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

# =========================
# GENERATORS
# =========================
def generate_subjective(topic, n, level):
    patterns = SUBJECTIVE_NCERT if level == "NCERT Level" else SUBJECTIVE_UPSC
    return [random.choice(patterns).format(c=topic) for _ in range(n)]

def generate_mcqs(topic, n, level, difficulty):
    mcqs = []

    if level == "NCERT Level":
        templates = NCERT_MCQ_TEMPLATES

    else:
        if difficulty == "Hard":
            templates = ASSERTION_REASON
        else:
            templates = UPSC_MCQ_TEMPLATES

    for t in templates[:n]:
        if "assertion" in t:
            q = f"Assertion (A): {t['assertion']}\nReason (R): {t['reason']}"
            mcqs.append({"question": q, "options": ASSERTION_OPTIONS, "answer": t["answer"]})

        elif "statements" in t:
            q = t["q"].format(c=topic) + "\n" + "\n".join(
                [f"{i+1}. {s}" for i, s in enumerate(t["statements"])]
            )
            mcqs.append({"question": q, "options": t["options"], "answer": t["answer"]})

        else:
            mcqs.append({
                "question": t["q"].format(c=topic),
                "options": t["options"],
                "answer": t["answer"]
            })

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
# UI
# =========================
subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic / Chapter")
num_q = st.slider("Number of Questions", 1, 10, 5)
level = st.radio("Level", ["NCERT Level", "UPSC Level"])
difficulty = st.select_slider("Difficulty", ["Easy", "Moderate", "Hard"], value="Moderate")

tab1, tab2 = st.tabs(["Subjective", "MCQs"])

texts = load_texts()
chunks = []
for t in texts:
    chunks.extend(semantic_chunk(t))
relevant = boolean_filter(chunks, topic, subject)

# =========================
# SUBJECTIVE
# =========================
with tab1:
    if st.button("Generate Subjective"):
        qs = generate_subjective(topic, num_q, level)
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

# =========================
# MCQs
# =========================
with tab2:
    if st.button("Generate MCQs"):
        mcqs = generate_mcqs(topic, num_q, level, difficulty)
        for i, m in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {m['question']}**")
            for j, o in enumerate(m["options"]):
                st.write(f"{chr(97+j)}) {o}")
            st.write(f"✅ Answer: {chr(97 + m['answer'])}")
            st.write("---")
