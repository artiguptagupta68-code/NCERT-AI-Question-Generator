# ============================================
# NCERT + UPSC Exam-Ready Question Generator
# ============================================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# -------------------------------
# CONFIG
# -------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert_books.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

KEYWORDS = [
    "freedom", "right", "constitution", "democracy",
    "justice", "equality", "liberty", "secular"
]

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# -------------------------------
# UTILITIES
# -------------------------------
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
    text = re.sub(
        r"(activity|let us|exercise|project|reprint|isbn).*",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 300:
            texts.append(t)
    return texts

def semantic_chunks(text):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]

# -------------------------------
# TEXT CLASSIFICATION
# -------------------------------
def classify_sentence(s):
    s = s.lower()
    if any(x in s for x in ["means", "refers to", "defined as"]):
        return "definition"
    if any(x in s for x in ["ensures", "protects", "guarantees"]):
        return "function"
    return "general"

# -------------------------------
# KEYWORD HIGHLIGHTER
# -------------------------------
def highlight_keywords(text):
    for k in KEYWORDS:
        text = re.sub(
            fr"\b({k})\b",
            r"**\1**",
            text,
            flags=re.I
        )
    return text

# -------------------------------
# DYNAMIC DISTRACTORS
# -------------------------------
def get_dynamic_distractors(chunks, topic, correct, k=3):
    pool = []
    for ch in chunks:
        for s in re.split(r'[.;]', ch):
            s = s.strip()
            if (
                len(s.split()) >= 8
                and topic.lower() not in s.lower()
                and s != correct
            ):
                pool.append(s)
    return random.sample(pool, min(k, len(pool)))

# -------------------------------
# NCERT MCQ GENERATOR
# -------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    used = set()

    for ch in chunks:
        for s in re.split(r'[.;]', ch):
            s = s.strip()
            if len(s.split()) < 10:
                continue
            if topic.lower() not in s.lower():
                continue
            if s in used:
                continue

            used.add(s)

            distractors = get_dynamic_distractors(chunks, topic, s)
            if len(distractors) < 2:
                continue

            options = [highlight_keywords(s)] + distractors
            random.shuffle(options)

            mcqs.append({
                "q": f"Which of the following best describes **{topic}**?",
                "options": options,
                "answer": options.index(highlight_keywords(s))
            })

            if len(mcqs) >= n:
                return mcqs

    return mcqs

# -------------------------------
# UPSC STATEMENT-BASED MCQs
# -------------------------------
def generate_upsc_statements(chunks, topic, n):
    qs = []
    for ch in chunks:
        sentences = [
            s.strip() for s in re.split(r'[.;]', ch)
            if topic.lower() in s.lower() and len(s.split()) > 8
        ]
        if len(sentences) >= 2:
            qs.append({
                "statements": sentences[:2],
                "answer": "1 and 2"
            })
        if len(qs) >= n:
            break
    return qs

# -------------------------------
# ASSERTION – REASON
# -------------------------------
def generate_assertion_reason(chunks, topic, n):
    qs = []
    for ch in chunks:
        sents = [
            s.strip() for s in re.split(r'[.;]', ch)
            if topic.lower() in s.lower() and len(s.split()) > 8
        ]
        if len(sents) >= 2:
            qs.append({
                "A": sents[0],
                "R": sents[1],
                "answer": "a"
            })
        if len(qs) >= n:
            break
    return qs

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Freedom, Preamble)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD DATA
# -------------------------------
texts = load_all_texts()
chunks = []
for t in texts:
    chunks.extend(semantic_chunks(t))

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["🧠 NCERT MCQs", "🏛 UPSC PYQ Style"])

# -------------------------------
# NCERT MCQs
# -------------------------------
with tab1:
    if st.button("Generate NCERT MCQs"):
        mcqs = generate_ncert_mcqs(chunks, topic, num_q)
        for i, m in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {m['q']}**")
            for j, opt in enumerate(m["options"]):
                st.write(f"{chr(97+j)}) {opt}")
            st.write(f"✅ Answer: {chr(97 + m['answer'])}")
            st.write("---")

# -------------------------------
# UPSC STYLE
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "UPSC Question Type",
        ["Statement Based", "Assertion Reason"]
    )

    if st.button("Generate UPSC MCQs"):
        if mcq_type == "Statement Based":
            qs = generate_upsc_statements(chunks, topic, num_q)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                for idx, s in enumerate(q["statements"], 1):
                    st.write(f"{idx}. {highlight_keywords(s)}")
                st.write("Which of the statements given above are correct?")
                st.write(f"✅ Answer: {q['answer']}")
                st.write("---")

        else:
            qs = generate_assertion_reason(chunks, topic, num_q)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}. Assertion (A):** {highlight_keywords(q['A'])}")
                st.write(f"**Reason (R):** {highlight_keywords(q['R'])}")
                st.write("a) Both A and R are true and R is the correct explanation of A")
                st.write("b) Both A and R are true but R is not the correct explanation of A")
                st.write("c) A is true but R is false")
                st.write("d) A is false but R is true")
                st.write("✅ Answer: a")
                st.write("---")
