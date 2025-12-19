# ===============================
# NCERT + UPSC Exam-Ready Generator
# ===============================

import os, zipfile, re, random
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader

# -------------------------------
# CONFIG (DO NOT CHANGE)
# -------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# -------------------------------
# FILE HANDLING
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

# -------------------------------
# PDF READING (STREAMLIT SAFE)
# -------------------------------
def read_pdf(path):
    text = ""
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            if p.extract_text():
                text += " " + p.extract_text()
    except Exception:
        pass
    return text

def clean_text(text):
    text = re.sub(
        r"(activity|let us|exercise|project|table|figure|copyright|isbn).*",
        " ",
        text,
        flags=re.I
    )
    return re.sub(r"\s+", " ", text).strip()

def load_all_texts():
    texts = []
    pdfs = list(Path(EXTRACT_DIR).rglob("*.pdf"))

    for pdf in pdfs:
        raw = read_pdf(str(pdf))
        cleaned = clean_text(raw)
        if len(cleaned.split()) > 100:
            texts.append(cleaned)

    return texts

# -------------------------------
# CHUNKING
# -------------------------------
def semantic_chunks(text):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [
        " ".join(sentences[i:i+3])
        for i in range(0, len(sentences), 3)
        if len(" ".join(sentences[i:i+3]).split()) >= 20
    ]

# -------------------------------
# CONCEPTUAL TOPIC MATCHING (FIX)
# -------------------------------
def get_relevant_chunks(chunks, topic):
    topic_words = topic.lower().split()
    relevant = []

    for ch in chunks:
        cl = ch.lower()

        # Remove junk
        if any(x in cl for x in ["exercise", "activity", "table", "figure"]):
            continue

        score = sum(1 for w in topic_words if w in cl)

        # Accept conceptually relevant NCERT text
        if score >= 1 or any(
            k in cl for k in [
                "government",
                "constitution",
                "state",
                "citizens",
                "rights",
                "people",
                "elections",
                "law"
            ]
        ):
            relevant.append(ch)

    return relevant

# -------------------------------
# QUESTION CAPACITY
# -------------------------------
def count_possible_mcqs(chunks):
    return len(chunks)

# -------------------------------
# KEYWORD HIGHLIGHT
# -------------------------------
def highlight_keywords(sentence):
    keywords = ["constitution", "freedom", "democracy", "rights", "equality", "india"]
    for k in keywords:
        sentence = re.sub(fr"\b({k})\b", r"**\1**", sentence, flags=re.I)
    return sentence

# -------------------------------
# DYNAMIC DISTRACTORS
# -------------------------------
def get_dynamic_distractors(chunks, correct, topic, k=3):
    pool = []
    for ch in chunks:
        for s in re.split(r'[.;]', ch):
            s = s.strip()
            if (
                len(s.split()) >= 10
                and s != correct
            ):
                pool.append(s)
    random.shuffle(pool)
    return pool[:k]

# -------------------------------
# SUBJECTIVE QUESTIONS
# -------------------------------
def generate_subjective(topic, n):
    templates = [
        f"Explain the concept of {topic}.",
        f"Discuss the significance of {topic}.",
        f"Describe the main features of {topic}.",
        f"Why is {topic} important in a democracy?",
        f"Examine the role of {topic} in the Indian Constitution.",
    ]
    return templates[:n]

# -------------------------------
# NCERT MCQs
# -------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    used = set()

    for ch in chunks:
        sentences = [s.strip() for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        if not sentences:
            continue

        correct = sentences[0]
        if correct in used:
            continue

        distractors = get_dynamic_distractors(chunks, correct, topic)
        if len(distractors) < 3:
            continue

        used.add(correct)
        options = [highlight_keywords(correct)] + distractors[:3]
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best describes **{topic}**?",
            "options": options,
            "answer": options.index(highlight_keywords(correct))
        })

        if len(mcqs) >= n:
            break

    return mcqs

# -------------------------------
# UPSC TYPES
# -------------------------------
def generate_upsc_statements(topic, n):
    return [{
        "statements": [
            f"{topic} reflects constitutional values.",
            f"{topic} guides democratic governance.",
            f"{topic} is enforceable only during emergencies."
        ],
        "answer": "1 and 2"
    } for _ in range(n)]

def generate_assertion_reason(topic, n):
    return [{
        "A": f"{topic} is a basic feature of the Indian Constitution.",
        "R": "It shapes the philosophy of governance.",
        "answer": "a"
    } for _ in range(n)]

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Democracy, Freedom, Equality)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD DATA
# -------------------------------
texts, chunks = [], []

if os.path.exists(EXTRACT_DIR):
    texts = load_all_texts()
    for t in texts:
        chunks.extend(semantic_chunks(t))

# DEBUG (SAFE TO KEEP)
st.caption(f"📄 PDFs loaded: {len(texts)} | 🧩 Chunks extracted: {len(chunks)}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📝 Subjective", "🧠 MCQs"])

# -------------------------------
# SUBJECTIVE TAB
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.error("Enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            max_q = count_possible_mcqs(relevant)
            final_n = min(num_q, max_q)

            if max_q == 0:
                st.error("No meaningful NCERT content found.")
            else:
                st.info(f"{max_q} questions possible. Showing {final_n}.")
                for i, q in enumerate(generate_subjective(topic, final_n), 1):
                    st.write(f"{i}. {q}")

# -------------------------------
# MCQ TAB
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"]
    )

    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            max_q = count_possible_mcqs(relevant)
            final_n = min(num_q, max_q)

            if max_q == 0:
                st.error("❌ No meaningful NCERT content found.")
            else:
                st.info(f"{max_q} MCQs possible. Showing {final_n}.")

                if mcq_type == "NCERT MCQs":
                    for i, m in enumerate(generate_ncert_mcqs(relevant, topic, final_n), 1):
                        st.write(f"**Q{i}. {m['q']}**")
                        for j, opt in enumerate(m["options"]):
                            st.write(f"{chr(97+j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                        st.write("---")

                elif mcq_type == "UPSC PYQ – Statements":
                    for i, q in enumerate(generate_upsc_statements(topic, final_n), 1):
                        st.write(f"**Q{i}. With reference to {topic}:**")
                        for j, s in enumerate(q["statements"], 1):
                            st.write(f"{j}. {s}")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

                else:
                    for i, q in enumerate(generate_assertion_reason(topic, final_n), 1):
                        st.write(f"**Assertion:** {q['A']}")
                        st.write(f"**Reason:** {q['R']}")
                        st.write("a) Both true, R explains A")
                        st.write("b) Both true, R not explanation")
                        st.write("c) A true, R false")
                        st.write("d) A false, R true")
                        st.write("✅ Answer: a")
                        st.write("---")
