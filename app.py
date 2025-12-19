# ===============================
# NCERT + UPSC Exam-Ready Generator
# ===============================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# -------------------------------
# CONFIG (UNCHANGED)
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
# SESSION STATE
# -------------------------------
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False

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
        r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*",
        " ",
        text,
        flags=re.I
    )
    return re.sub(r"\s+", " ", text).strip()

def semantic_chunks(text):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

# -------------------------------
# CACHED PDF LOADER (CRITICAL FIX)
# -------------------------------
@st.cache_data(show_spinner="📖 Reading NCERT PDFs...")
def load_texts_and_chunks():
    texts, chunks = [], []

    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        raw = read_pdf(str(pdf))
        cleaned = clean_text(raw)

        if len(cleaned.split()) > 100:
            texts.append(cleaned)
            chunks.extend(semantic_chunks(cleaned))

    return texts, chunks

# -------------------------------
# TOPIC RELEVANCE
# -------------------------------
def is_topic_relevant(sentence, topic):
    return any(word in sentence.lower() for word in topic.lower().split())

def get_relevant_chunks(chunks, topic):
    good = []
    for ch in chunks:
        if is_topic_relevant(ch, topic):
            if not any(x in ch.lower() for x in ["activity", "exercise", "project", "table", "figure"]):
                good.append(ch)
    return good

# -------------------------------
# QUESTION CAPACITY CHECK
# -------------------------------
def count_possible_mcqs(chunks):
    return sum(
        1 for ch in chunks
        if any(len(s.split()) >= 10 for s in re.split(r"[.;]", ch))
    )

# -------------------------------
# KEYWORD HIGHLIGHT
# -------------------------------
def highlight_keywords(sentence):
    keywords = ["constitution", "democracy", "freedom", "rights", "equality", "india"]
    for k in keywords:
        sentence = re.sub(fr"\b({k})\b", r"**\1**", sentence, flags=re.I)
    return sentence

# -------------------------------
# DYNAMIC DISTRACTORS
# -------------------------------
def get_dynamic_distractors(chunks, correct, topic, k=3):
    pool = []
    for ch in chunks:
        for s in re.split(r"[.;]", ch):
            s = s.strip()
            if len(s.split()) >= 8 and s != correct and is_topic_relevant(s, topic):
                pool.append(s)
    random.shuffle(pool)
    return pool[:k]

# -------------------------------
# SUBJECTIVE QUESTIONS
# -------------------------------
def generate_subjective(topic, n):
    base = [
        f"Explain the concept of {topic}.",
        f"Discuss the significance of {topic}.",
        f"Describe the main features of {topic}.",
        f"Examine the role of {topic} in the Indian Constitution.",
        f"Why is {topic} important in a democracy?"
    ]
    return base[:n]

# -------------------------------
# NCERT MCQs (FINAL FIXED VERSION)
# -------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs, used = [], set()

    for ch in chunks:
        sentences = [s.strip() for s in re.split(r"[.;]", ch) if len(s.split()) >= 10]
        if not sentences:
            continue

        correct = sentences[0]
        if correct in used:
            continue

        distractors = get_dynamic_distractors(chunks, correct, topic)
        if len(distractors) < 2:
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
# UPSC QUESTIONS
# -------------------------------
def generate_upsc_statements(topic, n):
    return [{
        "statements": [
            f"{topic} reflects the ideals of the Indian Constitution.",
            f"{topic} guides the interpretation of constitutional provisions.",
            f"{topic} is enforceable by ordinary laws only."
        ],
        "answer": "1 and 2"
    } for _ in range(n)]

def generate_assertion_reason(topic, n):
    return [{
        "A": f"The {topic} is an integral part of the Indian Constitution.",
        "R": f"It embodies the philosophy and objectives of the Constitution.",
        "answer": "a"
    } for _ in range(n)]

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.session_state.pdfs_loaded = True
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Democracy, Preamble, Freedom)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD DATA (ONLY AFTER BUTTON)
# -------------------------------
texts, chunks = [], []
if st.session_state.pdfs_loaded and os.path.exists(EXTRACT_DIR):
    texts, chunks = load_texts_and_chunks()

st.caption(f"📄 PDFs loaded: {len(texts)} | 🧩 Chunks extracted: {len(chunks)}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📝 Subjective (NCERT)", "🧠 MCQs (NCERT + UPSC)"])

# -------------------------------
# SUBJECTIVE TAB
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            max_possible = count_possible_mcqs(relevant)

            if max_possible == 0:
                st.error("No meaningful NCERT content found.")
            else:
                final_n = min(num_q, max_possible)
                st.info(f"📊 {max_possible} meaningful questions possible. Showing {final_n}.")
                for i, q in enumerate(generate_subjective(topic, final_n), 1):
                    st.write(f"{i}. {q}")

# -------------------------------
# MCQs TAB
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"]
    )

    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            max_possible = count_possible_mcqs(relevant)

            if max_possible == 0:
                st.error("❌ No meaningful NCERT content found.")
            else:
                final_n = min(num_q, max_possible)
                st.info(f"📊 {max_possible} meaningful MCQs possible. Showing {final_n}.")

                if mcq_type == "NCERT MCQs":
                    mcqs = generate_ncert_mcqs(relevant, topic, final_n)
                    for i, m in enumerate(mcqs, 1):
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
                        st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                        st.write(f"**Reason (R):** {q['R']}")
                        st.write("a) Both A and R are true and R is the correct explanation of A")
                        st.write("b) Both A and R are true but R is not the correct explanation of A")
                        st.write("c) A is true but R is false")
                        st.write("d) A is false but R is true")
                        st.write("✅ Answer: a")
                        st.write("---")
