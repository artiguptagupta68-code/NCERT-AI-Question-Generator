# ==========================================
# NCERT + UPSC Exam-Ready Question Generator
# ==========================================

import os
import zipfile
import re
import random
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader

# ------------------------------------------
# CONFIG
# ------------------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = [
    "Polity",
    "Economics",
    "Sociology",
    "Psychology",
    "Business Studies",
]

# ------------------------------------------
# STREAMLIT SETUP
# ------------------------------------------
st.set_page_config(
    page_title="NCERT & UPSC Generator",
    layout="wide",
)

st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# ------------------------------------------
# UTILITIES
# ------------------------------------------
def download_and_extract():
    if not Path(ZIP_PATH).exists():
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False,
        )

    if not Path(EXTRACT_DIR).exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)


def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def clean_text(text: str) -> str:
    text = re.sub(
        r"(activity|exercise|project|table|figure|copyright|isbn).*",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------
# PDF ROOT DETECTION (CRITICAL FIX)
# ------------------------------------------
def find_pdf_root(base_dir: str):
    for path in Path(base_dir).rglob("*"):
        if path.is_dir() and list(path.glob("*.pdf")):
            return path
    return None


def semantic_chunks(text: str):
    sentences = re.split(r"(?<=[.])\s+", text)
    return [
        " ".join(sentences[i:i + 3])
        for i in range(0, len(sentences), 3)
        if len(" ".join(sentences[i:i + 3]).split()) >= 20
    ]


def load_texts_and_chunks():
    texts, chunks = [], []

    pdf_root = find_pdf_root(EXTRACT_DIR)
    if pdf_root is None:
        return texts, chunks, 0

    pdfs = list(pdf_root.rglob("*.pdf"))

    for pdf in pdfs:
        raw = read_pdf(str(pdf))
        cleaned = clean_text(raw)

        if len(cleaned.split()) >= 100:
            texts.append(cleaned)
            chunks.extend(semantic_chunks(cleaned))

    return texts, chunks, len(pdfs)


# ------------------------------------------
# TOPIC FILTERING
# ------------------------------------------
def get_relevant_chunks(chunks, topic):
    topic_words = topic.lower().split()
    relevant = []

    for ch in chunks:
        ch_low = ch.lower()
        if any(w in ch_low for w in topic_words):
            if not any(x in ch_low for x in ["activity", "exercise", "table", "figure"]):
                relevant.append(ch)

    return relevant


def count_possible_mcqs(chunks):
    count = 0
    for ch in chunks:
        sents = [s for s in re.split(r"[.;]", ch) if len(s.split()) >= 10]
        if sents:
            count += 1
    return count


# ------------------------------------------
# KEYWORD HIGHLIGHT
# ------------------------------------------
def highlight_keywords(sentence):
    keywords = [
        "constitution",
        "democracy",
        "freedom",
        "equality",
        "rights",
        "india",
    ]
    for k in keywords:
        sentence = re.sub(
            fr"\b({k})\b",
            r"**\1**",
            sentence,
            flags=re.I,
        )
    return sentence


# ------------------------------------------
# MCQ GENERATION
# ------------------------------------------
def get_distractors(chunks, correct, topic, k=3):
    pool = []
    for ch in chunks:
        for s in re.split(r"[.;]", ch):
            s = s.strip()
            if (
                len(s.split()) >= 8
                and s != correct
                and topic.lower() in s.lower()
            ):
                pool.append(s)

    random.shuffle(pool)
    return pool[:k]


def generate_ncert_mcqs(chunks, topic, n):
    mcqs, used = [], set()

    for ch in chunks:
        sents = [s.strip() for s in re.split(r"[.;]", ch) if len(s.split()) >= 10]
        if not sents:
            continue

        correct = sents[0]
        if correct in used:
            continue

        distractors = get_distractors(chunks, correct, topic)
        if len(distractors) < 2:
            continue

        used.add(correct)
        correct_h = highlight_keywords(correct)

        options = [correct_h] + distractors[:3]
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best describes **{topic}**?",
            "options": options,
            "answer": options.index(correct_h),
        })

        if len(mcqs) >= n:
            break

    return mcqs


# ------------------------------------------
# UPSC FORMATS
# ------------------------------------------
def generate_upsc_statements(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "statements": [
                f"{topic} reflects constitutional values.",
                f"{topic} influences democratic governance.",
                f"{topic} applies only during emergencies.",
            ],
            "answer": "1 and 2",
        })
    return qs


def generate_assertion_reason(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "A": f"{topic} is a foundational principle of the Indian Constitution.",
            "R": "It guides interpretation of rights and duties.",
            "answer": "a",
        })
    return qs


# ------------------------------------------
# SIDEBAR
# ------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        texts, chunks, pdf_count = load_texts_and_chunks()

        st.session_state.texts = texts
        st.session_state.chunks = chunks
        st.session_state.loaded = True

        if pdf_count == 0:
            st.error("No PDFs found after extraction.")
        else:
            st.success(f"Loaded {pdf_count} PDFs")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Democracy, Freedom)")
num_q = st.number_input("Number of Questions", 1, 20, 5)

# ------------------------------------------
# DATA ACCESS
# ------------------------------------------
texts = st.session_state.get("texts", [])
chunks = st.session_state.get("chunks", [])

# ------------------------------------------
# TABS
# ------------------------------------------
tab1, tab2 = st.tabs(["📝 Subjective", "🧠 MCQs"])

# ------------------------------------------
# SUBJECTIVE
# ------------------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.error("Enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            max_q = count_possible_mcqs(relevant)

            if max_q == 0:
                st.error("No meaningful NCERT content found.")
            else:
                final_n = min(num_q, max_q)
                st.info(f"{max_q} meaningful questions possible. Showing {final_n}")

                for i in range(1, final_n + 1):
                    st.write(f"{i}. Explain {topic} in the context of NCERT.")


# ------------------------------------------
# MCQs
# ------------------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC Statements", "Assertion–Reason"],
    )

    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            max_q = count_possible_mcqs(relevant)

            if max_q == 0:
                st.error("No meaningful NCERT content found.")
            else:
                final_n = min(num_q, max_q)
                st.info(f"{max_q} meaningful MCQs possible. Showing {final_n}")

                if mcq_type == "NCERT MCQs":
                    mcqs = generate_ncert_mcqs(relevant, topic, final_n)
                    for i, m in enumerate(mcqs, 1):
                        st.write(f"**Q{i}. {m['q']}**")
                        for j, opt in enumerate(m["options"]):
                            st.write(f"{chr(97 + j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                        st.write("---")

                elif mcq_type == "UPSC Statements":
                    qs = generate_upsc_statements(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Consider the following statements:**")
                        for idx, s in enumerate(q["statements"], 1):
                            st.write(f"{idx}. {s}")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

                else:
                    qs = generate_assertion_reason(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                        st.write(f"**Reason (R):** {q['R']}")
                        st.write("a) Both A and R are true and R explains A")
                        st.write("b) Both true but R does not explain A")
                        st.write("c) A true, R false")
                        st.write("d) A false, R true")
                        st.write("✅ Answer: a")
                        st.write("---")
