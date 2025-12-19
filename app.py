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
    "Polity", "Economics", "Sociology",
    "Psychology", "Business Studies"
]

# ------------------------------------------
# STREAMLIT SETUP
# ------------------------------------------
st.set_page_config(
    page_title="NCERT & UPSC Generator",
    layout="wide"
)
st.title("📘 NCERT + UPSC Exam-Ready Question Generator")

# ------------------------------------------
# UTILITIES
# ------------------------------------------
def download_and_extract():
    if not Path(ZIP_PATH).exists():
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False
        )

    if not Path(EXTRACT_DIR).exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)


def find_all_pdfs(base_dir):
    return list(Path(base_dir).rglob("*.pdf"))


def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""


def clean_text(text):
    text = re.sub(
        r"(activity|let us|exercise|project|editor|reprint|isbn).*",
        " ",
        text,
        flags=re.I
    )
    return re.sub(r"\s+", " ", text).strip()


def semantic_chunks(text, size=3):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [
        " ".join(sentences[i:i+size])
        for i in range(0, len(sentences), size)
    ]


def load_texts_and_chunks():
    texts, chunks = [], []

    pdfs = find_all_pdfs(EXTRACT_DIR)
    st.write(f"📄 PDFs detected: {len(pdfs)}")

    if not pdfs:
        return texts, chunks

    for pdf in pdfs:
        text = clean_text(read_pdf(str(pdf)))
        if len(text.split()) >= 100:
            texts.append(text)
            chunks.extend(semantic_chunks(text))

    st.write(f"🧩 Chunks created: {len(chunks)}")
    return texts, chunks

# ------------------------------------------
# TOPIC FILTERING
# ------------------------------------------
def is_topic_relevant(text, topic):
    words = topic.lower().split()
    return any(w in text.lower() for w in words)


def get_relevant_chunks(chunks, topic):
    return [
        ch for ch in chunks
        if is_topic_relevant(ch, topic)
        and not any(x in ch.lower() for x in
                    ["exercise", "activity", "table", "figure"])
    ]


def count_possible_mcqs(chunks):
    count = 0
    for ch in chunks:
        sents = [s for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        if sents:
            count += 1
    return count

# ------------------------------------------
# HIGHLIGHT KEYWORDS
# ------------------------------------------
def highlight(sentence):
    keywords = [
        "constitution", "democracy",
        "freedom", "equality", "rights", "india"
    ]
    for k in keywords:
        sentence = re.sub(
            fr"\b({k})\b",
            r"**\1**",
            sentence,
            flags=re.I
        )
    return sentence

# ------------------------------------------
# NCERT MCQs
# ------------------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    used = set()

    for ch in chunks:
        sentences = [
            s.strip() for s in re.split(r'[.;]', ch)
            if len(s.split()) >= 10
        ]

        if not sentences:
            continue

        correct = sentences[0]
        if correct in used:
            continue

        distractors = []
        for other in chunks:
            for s in re.split(r'[.;]', other):
                if (
                    len(s.split()) >= 8
                    and s != correct
                    and is_topic_relevant(s, topic)
                ):
                    distractors.append(s.strip())

        if len(distractors) < 3:
            continue

        used.add(correct)

        options = [highlight(correct)] + distractors[:3]
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best describes **{topic}**?",
            "options": options,
            "answer": options.index(highlight(correct))
        })

        if len(mcqs) >= n:
            break

    return mcqs

# ------------------------------------------
# UPSC PATTERNS
# ------------------------------------------
def generate_upsc_statements(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "statements": [
                f"{topic} is a core value of the Indian Constitution.",
                f"{topic} guides interpretation of fundamental rights.",
                f"{topic} is enforceable only during emergencies."
            ],
            "answer": "1 and 2"
        })
    return qs


def generate_assertion_reason(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "A": f"{topic} is integral to the Indian Constitution.",
            "R": f"It reflects the philosophy and objectives of the Constitution.",
            "answer": "a"
        })
    return qs

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
with st.sidebar:
    st.header("Settings")

    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        texts, chunks = load_texts_and_chunks()
        st.session_state.texts = texts
        st.session_state.chunks = chunks
        st.success("NCERT content loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Democracy, Preamble)")
num_q = st.number_input("Number of Questions", 1, 20, 5)

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
            st.error("Enter a topic")
        else:
            rel = get_relevant_chunks(chunks, topic)
            max_q = count_possible_mcqs(rel)

            if max_q == 0:
                st.error("No meaningful NCERT content found.")
            else:
                st.info(f"{max_q} meaningful questions possible.")
                for i in range(min(num_q, max_q)):
                    st.write(f"{i+1}. Explain {topic} with reference to NCERT.")

# ------------------------------------------
# MCQs
# ------------------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC Statements", "Assertion-Reason"]
    )

    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Enter a topic")
        else:
            rel = get_relevant_chunks(chunks, topic)
            max_q = count_possible_mcqs(rel)

            if max_q == 0:
                st.error("❌ No meaningful NCERT content found.")
            else:
                final_n = min(num_q, max_q)
                st.info(f"{max_q} MCQs possible. Showing {final_n}")

                if mcq_type == "NCERT MCQs":
                    qs = generate_ncert_mcqs(rel, topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. {q['q']}**")
                        for j, opt in enumerate(q["options"]):
                            st.write(f"{chr(97+j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97+q['answer'])}")
                        st.write("---")

                elif mcq_type == "UPSC Statements":
                    qs = generate_upsc_statements(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Consider the following statements:**")
                        for j, s in enumerate(q["statements"], 1):
                            st.write(f"{j}. {s}")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

                else:
                    qs = generate_assertion_reason(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Assertion:** {q['A']}")
                        st.write(f"**Reason:** {q['R']}")
                        st.write("a) Both A and R are true and R explains A")
                        st.write("b) Both true but R does not explain A")
                        st.write("c) A true, R false")
                        st.write("d) A false, R true")
                        st.write("✅ Answer: a")
                        st.write("---")
