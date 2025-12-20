# ============================================
# NCERT + UPSC Exam-Ready Generator (RAG-based)
# ============================================

import os, zipfile, re, random
from pathlib import Path

import streamlit as st
import gdown
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# CONFIG
# --------------------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45

# --------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# --------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --------------------------------------------
# UTILITIES
# --------------------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # extract main zip
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    # extract nested zips
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except:
            pass


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
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts


def semantic_chunks(text):
    sentences = re.split(r"(?<=[.])\s+", text)
    return [
        " ".join(sentences[i : i + 3])
        for i in range(0, len(sentences), 3)
        if len(sentences[i : i + 3]) > 0
    ]


def is_conceptual(sentence):
    s = sentence.lower()
    skip = ["chapter", "unit", "page", "contents", "glossary", "figure", "table"]
    return not any(k in s for k in skip) and 8 <= len(s.split()) <= 60


# --------------------------------------------
# EMBEDDINGS & RETRIEVAL
# --------------------------------------------
@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)


def retrieve_relevant_chunks(chunks, embeddings, topic, standard="NCERT", top_k=20):
    topic_vec = embedder.encode([topic], convert_to_numpy=True)
    sims = cosine_similarity(topic_vec, embeddings)[0]

    threshold = (
        SIMILARITY_THRESHOLD_UPSC
        if standard == "UPSC"
        else SIMILARITY_THRESHOLD_NCERT
    )

    ranked = sorted(
        zip(chunks, sims),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for ch, score in ranked:
        if score >= threshold and is_conceptual(ch):
            results.append(ch)
        if len(results) >= top_k:
            break

    return results


def count_possible_questions(chunks):
    return len(
        [
            s
            for ch in chunks
            for s in re.split(r"[.;]", ch)
            if is_conceptual(s)
        ]
    )


# --------------------------------------------
# SUBJECTIVE QUESTIONS
# --------------------------------------------
def generate_subjective(topic, n, standard="NCERT"):
    if standard == "NCERT":
        templates = [
            f"Explain the concept of {topic}.",
            f"Describe the main features of {topic}.",
            f"Write a short note on {topic}.",
            f"Why is {topic} important in the Indian context?",
            f"Explain {topic} with suitable examples.",
        ]
    else:
        templates = [
            f"Analyse the constitutional significance of {topic}.",
            f"Discuss the role of {topic} in India’s governance framework.",
            f"Critically examine the relevance of {topic} in contemporary India.",
            f"Evaluate the functioning of {topic} with suitable illustrations.",
            f"Discuss challenges associated with {topic} in India.",
        ]

    return templates[:n]


# --------------------------------------------
# MCQ GENERATORS
# --------------------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    used = set()

    sentences = [
        s.strip()
        for ch in chunks
        for s in re.split(r"[.;]", ch)
        if is_conceptual(s)
    ]

    random.shuffle(sentences)

    for s in sentences:
        if s in used:
            continue
        used.add(s)

        distractors = [
            d
            for d in sentences
            if d != s and abs(len(d.split()) - len(s.split())) <= 5
        ]

        if len(distractors) < 2:
            continue

        options = [s] + random.sample(distractors, 3)
        random.shuffle(options)

        mcqs.append(
            {
                "q": f"Which of the following statements best describes **{topic}**?",
                "options": options,
                "answer": options.index(s),
            }
        )

        if len(mcqs) >= n:
            break

    return mcqs


def generate_assertion_reason(chunks, n):
    qs = []
    sentences = [
        s.strip()
        for ch in chunks
        for s in re.split(r"[.;]", ch)
        if is_conceptual(s)
    ]

    random.shuffle(sentences)

    for i in range(0, len(sentences) - 1, 2):
        qs.append(
            {
                "A": sentences[i],
                "R": sentences[i + 1],
                "answer": "a",
            }
        )
        if len(qs) >= n:
            break

    return qs


def generate_upsc_statements(topic, n):
    qs = []
    for _ in range(n):
        qs.append(
            {
                "statements": [
                    f"{topic} has constitutional backing in India.",
                    f"{topic} plays a role in strengthening democratic governance.",
                    f"{topic} is explicitly enforceable through courts.",
                ],
                "answer": "1 and 2",
            }
        )
    return qs


# --------------------------------------------
# SIDEBAR
# --------------------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded successfully")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Governor, Preamble, Equality)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# --------------------------------------------
# LOAD CONTENT
# --------------------------------------------
texts, chunks = [], []

if os.path.exists(EXTRACT_DIR):
    texts = load_all_texts()
    for t in texts:
        chunks.extend(semantic_chunks(t))

if chunks:
    chunk_embeddings = embed_chunks(chunks)
else:
    chunk_embeddings = []

st.write(f"📄 PDFs detected: {len(texts)}")
st.write(f"🧩 Total chunks extracted: {len(chunks)}")

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2 = st.tabs(["📝 Subjective", "🧠 MCQs (NCERT + UPSC)"])

# --------------------------------------------
# SUBJECTIVE TAB
# --------------------------------------------
with tab1:
    standard = st.radio("Question Standard", ["NCERT", "UPSC"])

    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            relevant_chunks = retrieve_relevant_chunks(
                chunks, chunk_embeddings, topic, standard=standard
            )

            max_possible = count_possible_questions(relevant_chunks)
            if max_possible == 0:
                st.error("❌ No meaningful NCERT content found for this topic.")
            else:
                final_n = min(num_q, max_possible)
                st.info(f"📊 {max_possible} possible. Showing {final_n}.")
                qs = generate_subjective(topic, final_n, standard)
                for i, q in enumerate(qs, 1):
                    st.write(f"{i}. {q}")

# --------------------------------------------
# MCQs TAB
# --------------------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"],
    )

    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            relevant_chunks = retrieve_relevant_chunks(
                chunks, chunk_embeddings, topic, standard="UPSC"
            )

            max_possible = count_possible_questions(relevant_chunks)
            if max_possible == 0:
                st.error("❌ No meaningful NCERT content found for this topic.")
            else:
                final_n = min(num_q, max_possible)
                st.info(f"📊 {max_possible} possible. Showing {final_n}.")

                if mcq_type == "NCERT MCQs":
                    mcqs = generate_ncert_mcqs(relevant_chunks, topic, final_n)
                    for i, m in enumerate(mcqs, 1):
                        st.write(f"**Q{i}. {m['q']}**")
                        for j, opt in enumerate(m["options"]):
                            st.write(f"{chr(97+j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                        st.write("---")

                elif mcq_type == "UPSC PYQ – Statements":
                    qs = generate_upsc_statements(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(
                            f"**Q{i}. With reference to {topic}, consider the following statements:**"
                        )
                        for idx, s in enumerate(q["statements"], 1):
                            st.write(f"{idx}. {s}")
                        st.write("Which of the statements given above are correct?")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

                else:
                    qs = generate_assertion_reason(relevant_chunks, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                        st.write(f"**Reason (R):** {q['R']}")
                        st.write(
                            "a) Both A and R are true and R is the correct explanation of A"
                        )
                        st.write(
                            "b) Both A and R are true but R is not the correct explanation of A"
                        )
                        st.write("c) A is true but R is false")
                        st.write("d) A is false but R is true")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")
