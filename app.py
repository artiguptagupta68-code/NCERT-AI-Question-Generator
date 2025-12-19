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
# SESSION STATE (CRITICAL)
# -------------------------------
if "texts" not in st.session_state:
    st.session_state.texts = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "loaded" not in st.session_state:
    st.session_state.loaded = False

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
    sents = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]

# -------------------------------
# LOAD PDFs (NO CACHE!)
# -------------------------------
def load_texts_and_chunks():
    texts, chunks = [], []

    pdfs = list(Path(EXTRACT_DIR).rglob("*.pdf"))
    for pdf in pdfs:
        raw = read_pdf(str(pdf))
        cleaned = clean_text(raw)
        if len(cleaned.split()) > 100:
            texts.append(cleaned)
            chunks.extend(semantic_chunks(cleaned))

    return texts, chunks, len(pdfs)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")

    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        texts, chunks, pdf_count = load_texts_and_chunks()

        st.session_state.texts = texts
        st.session_state.chunks = chunks
        st.session_state.loaded = True

        st.success(f"Loaded {pdf_count} PDFs")

# -------------------------------
# STATUS DISPLAY (DEBUG)
# -------------------------------
st.caption(
    f"📄 PDFs loaded: {len(st.session_state.texts)} | "
    f"🧩 Chunks extracted: {len(st.session_state.chunks)}"
)

# -------------------------------
# INPUTS
# -------------------------------
subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Democracy, Preamble, Freedom)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# SAFETY CHECK
# -------------------------------
if not st.session_state.loaded:
    st.warning("Please click **Load NCERT PDFs** first.")
    st.stop()

# -------------------------------
# TOPIC FILTER
# -------------------------------
def is_relevant(text, topic):
    return any(w in text.lower() for w in topic.lower().split())

relevant_chunks = [
    ch for ch in st.session_state.chunks if is_relevant(ch, topic)
]

max_possible = len(relevant_chunks)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📝 Subjective", "🧠 MCQs"])

# -------------------------------
# SUBJECTIVE
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if max_possible == 0:
            st.error("No meaningful NCERT content found.")
        else:
            final_n = min(num_q, max_possible)
            st.info(f"{max_possible} possible questions. Showing {final_n}.")
            for i in range(final_n):
                st.write(f"{i+1}. Explain the concept of {topic}.")

# -------------------------------
# MCQs
# -------------------------------
with tab2:
    if st.button("Generate MCQs"):
        if max_possible == 0:
            st.error("No meaningful NCERT content found.")
        else:
            final_n = min(num_q, max_possible)
            st.info(f"{max_possible} possible MCQs. Showing {final_n}.")

            for i, ch in enumerate(relevant_chunks[:final_n], 1):
                st.write(f"**Q{i}. Which of the following best describes {topic}?**")
                opts = [
                    ch,
                    "It applies only during emergencies",
                    "It is a temporary arrangement",
                    "It deals only with economic policy",
                ]
                random.shuffle(opts)
                for j, o in enumerate(opts):
                    st.write(f"{chr(97+j)}) {o}")
                st.write("✅ Answer: a")
                st.write("---")
