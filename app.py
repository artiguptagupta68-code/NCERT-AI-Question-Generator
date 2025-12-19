# ==========================================
# NCERT / UPSC Question Generator – FIXED
# ==========================================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# ---------------- CONFIG ----------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert_books.zip"
EXTRACT_DIR = "ncert_data"

QUESTIONS_PER_CHUNK = 2  # SAFE NCERT LIMIT

# ---------------- UI ----------------
st.set_page_config(page_title="NCERT / UPSC Question Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# ---------------- UTILITIES ----------------
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
        r"(activity|exercise|project|let us|table|figure|map|editor|isbn|reprint).*",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        txt = clean_text(read_pdf(str(pdf)))
        if len(txt.split()) > 300:
            texts.append(txt)
    return texts

def make_chunks(text):
    sents = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]

# ---------------- CHUNK VALIDATION ----------------
def is_valid_chunk(chunk, topic):
    bad = ["activity", "debate", "project", "exercise", "table", "figure"]
    if topic.lower() not in chunk.lower():
        return False
    if len(chunk.split()) < 30 or len(chunk.split()) > 140:
        return False
    return not any(b in chunk.lower() for b in bad)

# ---------------- QUESTION BUDGET ----------------
def compute_question_budget(chunks):
    return len(chunks) * QUESTIONS_PER_CHUNK

# ---------------- MCQ GENERATION ----------------
def generate_ncert_mcqs(chunks, topic, required):
    mcqs = []
    used_chunks = set()

    for chunk in chunks:
        if chunk in used_chunks:
            continue

        sentences = [s.strip() for s in re.split(r'[.;]', chunk) if len(s.split()) > 10]

        if not sentences:
            continue

        correct = sentences[0]

        distractors_pool = []
        for other in chunks:
            if other == chunk:
                continue
            for s in re.split(r'[.;]', other):
                s = s.strip()
                if (
                    len(s.split()) > 8
                    and topic.lower() in s.lower()
                    and s != correct
                ):
                    distractors_pool.append(s)

        if len(distractors_pool) < 3:
            continue

        distractors = random.sample(distractors_pool, 3)

        options = [correct] + distractors
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best explains **{topic}**?",
            "options": options,
            "answer": options.index(correct)
        })

        used_chunks.add(chunk)

        if len(mcqs) >= required:
            break

    return mcqs

# ---------------- SUBJECTIVE ----------------
def generate_subjective_questions(chunks, topic, required):
    qs = []
    for chunk in chunks:
        if len(qs) >= required:
            break
        qs.append(f"Explain the concept of **{topic}** with reference to the Constitution.")
        qs.append(f"Discuss the significance of **{topic}** in a democratic system.")
    return qs[:required]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

topic = st.text_input("Enter Topic (e.g. Democracy, Preamble)")
num_q = st.number_input("Number of Questions", 1, 50, 10)

# ---------------- MAIN ----------------
texts = load_all_texts()
all_chunks = []
for t in texts:
    all_chunks.extend(make_chunks(t))

relevant_chunks = [c for c in all_chunks if is_valid_chunk(c, topic)]

max_possible = compute_question_budget(relevant_chunks)

tab1, tab2 = st.tabs(["📝 Subjective", "🧠 MCQs"])

# ---------------- SUBJECTIVE TAB ----------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if not relevant_chunks:
            st.error("No relevant NCERT content found.")
        else:
            final_n = min(num_q, max_possible)
            st.info(f"{max_possible} quality questions possible. Showing {final_n}.")
            qs = generate_subjective_questions(relevant_chunks, topic, final_n)
            for i, q in enumerate(qs, 1):
                st.write(f"{i}. {q}")

# ---------------- MCQ TAB ----------------
with tab2:
    if st.button("Generate MCQs"):
        if not relevant_chunks:
            st.error("No relevant NCERT content found.")
        else:
            final_n = min(num_q, max_possible)
            st.info(f"{max_possible} quality MCQs possible. Showing {final_n}.")
            mcqs = generate_ncert_mcqs(relevant_chunks, topic, final_n)

            if not mcqs:
                st.warning("Not enough clean content to form MCQs.")
            else:
                for i, m in enumerate(mcqs, 1):
                    st.write(f"**Q{i}. {m['q']}**")
                    for j, opt in enumerate(m["options"]):
                        st.write(f"{chr(97+j)}) {opt}")
                    st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                    st.write("---")
