# NCERT & UPSC Context-aware MCQ Generator
import os, zipfile, re, random
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
    "Polity": ["constitution", "writ", "rights", "judiciary", "parliament", "federalism", "emergency"],
    "Sociology": ["society", "social", "caste", "class", "gender", "movement"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "planning", "organising", "leadership", "marketing"],
    "Economics": ["economy", "growth", "gdp", "poverty", "inflation", "development"]
}

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT & UPSC MCQ Generator", layout="wide")
st.title("📘 NCERT & UPSC MCQ Generator")

# =========================
# UTILITIES
# =========================
def download_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        extract_nested_zips(EXTRACT_DIR)

def extract_nested_zips(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_dir = os.path.join(root, Path(file).stem)
                os.makedirs(nested_extract_dir, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, "r") as nz:
                    nz.extractall(nested_extract_dir)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# SEMANTIC CHUNKING
# =========================
def semantic_chunk(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i:i+4])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks

# =========================
# BOOLEAN FILTER
# =========================
def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [c for c in chunks if topic in c.lower() or any(k in c.lower() for k in keys)]

# =========================
# VALIDATION
# =========================
def is_valid_question(q):
    return len(q.split()) > 5

# =========================
# MCQ GENERATION
# =========================
import random
import re

import random
import re

import random
import re

def generate_mcqs_upsc(chunks, topic, n, level, subject):
    """
    Generate UPSC or NCERT standard MCQs from text.
    - chunks: semantically chunked text
    - topic: topic string
    - n: number of questions
    - level: "NCERT Level" or "UPSC Level"
    - subject: subject context
    """
    mcqs = []
    used_questions = set()
    topic_lower = topic.lower()
    keywords = [k.lower() for k in SUBJECT_KEYWORDS.get(subject, [])]

    random.shuffle(chunks)

    for chunk in chunks:
        # sentences containing topic or keywords
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', chunk)
                     if topic_lower in s.lower() or any(k in s.lower() for k in keywords)]
        if not sentences:
            continue

        for base_sentence in sentences:
            if len(mcqs) >= n:
                break

            # Create NCERT-style MCQ
            if level == "NCERT Level":
                question = f"Which of the following statements about {topic} is correct?"
                answer = base_sentence
                distractors = [s for s in sentences if s != answer]
                distractors = random.sample(distractors, min(3, len(distractors)))
                # Fill remaining distractors if less than 3
                while len(distractors) < 3:
                    distractors.append("This statement is incorrect.")
                options = [answer] + distractors
                random.shuffle(options)
                correct_index = options.index(answer)
                mcqs.append({"question": question, "options": options, "answer": correct_index})

            # Create UPSC-style MCQ
            elif level == "UPSC Level":
                question = f"Consider the following statements about {topic} and select the correct option(s):"
                correct_sentences = random.sample(sentences, min(2, len(sentences)))
                potential_distractors = [s for s in sentences if s not in correct_sentences]
                distractors = random.sample(potential_distractors, min(2, len(potential_distractors)))
                while len(distractors) < 2:
                    distractors.append("Incorrect statement based on context.")
                options = correct_sentences + distractors
                random.shuffle(options)
                correct_indexes = [options.index(c) for c in correct_sentences]
                mcqs.append({"question": question, "options": options, "answer": correct_indexes})

            if len(mcqs) >= n:
                break

        if len(mcqs) >= n:
            break

    # Fallback simple question if not enough
    while len(mcqs) < n:
        mcqs.append({
            "question": f"What is {topic}?",
            "options": ["It is important", "Not relevant", "Unrelated", "Random fact"],
            "answer": 0
        })

    return mcqs



# =========================
# LOAD TEXTS
# =========================
def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if st.button("📥 Load NCERT Content"):
        download_zip()
        extract_zip()
        st.success("NCERT content loaded")

# =========================
# UI INPUT
# =========================
subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic / Chapter (e.g. Constitution, Federalism)")
num_q = st.number_input("Number of Questions", 1, 10, 5)
level = st.radio("Select Level", ["NCERT Level", "UPSC Level"])

tab1, tab2 = st.tabs(["Subjective Questions", "MCQs"])

# =========================
# GENERATE MCQs
# =========================
with tab2:
    if st.button("Generate MCQs"):
        if not topic.strip():
            st.warning("Enter a topic")
            st.stop()

        texts = load_all_texts()
        if not texts:
            st.error("No readable NCERT PDFs found")
            st.stop()

        chunks = []
        for t in texts:
            chunks.extend(semantic_chunk(t))

        relevant = boolean_filter(chunks, topic, subject)
        if len(relevant) < 5:
            relevant = chunks[:15]

        mcqs = generate_mcqs_upsc(relevant, topic, num_q, level, subject)

        st.success(f"Generated {len(mcqs)} MCQs ({level})")

        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")

            if isinstance(mcq['answer'], list):  # UPSC: multiple answers
                answers = ", ".join([chr(97 + a) for a in mcq['answer']])
                st.write(f"✅ **Answer(s):** {answers}")
            else:  # NCERT: single answer
                st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            
            st.write("---")
