# app.py
import os
import zipfile
from pathlib import Path
import re
import random

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

# =========================
# CONFIG
# =========================
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert_subjects.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = [
    "Polity",
    "Sociology",
    "Psychology",
    "Business Studies",
    "Economics"
]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "rights", "freedom", "emergency", "judiciary", "parliament"],
    "Sociology": ["society", "social", "movement", "caste", "class", "gender"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "leadership", "planning", "controlling", "marketing"],
    "Economics": ["economy", "gdp", "growth", "inflation", "poverty", "development"]
}

GENERIC_PATTERNS = [
    "Which of the following is true about {c}?",
    "The importance of {c} lies in:",
    "Identify the correct statement about {c}:",
    "Why is {c} significant in its context?"
]

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT UPSC MCQ Generator", layout="wide")
st.title("📘 NCERT UPSC-Style MCQ Generator")

# =========================
# UTILITIES
# =========================
def download_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
        return text
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def semantic_chunking(text, max_sentences=5):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        if len(chunk.split()) > 20:
            chunks.append(chunk)
    return chunks

def filter_chunks(chunks, topic, subject):
    filtered = []
    topic_lower = topic.lower()
    keywords = SUBJECT_KEYWORDS[subject]
    for c in chunks:
        c_lower = c.lower()
        if topic_lower in c_lower or any(k in c_lower for k in keywords):
            filtered.append(c)
    return filtered

def validate_mcq(mcq):
    if len(mcq['question']) < 10:
        return False
    if len(mcq['options']) != 4:
        return False
    return True

def extract_concepts(chunk, topic):
    candidates = re.findall(r"\b([A-Z][a-zA-Z ]{3,40})\b", chunk)
    if topic.lower() not in " ".join(candidates).lower():
        candidates.append(topic)
    return list(set(candidates))[:5]

def generate_mcqs(chunks, topic, num_q):
    mcqs = []
    used = set()
    random.shuffle(chunks)

    for chunk in chunks:
        concepts = extract_concepts(chunk, topic)
        for concept in concepts:
            template = random.choice(GENERIC_PATTERNS)
            question = template.format(c=concept)
            if question in used:
                continue

            correct = f"{concept} is key in its context."
            distractors = random.sample([
                "Not related concept",
                "Opposite statement",
                "Different topic",
                "Unrelated fact",
                "Incorrect statement"
            ], 3)
            options = [correct] + distractors
            random.shuffle(options)
            answer_index = options.index(correct)

            mcq = {"question": question, "options": options, "answer": answer_index}
            if validate_mcq(mcq):
                mcqs.append(mcq)
                used.add(question)
                if len(mcqs) >= num_q:
                    return mcqs

    # fallback
    while len(mcqs) < num_q:
        mcqs.append({
            "question": f"What is {topic}?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": 0
        })
    return mcqs

# =========================
# STREAMLIT INTERFACE
# =========================
with st.sidebar:
    st.info("Load NCERT subjects (zip) to generate UPSC-style MCQs")
    if st.button("Load NCERT Content"):
        download_zip()
        extract_zip()
        st.success("NCERT content loaded!")

subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic/Chapter")
num_q = st.number_input("Number of MCQs", 1, 20, 5)

if st.button("Generate UPSC-style MCQs"):
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    texts = []
    for pdf_file in Path(EXTRACT_DIR).rglob("*.pdf"):
        txt = clean_text(read_pdf(str(pdf_file)))
        if len(txt.split()) > 50:
            texts.append(txt)

    if not texts:
        st.error("No readable NCERT text found after extraction.")
        st.stop()

    # Semantic chunking
    chunks = []
    for text in texts:
        chunks.extend(semantic_chunking(text))

    # Boolean filtering
    relevant_chunks = filter_chunks(chunks, topic, subject)
    if len(relevant_chunks) < 5:
        relevant_chunks = chunks[:10]

    # Generate MCQs
    mcqs = generate_mcqs(relevant_chunks, topic, num_q)
    st.success(f"Generated {len(mcqs)} UPSC-style MCQs for '{topic}'")

    for i, mcq in enumerate(mcqs, 1):
        st.write(f"**Q{i}. {mcq['question']}**")
        for idx, opt in enumerate(mcq["options"]):
            st.write(f"{chr(97 + idx)}) {opt}")
        st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
        st.write("---")
