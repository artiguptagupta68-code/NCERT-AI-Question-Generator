#mcq ncert level
# app.py
import os
import zipfile
from pathlib import Path
import re
import random
from difflib import SequenceMatcher

import streamlit as st
import gdown
from pypdf import PdfReader

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz/view?usp=sharing"  # NCERT Books ZIP
ZIP_PATH = "/content/drive/MyDrive/ncrt subject.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "writ", "rights", "emergency", "judiciary", "parliament"],
    "Sociology": ["society", "social", "movement", "caste", "class", "gender"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "leadership", "planning", "marketing", "controlling"],
    "Economics": ["economy", "gdp", "growth", "inflation", "poverty", "development"]
}

GENERIC_QUESTION_PATTERNS = [
    "Define {c}.",
    "Explain the concept of {c}.",
    "Describe the main features of {c}.",
    "Why is {c} important? Explain.",
    "Give an example to illustrate {c}.",
    "List any two characteristics of {c}.",
    "Explain any two points related to {c}."
]

# ----------------------------
# STREAMLIT SETUP
# ----------------------------
st.set_page_config(page_title="NCERT Question Generator", layout="wide")
st.title("📘 NCERT NCERT-Style Question Generator (Class XI–XII)")

# ----------------------------
# UTILS
# ----------------------------
def download_zip(file_id=FILE_ID, out_path=ZIP_PATH):
    if os.path.exists(out_path):
        return True
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)
    return os.path.exists(out_path)

def extract_zip(zip_path=ZIP_PATH, dest_dir=EXTRACT_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def topic_match(chunk, topic):
    topic_words = set(re.findall(r"\w+", topic.lower()))
    chunk_words = set(re.findall(r"\w+", chunk.lower()))
    if not topic_words:
        return False
    overlap = topic_words.intersection(chunk_words)
    return len(overlap) / len(topic_words) >= 0.4

# =========================
# LOAD CONTENT
# =========================
def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 150:
            texts.append(t)
    return texts

def chunk_text(text):
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if 60 <= len(p.split()) <= 200]

# =========================
# QUESTION GENERATION
# =========================
def extract_concepts(chunk, topic):
    candidates = re.findall(r"\b([A-Z][a-zA-Z ]{3,40})\b", chunk)
    if topic.lower() not in " ".join(candidates).lower():
        candidates.append(topic)
    return list(set(candidates))[:5]

def is_valid(q):
    if len(q.split()) < 6:
        return False
    if re.search(r"(instruction|marks|time allowed)", q.lower()):
        return False
    return True

def generate_subjective(chunks, topic, n):
    questions, seen = [], set()
    random.shuffle(chunks)
    for ch in chunks:
        concepts = extract_concepts(ch, topic)
        for c in concepts:
            q = random.choice(GENERIC_PATTERNS).format(c=c)
            if q not in seen and is_valid(q):
                seen.add(q)
                questions.append(q)
            if len(questions) >= n:
                return questions
    # fallback
    if not questions:
        questions = [f"Explain the concept of {topic}." for _ in range(n)]
    return questions

# =========================
# MCQ GENERATION
# =========================
MCQ_TEMPLATES = [
    {
        "q": "What is meant by {c}?",
        "options": [
            "A moral guideline",
            "A government policy document",
            "A set of fundamental principles governing a state",
            "A political ideology"
        ],
        "answer": 2
    },
    {
        "q": "Why do societies need {c}?",
        "options": [
            "To centralise power",
            "To limit misuse of power and protect rights",
            "To support political parties",
            "To ensure economic equality"
        ],
        "answer": 1
    },
    {
        "q": "Which of the following is a feature of {c}?",
        "options": [
            "Arbitrary exercise of power",
            "Supremacy of the Constitution",
            "Rule by a single authority",
            "Absence of rights"
        ],
        "answer": 1
    },
    {
        "q": "The importance of {c} in a democracy lies in the fact that it:",
        "options": [
            "Gives unlimited power to the government",
            "Defines limits on government authority",
            "Weakens the judiciary",
            "Reduces citizen participation"
        ],
        "answer": 1
    }
]

def generate_mcqs(topic, num_q):
    mcqs = []
    used = set()
    random.shuffle(MCQ_TEMPLATES)
    for temp in MCQ_TEMPLATES:
        if len(mcqs) >= num_q:
            break
        question = temp["q"].format(c=topic)
        if question in used:
            continue
        options = temp["options"]
        answer = temp["answer"]
        mcqs.append({"question": question, "options": options, "answer": answer})
        used.add(question)
    # fallback
    while len(mcqs) < num_q:
        mcqs.append({"question": f"Explain the concept of {topic}.",
                     "options": ["Option1", "Option2", "Option3", "Option4"], "answer": 0})
    return mcqs

# =========================
# UI
# =========================
with st.sidebar:
    st.info("NCERT-aligned | Sample-paper driven")
    load_btn = st.button("Load NCERT Content")

if load_btn:
    download_zip()
    extract_zip()
    st.success("NCERT content loaded")

subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Chapter/Topic")
num_q = st.number_input("Number of Questions", 1, 20, 5)

q_type = st.radio("Select Question Type", ["Subjective", "Multiple Choice (MCQ)"])

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Enter a topic/chapter")
        st.stop()

    texts = load_all_texts()
    if not texts:
        st.error("No readable PDFs found.")
        st.stop()

    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))

    # Filter relevant chunks
    relevant = [c for c in chunks if topic.lower() in c.lower()]
    if len(relevant) < 5:
        relevant = [c for c in chunks if any(k in c.lower() for k in SUBJECT_KEYWORDS[subject])]
    if not relevant:
        relevant = chunks[:10]

    if q_type == "Subjective":
        questions = generate_subjective(relevant, topic, num_q)
        st.success(f"Generated {len(questions)} NCERT-style subjective questions")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
    else:
        mcqs = generate_mcqs(topic, num_q)
        st.success(f"Generated {len(mcqs)} NCERT-style MCQs")
        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            st.write("---")
