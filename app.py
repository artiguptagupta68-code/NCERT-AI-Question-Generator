# app.py

import os
import zipfile
from pathlib import Path
import re
import random
import streamlit as st
import gdown
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
FILE_ID = "13EiS5fBw4Crjie2wqpR-qZ2CI8FRF4sT"  # Sample NCERT ZIP
ZIP_PATH = "ncert_sample.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "rights", "freedom", "emergency", "judiciary", "parliament"],
    "Sociology": ["society", "social", "movement", "caste", "class", "gender"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "leadership", "planning", "controlling", "marketing"],
    "Economics": ["economy", "gdp", "growth", "inflation", "poverty", "development"]
}

GENERIC_PATTERNS = [
    "Define {c}.",
    "Explain the concept of {c}.",
    "Describe the main features of {c}.",
    "Why is {c} important? Explain.",
    "Give an example to illustrate {c}.",
    "List any two characteristics of {c}.",
    "Explain any two points related to {c}."
]

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

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="NCERT Question Generator", layout="wide")
st.title("📘 NCERT Subjective & UPSC-style MCQ Generator")

# =========================
# UTILS
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
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.warning(f"Failed to read PDF: {path}, {e}")
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts():
    texts = []
    for pdf_path in Path(EXTRACT_DIR).rglob("*.pdf"):
        text = clean_text(read_pdf(str(pdf_path)))
        if len(text.split()) > 100:  # only keep meaningful content
            texts.append(text)
    return texts

def chunk_text(text):
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if 60 <= len(p.split()) <= 200]

def extract_concepts(chunk, topic):
    concepts = re.findall(r"\b([A-Z][a-zA-Z ]{3,40})\b", chunk)
    if topic.lower() not in " ".join(concepts).lower():
        concepts.append(topic)
    return list(set(concepts))[:5]

def is_valid(question):
    if len(question.split()) < 6:
        return False
    if re.search(r"(instruction|marks|time allowed)", question.lower()):
        return False
    return True

def generate_subjective(chunks, topic, n):
    questions, seen = [], set()
    random.shuffle(chunks)
    for chunk in chunks:
        concepts = extract_concepts(chunk, topic)
        for concept in concepts:
            q = random.choice(GENERIC_PATTERNS).format(c=concept)
            if q not in seen and is_valid(q):
                seen.add(q)
                questions.append(q)
            if len(questions) >= n:
                return questions
    return [f"Explain the concept of {topic}." for _ in range(n)]

def generate_mcqs(topic, num_q):
    mcqs, used = [], set()
    random.shuffle(MCQ_TEMPLATES)
    for template in MCQ_TEMPLATES:
        if len(mcqs) >= num_q:
            break
        question = template["q"].format(c=topic)
        if question in used:
            continue
        mcqs.append({
            "question": question,
            "options": template["options"],
            "answer": template["answer"]
        })
        used.add(question)
    while len(mcqs) < num_q:
        mcqs.append({
            "question": f"What is meant by {topic}?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": 0
        })
    return mcqs

# =========================
# LOAD ZIP BUTTON
# =========================
with st.sidebar:
    st.info("NCERT-aligned | Concept-based & UPSC-style questions")
    if st.button("Load NCERT Content"):
        download_zip()
        extract_zip()
        st.success("NCERT content loaded successfully!")

# =========================
# USER INPUT
# =========================
subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Chapter / Topic")
num_q = st.number_input("Number of Questions", 1, 20, 5)
q_type = st.radio("Question Type", ["Subjective", "Multiple Choice (MCQ)"])

# =========================
# GENERATE QUESTIONS
# =========================
if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    texts = load_all_texts()
    if not texts:
        st.error("No readable PDFs found in NCERT data!")
        st.stop()

    chunks = []
    for text in texts:
        chunks.extend(chunk_text(text))

    relevant = [c for c in chunks if topic.lower() in c.lower()]
    if len(relevant) < 5:
        relevant = [c for c in chunks if any(k in c.lower() for k in SUBJECT_KEYWORDS[subject])]
    if not relevant:
        relevant = chunks[:10]

    if q_type == "Subjective":
        questions = generate_subjective(relevant, topic, num_q)
        st.success(f"Generated {len(questions)} subjective questions")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
    else:
        mcqs = generate_mcqs(topic, num_q)
        st.success(f"Generated {len(mcqs)} MCQs")
        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            st.write("---")
