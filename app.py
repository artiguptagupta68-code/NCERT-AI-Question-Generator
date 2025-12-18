# =========================================================
# NCERT & UPSC Question Generator (FINAL – EXAM READY)
# =========================================================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert_books.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "preamble", "rights", "federalism", "judiciary", "parliament"],
    "Sociology": ["society", "caste", "class", "gender", "movement"],
    "Psychology": ["learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "planning", "organising", "marketing"],
    "Economics": ["growth", "development", "poverty", "inflation", "gdp"]
}

# ----------------------------
# STREAMLIT SETUP
# ----------------------------
st.set_page_config("NCERT & UPSC Question Generator", layout="wide")
st.title("📘 NCERT & UPSC Context-Aware Question Generator")

# ----------------------------
# UTILITIES
# ----------------------------
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        txt = clean_text(read_pdf(str(pdf)))
        if len(txt.split()) > 200:
            texts.append(txt)
    return texts

def semantic_chunks(text):
    sents = re.split(r'(?<=[.])\s+', text)
    chunks = []
    for i in range(0, len(sents), 4):
        ch = " ".join(sents[i:i+4])
        if 60 <= len(ch.split()) <= 180:
            chunks.append(ch)
    return chunks

# ----------------------------
# HARD CONTENT FILTER
# ----------------------------
def is_content_sentence(s):
    banned = [
        "let’s", "let us", "debate", "activity", "exercise",
        "write a short note", "discuss", "do you think",
        "project", "answer the following"
    ]
    s_low = s.lower()
    if len(s.split()) < 8 or len(s.split()) > 30:
        return False
    return not any(b in s_low for b in banned)

# ----------------------------
# SUBJECTIVE QUESTIONS (NCERT)
# ----------------------------
SUBJECTIVE_PATTERNS = [
    "Define {c}.",
    "Explain the meaning of {c}.",
    "Describe the significance of {c}.",
    "What do you understand by {c}?",
    "Explain the role of {c}."
]

def extract_concepts(chunk, topic):
    concepts = re.findall(r"\b[A-Z][a-zA-Z ]{3,40}\b", chunk)
    if topic.lower() not in " ".join(concepts).lower():
        concepts.append(topic)
    return list(dict.fromkeys(concepts))[:4]

def generate_subjective(chunks, topic, n):
    qs, used = [], set()
    for ch in chunks:
        if topic.lower() not in ch.lower():
            continue
        for c in extract_concepts(ch, topic):
            for p in SUBJECTIVE_PATTERNS:
                q = p.format(c=c)
                if q not in used:
                    used.add(q)
                    qs.append(q)
                if len(qs) >= n:
                    return qs
    return qs

# ----------------------------
# NCERT MCQs
# ----------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    for ch in chunks:
        if topic.lower() not in ch.lower():
            continue

        sents = [s.strip() for s in re.split(r'[.;]', ch) if is_content_sentence(s)]
        if len(sents) < 4:
            continue

        correct = sents[0]
        distractors = sents[1:4]
        options = [correct] + distractors
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best describes {topic}?",
            "options": options,
            "ans": options.index(correct)
        })

        if len(mcqs) >= n:
            break
    return mcqs

# ----------------------------
# UPSC – STATEMENT BASED
# ----------------------------
def generate_upsc_statements(chunks, topic, n):
    mcqs = []
    for ch in chunks:
        if topic.lower() not in ch.lower():
            continue

        sents = [s.strip() for s in re.split(r'[.;]', ch) if is_content_sentence(s)]
        if len(sents) < 3:
            continue

        mcqs.append({
            "statements": sents[:3],
            "options": [
                "1 and 2 only",
                "2 and 3 only",
                "1 and 3 only",
                "1, 2 and 3"
            ],
            "ans": 3
        })

        if len(mcqs) >= n:
            break
    return mcqs

# ----------------------------
# UPSC – ASSERTION REASON
# ----------------------------
def generate_assertion_reason(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "A": f"The {topic} reflects the basic philosophy of the Constitution.",
            "R": f"The {topic} expresses the objectives and ideals of the state.",
            "ans": 0
        })
    return qs

# =========================================================
# UI
# =========================================================
download_and_extract()

subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic")
num_q = st.number_input("Number of Questions", 1, 15, 5)

tab1, tab2 = st.tabs(["📖 Subjective", "📝 MCQs"])

texts = load_all_texts()
chunks = []
for t in texts:
    chunks.extend(semantic_chunks(t))

# ----------------------------
# SUBJECTIVE TAB
# ----------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        qs = generate_subjective(chunks, topic, num_q)
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

# ----------------------------
# MCQs TAB
# ----------------------------
with tab2:
    mcq_type = st.radio("Select MCQ Type", ["NCERT", "UPSC"])

    if mcq_type == "NCERT":
        if st.button("Generate NCERT MCQs"):
            mcqs = generate_ncert_mcqs(chunks, topic, num_q)
            for i, m in enumerate(mcqs, 1):
                st.write(f"**Q{i}. {m['q']}**")
                for j, o in enumerate(m["options"]):
                    st.write(f"{chr(97+j)}) {o}")
                st.write(f"✅ **Answer:** {chr(97+m['ans'])}")
                st.write("---")

    else:
        upsc_type = st.radio("UPSC Question Type", ["Statement-based", "Assertion–Reason"])

        if upsc_type == "Statement-based" and st.button("Generate UPSC MCQs"):
            mcqs = generate_upsc_statements(chunks, topic, num_q)
            for i, m in enumerate(mcqs, 1):
                st.write(f"**Q{i}. Consider the following statements:**")
                for idx, s in enumerate(m["statements"], 1):
                    st.write(f"{idx}. {s}")
                for j, o in enumerate(m["options"]):
                    st.write(f"{chr(97+j)}) {o}")
                st.write(f"✅ **Answer:** {chr(97+m['ans'])}")
                st.write("---")

        if upsc_type == "Assertion–Reason" and st.button("Generate Assertion–Reason"):
            qs = generate_assertion_reason(topic, num_q)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}.**")
                st.write(f"Assertion (A): {q['A']}")
                st.write(f"Reason (R): {q['R']}")
                st.write("a) Both A and R are true and R is the correct explanation of A")
                st.write("b) Both A and R are true but R is not the correct explanation of A")
                st.write("c) A is true but R is false")
                st.write("d) A is false but R is true")
                st.write("✅ **Answer:** a")
                st.write("---")
