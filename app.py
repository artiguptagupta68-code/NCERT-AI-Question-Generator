# ===============================
# NCERT + UPSC Exam-Ready Generator
# ===============================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# -------------------------------
# CONFIG
# -------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert_books.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "preamble", "rights", "federalism", "parliament"],
    "Economics": ["growth", "development", "inflation", "poverty"],
    "Sociology": ["society", "equality", "justice", "movement"],
    "Psychology": ["learning", "memory", "emotion"],
    "Business Studies": ["management", "planning", "organisation"]
}

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config("NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

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
    text = re.sub(r"(activity|let us|exercise|project|editor|reprint|copyright).*",
                  " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 200:
            texts.append(t)
    return texts

def semantic_chunks(text):
    sents = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]

def is_exam_worthy(s):
    bad = ["activity", "debate", "editor", "project", "let us"]
    if len(s.split()) < 10 or len(s.split()) > 35:
        return False
    return not any(b in s.lower() for b in bad)

def classify_sentence(sentence):
    s = sentence.lower()

    if any(x in s for x in ["means", "refers to", "is defined as"]):
        return "definition"

    if any(x in s for x in ["ensures", "protects", "aims to", "seeks to"]):
        return "function"

    if any(x in s for x in ["therefore", "thus", "hence"]):
        return "implication"

    if any(x in s for x in ["for example", "such as"]):
        return "example"

    if any(x in s for x in ["article", "amendment", "schedule"]):
        return "constitutional_fact"

    return "ignore"


# -------------------------------
# SUBJECTIVE QUESTIONS (NCERT)
# -------------------------------
def generate_subjective(chunks, topic, n):
    qs = []
    for ch in chunks:
        if topic.lower() in ch.lower():
            qs.append(f"Explain the concept of {topic}.")
        if len(qs) >= n:
            break
    return qs or [f"Explain {topic}."] * n

# -------------------------------
# NCERT MCQs
# -------------------------------
def generate_ncert_mcqs_refined(chunks, topic, n):
    mcqs = []

    for ch in chunks:
        sentences = re.split(r'[.;]', ch)
        for s in sentences:
            s = s.strip()
            if topic.lower() not in s.lower():
                continue

            kind = classify_sentence(s)

            if kind == "definition":
                question = f"What is meant by {topic}?"
                correct = s

                distractors = [
                    f"It is a political ideology",
                    f"It is a government policy",
                    f"It is a temporary legal arrangement"
                ]

            elif kind == "function":
                question = f"Why is {topic} important?"
                correct = s

                distractors = [
                    f"It centralises all political power",
                    f"It weakens democratic institutions",
                    f"It reduces citizen participation"
                ]

            else:
                continue

            options = [correct] + distractors
            random.shuffle(options)

            mcqs.append({
                "q": question,
                "options": options,
                "answer": options.index(correct)
            })

            if len(mcqs) >= n:
                return mcqs

    return mcqs


# -------------------------------
# UPSC PYQ – Statement Based
# -------------------------------
def generate_upsc_statements(chunks, topic, n):
    qs = []
    for ch in chunks:
        sents = [s for s in re.split(r'[.;]', ch)
                 if is_exam_worthy(s) and topic.lower() in s.lower()]
        if len(sents) >= 3:
            qs.append({
                "statements": sents[:3],
                "answer": "1, 2 and 3"
            })
        if len(qs) >= n:
            break
    return qs

# -------------------------------
# UPSC PYQ – Assertion Reason
# -------------------------------
def generate_assertion_reason(chunks, topic, n):
    qs = []
    for ch in chunks:
        sents = [s for s in re.split(r'[.;]', ch) if is_exam_worthy(s)]
        if len(sents) >= 2:
            qs.append({
                "A": sents[0],
                "R": sents[1],
                "answer": "a"
            })
        if len(qs) >= n:
            break
    return qs

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs Loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Preamble, Federalism)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["Subjective", "MCQs"])

texts = load_all_texts()
chunks = []
for t in texts:
    chunks.extend(semantic_chunks(t))

relevant = [c for c in chunks if topic.lower() in c.lower()]
if not relevant:
    relevant = chunks[:20]

# -------------------------------
# SUBJECTIVE TAB
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        qs = generate_subjective(relevant, topic, num_q)
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

# -------------------------------
# MCQs TAB
# -------------------------------
with tab2:
    mcq_type = st.radio("MCQ Type", ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"])

    if st.button("Generate MCQs"):
        if mcq_type == "NCERT MCQs":
            mcqs = generate_ncert_mcqs(relevant, topic, num_q)
            for i, m in enumerate(mcqs, 1):
                st.write(f"**Q{i}. {m['q']}**")
                for j, opt in enumerate(m["options"]):
                    st.write(f"{chr(97+j)}) {opt}")
                st.write(f"✅ Answer: {chr(97+m['answer'])}")
                st.write("---")

        elif mcq_type == "UPSC PYQ – Statements":
            qs = generate_upsc_statements(relevant, topic, num_q)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                for idx, s in enumerate(q["statements"], 1):
                    st.write(f"{idx}. {s}")
                st.write("Which of the statements given above are correct?")
                st.write(f"✅ Answer: {q['answer']}")
                st.write("---")

        else:
            qs = generate_assertion_reason(relevant, topic, num_q)
            for i, q in enumerate(qs, 1):
                st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                st.write(f"**Reason (R):** {q['R']}")
                st.write("a) Both A and R are true and R explains A")
                st.write("b) Both A and R are true but R does not explain A")
                st.write("c) A is true, R is false")
                st.write("d) A is false, R is true")
                st.write("✅ Answer: a")
                st.write("---")
