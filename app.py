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
SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
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
    text = re.sub(
        r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*",
        " ", text, flags=re.I
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
    sentences = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]

def classify_sentence(sentence):
    s = sentence.lower()
    if any(x in s for x in ["means", "refers to", "defined as", "can be understood as"]):
        return "definition"
    if any(x in s for x in ["ensures", "protects", "aims to", "seeks to","embodies","represents","declares","lays down"]):
        return "function"
    if any(x in s for x in ["therefore", "thus", "hence", "as a result"]):
        return "implication"
    if any(x in s for x in ["article", "amendment", "schedule"]):
        return "constitutional_fact"
    return "general"

# -------------------------------
# SUBJECTIVE QUESTIONS
# -------------------------------
def generate_subjective(topic, n):
    templates = [
        f"Explain the concept of {topic}.",
        f"Discuss the significance of {topic}.",
        f"Describe the main features of {topic}.",
        f"Why is {topic} important in a democracy?",
        f"Examine the role of {topic} in the Indian political system."
    ]
    return templates[:n]

# -------------------------------
# NCERT MCQs
# -------------------------------
def generate_ncert_mcqs(topic, n):
    """
    Generates NCERT-style MCQs for a given topic using template-based correct answers.
    """
    mcqs = []

    # Example template answers (expandable for each topic)
    topic_answers = {
        "preamble": "It reflects the ideals and philosophy of the Constitution.",
        "fundamental rights": "They protect the liberties of citizens.",
        "federalism": "It divides powers between central and state governments."
    }

    correct = topic_answers.get(topic.lower(), f"The {topic} is important in the Indian Constitution.")

    distractors = [
        "It is a temporary political arrangement.",
        "It deals only with economic policies.",
        "It applies only during emergency situations."
    ]

    for i in range(n):
        options = [correct] + distractors
        random.shuffle(options)
        mcqs.append({
            "q": f"Which of the following best describes {topic}?",
            "options": options,
            "answer": options.index(correct)
        })

    return mcqs


# -------------------------------
# UPSC PYQ – STATEMENTS
# -------------------------------
def generate_upsc_statements(topic, n):
    qs = [{
        "statements":[
            f"{topic} reflects the objectives of the Indian Constitution.",
            f"{topic} guides the interpretation of constitutional provisions."
        ],
        "answer":"1 and 2"
    }]
    return qs[:n]

# -------------------------------
# UPSC PYQ – ASSERTION REASON
# -------------------------------
def generate_assertion_reason(topic, n):
    qs = [{
        "A": f"The {topic} is an integral part of the Indian Constitution.",
        "R": f"It embodies the ideals and philosophy on which the Constitution is based.",
        "answer":"a"
    }]
    return qs[:n]

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded successfully")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Preamble, Federalism)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD CONTENT
# -------------------------------
texts = load_all_texts()
chunks = []
for t in texts:
    chunks.extend(semantic_chunks(t))
if not chunks:
    st.warning("No PDF content loaded. Please load NCERT PDFs first.")
    st.stop()

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📝 Subjective (NCERT)", "🧠 MCQs (NCERT + UPSC)"])

# -------------------------------
# SUBJECTIVE TAB
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions", key="subj"):
        qs = generate_subjective(topic, num_q)
        for i,q in enumerate(qs,1):
            st.write(f"{i}. {q}")

# -------------------------------
# MCQs TAB
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs","UPSC PYQ – Statements","UPSC PYQ – Assertion Reason"]
    )
    if st.button("Generate MCQs", key="mcq"):
        if mcq_type=="NCERT MCQs":
            mcqs = generate_ncert_mcqs(topic, n)
            for i,m in enumerate(mcqs,1):
                st.write(f"**Q{i}. {m['q']}**")
                for j,opt in enumerate(m['options']):
                    st.write(f"{chr(97+j)}) {opt}")
                st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                st.write("---")
        elif mcq_type=="UPSC PYQ – Statements":
            qs = generate_upsc_statements(topic, num_q)
            for i,q in enumerate(qs,1):
                st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                for idx,s in enumerate(q["statements"],1):
                    st.write(f"{idx}. {s}")
                st.write("Which of the statements given above are correct?")
                st.write(f"✅ Answer: {q['answer']}")
                st.write("---")
        else:
            qs = generate_assertion_reason(topic, num_q)
            for i,q in enumerate(qs,1):
                st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                st.write(f"**Reason (R):** {q['R']}")
                st.write("a) Both A and R are true and R is the correct explanation of A")
                st.write("b) Both A and R are true but R is not the correct explanation of A")
                st.write("c) A is true but R is false")
                st.write("d) A is false but R is true")
                st.write("✅ Answer: a")
                st.write("---")

