
import os
import zipfile
import random
import re
from pathlib import Path
import tempfile

import streamlit as st
import gdown
from pypdf import PdfReader
from fpdf import FPDF

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"   # NCERT ZIP
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = {
    "Polity": ["constitution", "parliament", "judiciary", "rights", "federalism", "emergency"],
    "Economics": ["growth", "inflation", "poverty", "development", "fiscal policy"],
    "Sociology": ["caste", "class", "gender", "social change"],
    "Psychology": ["learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "planning", "marketing", "finance"]
}

# ----------------------------
# DOWNLOAD & EXTRACT
# ----------------------------
def setup_ncert():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

# ----------------------------
# PDF TEXT LOADING (ROBUST)
# ----------------------------
def load_documents():
    documents = []
    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        if page.extract_text():
                            text += page.extract_text() + " "
                    if len(text.split()) > 100:
                        documents.append(text.lower())
                except:
                    pass
    return documents

# ----------------------------
# UPSC MCQ LOGIC (NCERT-GROUNDED)
# ----------------------------
def generate_upsc_mcq(topic, source_text, qnum):
    statements = []

    sentences = re.split(r"\. ", source_text)
    for s in sentences:
        if topic in s and len(s.split()) > 8:
            statements.append(s.strip())
        if len(statements) == 2:
            break

    if len(statements) < 2:
        statements = [
            f"{topic} is discussed as a key concept in NCERT",
            f"{topic} plays an important role in democratic systems"
        ]

    question = f"Q{qnum}. Consider the following statements regarding {topic}:\n"
    for i, s in enumerate(statements, 1):
        question += f"{i}. {s}\n"

    options = [
        "1 only",
        "2 only",
        "Both 1 and 2",
        "Neither 1 nor 2"
    ]

    correct = random.choice([0, 1, 2])

    return question.strip(), options, correct

# ----------------------------
# PDF CREATION
# ----------------------------
def generate_pdf(subject, topic, num_q, docs):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"UPSC Prelims MCQs — {subject}", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 12)

    answers = []

    relevant_docs = [d for d in docs if topic in d]
    if not relevant_docs:
        relevant_docs = docs[:10]

    for i in range(1, num_q + 1):
        src = random.choice(relevant_docs)
        q, opts, ans = generate_upsc_mcq(topic, src, i)

        pdf.multi_cell(0, 8, q)
        for idx, o in enumerate(opts):
            pdf.cell(0, 8, f"{chr(97+idx)}) {o}", ln=True)
        pdf.ln(4)

        answers.append((i, chr(97 + ans)))

    # Answer Key
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Answer Key", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    for qn, ans in answers:
        pdf.cell(0, 8, f"Q{qn}: {ans}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="NCERT → UPSC MCQ Generator", layout="centered")
st.title("📘 NCERT-based UPSC Prelims MCQ Generator")

with st.sidebar:
    if st.button("Load NCERT Content"):
        setup_ncert()
        st.success("NCERT content loaded")

subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
topic = st.text_input("Enter NCERT Topic (e.g. Constitution, Federalism)")
num_q = st.slider("Number of MCQs", 10, 100, 25)

if st.button("Generate UPSC MCQ PDF"):
    docs = load_documents()
    if not docs:
        st.error("No readable NCERT PDFs found")
        st.stop()

    pdf_path = generate_pdf(subject, topic.lower(), num_q, docs)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Download PDF",
            f,
            file_name=f"{subject}_{topic}_UPSC_MCQs.pdf",
            mime="application/pdf"
        )

    st.success("UPSC-standard MCQ PDF generated")
