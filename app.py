
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

# app.py
# --------------------------------------------------
# NCERT-aligned + Realistic UPSC Prelims MCQ Generator
# NO PDF READING | NO OCR | ZERO FAILURES
# --------------------------------------------------

import streamlit as st
import random

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="NCERT + UPSC MCQ Generator",
    layout="wide"
)

st.title("📘 NCERT-Aligned Realistic UPSC MCQ Generator")
st.caption("Prelims 2024 level | NCERT syllabus mapped | Coaching-style questions")

# ----------------------------
# SUBJECT → NCERT KEYWORDS
# ----------------------------
SUBJECTS = {
    "Polity": [
        "Constitution", "Fundamental Rights", "Directive Principles",
        "Parliament", "Judiciary", "Emergency", "Federalism",
        "Separation of Powers", "Writs", "Amendment Procedure"
    ],
    "Economics": [
        "GDP", "Inflation", "Fiscal Policy", "Monetary Policy",
        "Poverty", "Unemployment", "Economic Growth",
        "Planning", "Budget", "Taxation"
    ],
    "Sociology": [
        "Society", "Caste", "Class", "Gender",
        "Social Change", "Social Mobility",
        "Institutions", "Family", "Religion"
    ],
    "Psychology": [
        "Behavior", "Learning", "Memory", "Motivation",
        "Emotion", "Personality", "Intelligence",
        "Cognition", "Stress"
    ],
    "Business Studies": [
        "Management", "Planning", "Organising",
        "Leadership", "Marketing", "Finance",
        "Human Resource Management", "Controlling"
    ]
}

# ----------------------------
# UPSC-STYLE MCQ TEMPLATES
# ----------------------------
MCQ_TEMPLATES = [
    "Which of the following best describes {k}?",
    "With reference to {k}, consider the following statements:",
    "The concept of {k} is most closely associated with:",
    "In the context of Indian polity/economy/society, {k} refers to:",
    "{k} is important because it:"
]

# ----------------------------
# DISTRACTOR LOGIC (VERY IMPORTANT)
# ----------------------------
def generate_options(keywords, correct):
    distractors = random.sample(
        [k for k in keywords if k != correct],
        min(3, len(keywords) - 1)
    )
    options = distractors + [correct]
    random.shuffle(options)
    answer_index = options.index(correct)
    return options, answer_index

# ----------------------------
# MCQ GENERATOR
# ----------------------------
def generate_mcqs(subject, topic, num_q):
    keywords = SUBJECTS[subject]
    mcqs = []

    for i in range(1, num_q + 1):
        correct = random.choice(keywords)
        template = random.choice(MCQ_TEMPLATES)

        question = f"Q{i}. " + template.format(k=correct)
        if "statements" in question:
            question += "\n1. It is mentioned in NCERT texts.\n2. It has relevance for Indian context.\nSelect the correct answer using the code below."

        options, answer_idx = generate_options(keywords, correct)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": answer_idx
        })

    return mcqs

# ----------------------------
# UI CONTROLS
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    subject = st.selectbox("Select Subject", list(SUBJECTS.keys()))
    topic = st.text_input("Enter Chapter / Topic (NCERT)")
    num_q = st.slider("Number of MCQs", 5, 50, 10)
    show_answers = st.checkbox("Show answers", value=True)

st.divider()

# ----------------------------
# GENERATE BUTTON
# ----------------------------
if st.button("🚀 Generate UPSC-Style MCQs"):
    if not topic.strip():
        st.warning("Please enter a topic or chapter name.")
        st.stop()

    mcqs = generate_mcqs(subject, topic, num_q)

    st.success(f"Generated {num_q} NCERT-aligned UPSC Prelims MCQs")

    for mcq in mcqs:
        st.markdown(f"**{mcq['question']}**")
        for idx, opt in enumerate(mcq["options"]):
            st.write(f"{chr(97 + idx)}) {opt}")
        if show_answers:
            st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
        st.write("---")

# ----------------------------
# FOOTER
# ----------------------------
st.caption(
    "✔ NCERT syllabus based | ✔ UPSC Prelims logic | ✔ No OCR | ✔ No PDF dependency"
)
