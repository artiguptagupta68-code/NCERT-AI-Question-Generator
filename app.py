import streamlit as st
import random

st.set_page_config(page_title="NCERT + UPSC MCQ Generator", layout="wide")

st.title("📘 NCERT-Aligned UPSC Prelims MCQ Generator")
st.caption("Prelims 2024 level | NCERT-based | Coaching-style questions")

SUBJECTS = {
    "Polity": [
        "Constitution", "Fundamental Rights", "Directive Principles",
        "Parliament", "Judiciary", "Emergency", "Federalism",
        "Writs", "Amendment Procedure"
    ],
    "Economics": [
        "GDP", "Inflation", "Fiscal Policy", "Monetary Policy",
        "Poverty", "Unemployment", "Budget", "Taxation"
    ],
    "Sociology": [
        "Society", "Caste", "Class", "Gender",
        "Social Change", "Institutions", "Family"
    ],
    "Psychology": [
        "Learning", "Memory", "Motivation",
        "Emotion", "Personality", "Stress"
    ],
    "Business Studies": [
        "Management", "Planning", "Organising",
        "Leadership", "Marketing", "Finance", "HRM"
    ]
}

QUESTION_TEMPLATES = [
    "Which of the following best explains {k}?",
    "With reference to {k}, consider the following statements:",
    "{k} is important in the Indian context because it:",
    "The concept of {k} is associated with:"
]

def generate_options(keywords, correct):
    distractors = random.sample([k for k in keywords if k != correct], 3)
    options = distractors + [correct]
    random.shuffle(options)
    return options, options.index(correct)

def generate_mcqs(subject, n):
    keywords = SUBJECTS[subject]
    mcqs = []

    for i in range(1, n + 1):
        correct = random.choice(keywords)
        question = f"Q{i}. " + random.choice(QUESTION_TEMPLATES).format(k=correct)
        options, ans = generate_options(keywords, correct)

        mcqs.append((question, options, ans))

    return mcqs

with st.sidebar:
    subject = st.selectbox("Select Subject", SUBJECTS.keys())
    num_q = st.slider("Number of MCQs", 5, 50, 10)
    show_ans = st.checkbox("Show Answers", True)

if st.button("Generate MCQs"):
    mcqs = generate_mcqs(subject, num_q)
    for q, opts, ans in mcqs:
        st.markdown(f"**{q}**")
        for i, o in enumerate(opts):
            st.write(f"{chr(97+i)}) {o}")
        if show_ans:
            st.write(f"✅ **Answer:** {chr(97+ans)}")
        st.divider()
