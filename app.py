# ===============================
# NCERT + UPSC Exam-Ready Generator (STREAMLIT SAFE)
# ===============================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
from pypdf import PdfReader

# -------------------------------
# CONFIG
# -------------------------------
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# -------------------------------
# ZIP UPLOAD (REQUIRED FOR STREAMLIT)
# -------------------------------
with st.sidebar:
    st.header("📥 Load NCERT Books")

    uploaded_zip = st.file_uploader(
        "Upload NCERT ZIP file", type=["zip"]
    )

    if uploaded_zip:
        with open(ZIP_PATH, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

        st.success("NCERT PDFs extracted successfully!")

# -------------------------------
# UTILITIES
# -------------------------------
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
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
        text = clean_text(read_pdf(str(pdf)))
        if len(text.split()) > 100:
            texts.append(text)
    return texts

def semantic_chunks(text):
    sents = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]

# -------------------------------
# TOPIC MATCHING (ROBUST)
# -------------------------------
def is_topic_relevant(text, topic):
    topic_words = topic.lower().split()
    text = text.lower()
    return any(word in text for word in topic_words)

def get_relevant_chunks(chunks, topic):
    return [
        ch for ch in chunks
        if is_topic_relevant(ch, topic)
        and len(ch.split()) >= 25
        and not any(x in ch.lower() for x in ["activity", "exercise", "table"])
    ]

# -------------------------------
# MCQ CAPACITY
# -------------------------------
def count_possible_mcqs(chunks):
    return len(chunks)

# -------------------------------
# SUBJECTIVE QUESTIONS
# -------------------------------
def generate_subjective(topic, n):
    base = [
        f"Explain the concept of {topic}.",
        f"Discuss the importance of {topic}.",
        f"Describe the main features of {topic}.",
        f"Examine the role of {topic} in Indian democracy.",
        f"Analyse {topic} with suitable examples."
    ]
    return base[:n]

# -------------------------------
# NCERT MCQs
# -------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []

    for ch in chunks[:n]:
        sentences = [s for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        if not sentences:
            continue

        correct = sentences[0]

        options = [
            correct,
            f"{topic} is limited only to economic aspects.",
            f"{topic} applies only during emergencies.",
            f"{topic} has no constitutional basis."
        ]
        random.shuffle(options)

        mcqs.append({
            "q": f"Which of the following best explains **{topic}**?",
            "options": options,
            "answer": options.index(correct)
        })

    return mcqs

# -------------------------------
# UPSC TYPES
# -------------------------------
def generate_upsc_statements(topic, n):
    return [{
        "statements": [
            f"{topic} is mentioned in the Constitution of India.",
            f"{topic} influences governance in India.",
            f"{topic} is enforceable only by courts."
        ],
        "answer": "1 and 2"
    } for _ in range(n)]

def generate_assertion_reason(topic, n):
    return [{
        "A": f"{topic} is a foundational principle of the Constitution.",
        "R": "It reflects the values on which the Constitution is based."
    } for _ in range(n)]

# -------------------------------
# USER INPUT
# -------------------------------
subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Enter Topic (e.g. Democracy, Equality)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD CONTENT (SAFE)
# -------------------------------
texts, chunks = [], []

if Path(EXTRACT_DIR).exists():
    texts = load_all_texts()
    for t in texts:
        chunks.extend(semantic_chunks(t))

# DEBUG (IMPORTANT)
st.write("📄 PDFs Loaded:", len(texts))
st.write("🧩 Total Chunks Extracted:", len(chunks))

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📝 Subjective", "🧠 MCQs"])

# -------------------------------
# SUBJECTIVE
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic:
            st.error("Please enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            st.write("🔍 Relevant Chunks Found:", len(relevant))

            if not relevant:
                st.error("❌ No relevant NCERT content found.")
            else:
                qs = generate_subjective(topic, min(num_q, len(relevant)))
                for i, q in enumerate(qs, 1):
                    st.write(f"{i}. {q}")

# -------------------------------
# MCQs
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"]
    )

    if st.button("Generate MCQs"):
        if not topic:
            st.error("Please enter a topic.")
        else:
            relevant = get_relevant_chunks(chunks, topic)
            st.write("🔍 Relevant Chunks Found:", len(relevant))

            if not relevant:
                st.error("❌ No meaningful NCERT content found.")
            else:
                final_n = min(num_q, len(relevant))

                if mcq_type == "NCERT MCQs":
                    mcqs = generate_ncert_mcqs(relevant, topic, final_n)
                    for i, m in enumerate(mcqs, 1):
                        st.write(f"**Q{i}. {m['q']}**")
                        for j, opt in enumerate(m["options"]):
                            st.write(f"{chr(97+j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                        st.write("---")

                elif mcq_type == "UPSC PYQ – Statements":
                    qs = generate_upsc_statements(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. With reference to {topic}, consider the following:**")
                        for idx, s in enumerate(q["statements"], 1):
                            st.write(f"{idx}. {s}")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

                else:
                    qs = generate_assertion_reason(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Assertion:** {q['A']}")
                        st.write(f"**Reason:** {q['R']}")
                        st.write("a) Both true and R explains A")
                        st.write("b) Both true but R not explanation")
                        st.write("c) A true, R false")
                        st.write("d) A false, R true")
                        st.write("✅ Answer: a")
                        st.write("---")
