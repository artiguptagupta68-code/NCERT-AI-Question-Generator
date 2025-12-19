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
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
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

# -------------------------------
# TOPIC RELEVANCE
# -------------------------------
def is_topic_relevant(sentence, topic):
    topic_words = topic.lower().split()
    return any(word in sentence.lower() for word in topic_words)

def get_relevant_chunks(chunks, topic):
    topic_words = topic.lower().split()
    good = []
    for ch in chunks:
        ch_lower = ch.lower()
        matches = sum(1 for w in topic_words if w in ch_lower)
        if matches >= 1 and not any(x in ch_lower for x in ["activity", "exercise", "project", "table", "figure"]):
            good.append(ch)
    return good

# -------------------------------
# COUNT POSSIBLE MCQs
# -------------------------------
def count_possible_mcqs(chunks):
    count = 0
    for ch in chunks:
        sents = [s for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        if sents:
            count += 1
    return count

# -------------------------------
# KEYWORD HIGHLIGHT
# -------------------------------
def highlight_keywords(sentence):
    keywords = ["constitution", "freedom", "rights", "democracy", "equality", "india"]
    for k in keywords:
        sentence = re.sub(fr"\b({k})\b", r"**\1**", sentence, flags=re.I)
    return sentence

# -------------------------------
# DYNAMIC DISTRACTORS
# -------------------------------
def get_dynamic_distractors(chunks, correct, topic, k=3):
    pool = []
    for ch in chunks:
        for s in re.split(r'[.;]', ch):
            s = s.strip()
            if len(s.split()) >= 8 and s != correct and is_topic_relevant(s, topic):
                pool.append(s)
    random.shuffle(pool)
    return pool[:k]

# -------------------------------
# SUBJECTIVE QUESTIONS
# -------------------------------
def generate_subjective(topic, n):
    templates = [
        f"Explain the concept of {topic}.",
        f"Discuss the significance of {topic}.",
        f"Describe the main features of {topic}.",
        f"Why is {topic} important in a democracy?",
        f"Examine the role of {topic} in the Indian Constitution.",
    ]
    return templates[:n]

# -------------------------------
# NCERT MCQs
# -------------------------------
def generate_ncert_mcqs(chunks, topic, n):
    mcqs = []
    used = set()
    for ch in chunks:
        sentences = [s.strip() for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        if not sentences:
            continue
        correct = sentences[0]
        if correct in used:
            continue
        distractors = get_dynamic_distractors(chunks, correct, topic)
        if len(distractors) < 2:
            continue
        used.add(correct)
        options = [highlight_keywords(correct)] + distractors[:3]
        random.shuffle(options)
        mcqs.append({
            "q": f"Which of the following best describes **{topic}**?",
            "options": options,
            "answer": options.index(highlight_keywords(correct))
        })
        if len(mcqs) >= n:
            break
    # fallback
    while len(mcqs) < n:
        mcqs.append({
            "q": f"Which of the following statements best reflects **{topic}**?",
            "options": [
                f"{topic} is a foundational value of the Indian Constitution.",
                "It applies only during emergency situations.",
                "It is a temporary political arrangement.",
                "It deals only with economic policies.",
            ],
            "answer": 0
        })
    return mcqs

# -------------------------------
# UPSC STATEMENTS (context-aware)
# -------------------------------
def generate_upsc_statements_dynamic(chunks, topic, n):
    qs = []
    used = set()
    for ch in chunks:
        sentences = [s.strip() for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        for s in sentences:
            if s in used:
                continue
            used.add(s)
            statements = [
                s,
                f"{topic} guides the interpretation of constitutional provisions.",
                f"{topic} reflects the ideals of the Indian Constitution."
            ]
            qs.append({"statements": statements, "answer": "1 and 2"})
            if len(qs) >= n:
                return qs
    return qs

# -------------------------------
# UPSC ASSERTION-REASON (context-aware)
# -------------------------------
def generate_assertion_reason_dynamic(chunks, topic, n):
    qs = []
    used = set()
    for ch in chunks:
        sentences = [s.strip() for s in re.split(r'[.;]', ch) if len(s.split()) >= 10]
        for s in sentences:
            if s in used:
                continue
            used.add(s)
            qs.append({
                "A": s,
                "R": f"It embodies the philosophy and objectives of the Constitution.",
                "answer": "a"
            })
            if len(qs) >= n:
                return qs
    return qs

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded successfully")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Preamble, Freedom, Equality)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD CONTENT
# -------------------------------
texts = []
chunks = []

if os.path.exists(EXTRACT_DIR):
    texts = load_all_texts()
    for t in texts:
        chunks.extend(semantic_chunks(t))

st.write(f"📄 PDFs detected: {len(texts)}")
st.write(f"🧩 Total chunks extracted: {len(chunks)}")

relevant_chunks = get_relevant_chunks(chunks, topic)
max_possible = count_possible_mcqs(relevant_chunks)
st.write(f"🔍 Relevant chunks found: {len(relevant_chunks)}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2 = st.tabs(["📝 Subjective (NCERT)", "🧠 MCQs (NCERT + UPSC)"])

# -------------------------------
# SUBJECTIVE TAB
# -------------------------------
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.error("Please enter a topic first.")
        else:
            relevant_chunks = get_relevant_chunks(chunks, topic)
            max_possible = count_possible_mcqs(relevant_chunks)
            final_n = min(num_q, max_possible)
            if max_possible == 0:
                st.error("❌ No meaningful NCERT content found for this topic.")
            else:
                st.info(f"📊 {max_possible} meaningful questions possible. Showing {final_n}.")
                qs = generate_subjective(topic, final_n)
                for i, q in enumerate(qs, 1):
                    st.write(f"{i}. {q}")

# -------------------------------
# MCQs TAB
# -------------------------------
with tab2:
    mcq_type = st.radio(
        "MCQ Type",
        ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"]
    )

    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Please enter a topic first.")
        else:
            relevant_chunks = get_relevant_chunks(chunks, topic)
            max_possible = count_possible_mcqs(relevant_chunks)
            final_n = min(num_q, max_possible)

            if max_possible == 0:
                st.error("❌ No meaningful NCERT content found for this topic.")
            else:
                st.info(f"📊 {max_possible} meaningful MCQs possible. Showing {final_n}.")

                if mcq_type == "NCERT MCQs":
                    mcqs = generate_ncert_mcqs(relevant_chunks, topic, final_n)
                    for i, m in enumerate(mcqs, 1):
                        st.write(f"**Q{i}. {m['q']}**")
                        for j, opt in enumerate(m["options"]):
                            st.write(f"{chr(97+j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                        st.write("---")

                elif mcq_type == "UPSC PYQ – Statements":
                    qs = generate_upsc_statements_dynamic(relevant_chunks, topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                        for idx, s in enumerate(q["statements"], 1):
                            st.write(f"{idx}. {s}")
                        st.write("Which of the statements given above are correct?")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

                else:
                    qs = generate_assertion_reason_dynamic(relevant_chunks, topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                        st.write(f"**Reason (R):** {q['R']}")
                        st.write("a) Both A and R are true and R is the correct explanation of A")
                        st.write("b) Both A and R are true but R is not the correct explanation of A")
                        st.write("c) A is true but R is false")
                        st.write("d) A is false but R is true")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")
