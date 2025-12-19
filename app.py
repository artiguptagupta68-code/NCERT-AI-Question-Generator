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
EXTRACT_DIR = "ncert_extraction"
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
texts = load_all_texts()
st.write(f"Loaded {len(texts)} NCERT PDFs with meaningful text.")

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
        # Count how many topic words appear in the chunk
        matches = sum(1 for w in topic_words if w in ch_lower)

        # Keep chunks with at least one match
        if matches >= 1:
            # Filter out tables/figures/exercises etc
            if not any(x in ch_lower for x in ["activity", "exercise", "project", "table", "figure"]):
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
# UPSC STATEMENTS
# -------------------------------
def generate_upsc_statements(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "statements": [
                f"{topic} reflects the ideals of the Indian Constitution.",
                f"{topic} guides the interpretation of constitutional provisions.",
                f"{topic} is enforceable by ordinary laws only."
            ],
            "answer": "1 and 2"
        })
    return qs

# -------------------------------
# UPSC ASSERTION-REASON
# -------------------------------
def generate_assertion_reason(topic, n):
    qs = []
    for _ in range(n):
        qs.append({
            "A": f"The {topic} is an integral part of the Indian Constitution.",
            "R": f"It embodies the philosophy and objectives of the Constitution.",
            "answer": "a"
        })
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

relevant_chunks = get_relevant_chunks(chunks, topic)
max_possible = count_possible_mcqs(relevant_chunks)

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
                st.error("No meaningful content found for this topic in NCERT.")
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
                    if not mcqs:
                        st.warning("Not enough clean sentences to generate MCQs.")
                    else:
                        for i, m in enumerate(mcqs, 1):
                            st.write(f"**Q{i}. {m['q']}**")
                            for j, opt in enumerate(m["options"]):
                                st.write(f"{chr(97+j)}) {opt}")
                            st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                            st.write("---")
                elif mcq_type == "UPSC PYQ – Statements":
                    qs = generate_upsc_statements(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                        for idx, s in enumerate(q["statements"], 1):
                            st.write(f"{idx}. {s}")
                        st.write("Which of the statements given above are correct?")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")
                else:
                    qs = generate_assertion_reason(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                        st.write(f"**Reason (R):** {q['R']}")
                        st.write("a) Both A and R are true and R is the correct explanation of A")
                        st.write("b) Both A and R are true but R is not the correct explanation of A")
                        st.write("c) A is true but R is false")
                        st.write("d) A is false but R is true")
                        st.write("✅ Answer: a")
                        st.write("---")
