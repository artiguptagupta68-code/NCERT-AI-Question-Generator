# NCERT & UPSC Context-aware Question Generator
import os
import zipfile
from pathlib import Path
import re
import random
import streamlit as st
import gdown
from pypdf import PdfReader

# ----------------------------
# CONFIG / PATHS
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # NCERT Books ZIP
ZIP_PATH = "ncert_books.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "writ", "rights", "emergency", "judiciary", "parliament", "federalism"],
    "Sociology": ["society", "social", "movement", "caste", "class", "gender"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "leadership", "planning", "marketing", "controlling"],
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

# ----------------------------
# STREAMLIT SETUP
# ----------------------------
st.set_page_config(page_title="NCERT Question Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# ----------------------------
# UTILS
# ----------------------------
def download_zip(file_id=FILE_ID, out_path=ZIP_PATH):
    if os.path.exists(out_path):
        return True
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)
    return os.path.exists(out_path)

def extract_zip(zip_path=ZIP_PATH, dest_dir=EXTRACT_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(instructions|time allowed|marks|copyright).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# SEMANTIC CHUNKING
# ----------------------------
def semantic_chunk(text, chunk_size=4):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks

# ----------------------------
# CONCEPT EXTRACTION
# ----------------------------
def extract_concepts(chunk, topic):
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', chunk)
                 if len(s.split()) > 8 and all(x not in s.lower() for x in
                                               ["professor", "school", "chapter", "reprint", "pdf", "address", "contributor"])]
    relevant = [s for s in sentences if topic.lower() in s.lower()]
    return relevant or sentences[:3]

# ----------------------------
# LOAD ALL PDFS
# ----------------------------
def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

def chunk_texts(texts):
    chunks = []
    for t in texts:
        chunks.extend(semantic_chunk(t))
    return chunks

# ----------------------------
# SUBJECTIVE QUESTION GENERATION
# ----------------------------
def generate_subjective(chunks, topic, num_q):
    questions, seen = [], set()
    random.shuffle(chunks)
    for ch in chunks:
        concepts = extract_concepts(ch, topic)
        for c in concepts:
            q = random.choice(GENERIC_PATTERNS).format(c=c)
            if q not in seen:
                seen.add(q)
                questions.append(q)
            if len(questions) >= num_q:
                return questions
    if not questions:
        questions = [f"Explain the concept of {topic}." for _ in range(num_q)]
    return questions

# ----------------------------
# MCQ GENERATION
# ----------------------------
def generate_mcqs(chunks, topic, num_q, subject):
    mcqs = []
    used_sentences = set()
    keywords = [k.lower() for k in SUBJECT_KEYWORDS[subject]]
    random.shuffle(chunks)

    for chunk in chunks:
        sentences = extract_concepts(chunk, topic)
        for _ in range(num_q):
            if len(mcqs) >= num_q:
                break
            correct = random.choice(sentences)
            if correct in used_sentences:
                continue
            used_sentences.add(correct)

            distractors = [s for s in sentences if s != correct]
            distractors = random.sample(distractors, min(3, len(distractors)))
            while len(distractors) < 3:
                distractors.append("Incorrect statement")

            options = [correct] + distractors
            random.shuffle(options)
            answer_index = options.index(correct)

            mcqs.append({
                "question": f"Consider the following statements about {topic} and select the correct option:",
                "options": options,
                "answer": answer_index
            })

        if len(mcqs) >= num_q:
            break

    while len(mcqs) < num_q:
        mcqs.append({
            "question": f"What is {topic}?",
            "options": ["It is important", "Not relevant", "Unrelated", "Random fact"],
            "answer": 0
        })
    return mcqs

# ----------------------------
# STREAMLIT UI
# ----------------------------
with st.sidebar:
    st.info("NCERT-aligned | Exam-ready")
    if st.button("Load NCERT Content"):
        download_zip()
        extract_zip()
        st.success("NCERT content loaded")

subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Chapter/Topic")
num_q = st.number_input("Number of Questions", 1, 20, 5)
q_type = st.radio("Select Question Type", ["Subjective", "Multiple Choice (MCQ)"])

tab1, tab2 = st.tabs(["Subjective Questions", "MCQs"])

with tab1:
    st.write("Subjective Questions will appear here.")

with tab2:
    st.write("MCQs will appear here.")

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Enter a topic/chapter")
        st.stop()

    texts = load_all_texts()
    if not texts:
        st.error("No readable PDFs found.")
        st.stop()

    chunks = chunk_texts(texts)

    relevant = [c for c in chunks if topic.lower() in c.lower()]
    if len(relevant) < 5:
        relevant = [c for c in chunks if any(k in c.lower() for k in SUBJECT_KEYWORDS[subject])]
    if not relevant:
        relevant = chunks[:10]

    if q_type == "Subjective":
        questions = generate_subjective(relevant, topic, num_q)
        st.success(f"Generated {len(questions)} NCERT-style subjective questions")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
    else:
        mcqs = generate_mcqs(relevant, topic, num_q, subject)
        st.success(f"Generated {len(mcqs)} NCERT-style MCQs")
        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            st.write("---")
