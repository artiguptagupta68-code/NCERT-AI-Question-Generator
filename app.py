# NCERT & UPSC Context-aware Question Generator
import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_data"

SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

SUBJECT_KEYWORDS = {
    "Polity": ["constitution", "writ", "rights", "judiciary", "parliament", "federalism", "emergency"],
    "Sociology": ["society", "social", "caste", "class", "gender", "movement"],
    "Psychology": ["behaviour", "learning", "memory", "emotion", "motivation"],
    "Business Studies": ["management", "planning", "organising", "leadership", "marketing"],
    "Economics": ["economy", "growth", "gdp", "poverty", "inflation", "development"]
}

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="NCERT & UPSC Question Generator", layout="wide")
st.title("📘 NCERT & UPSC Question Generator")

# =========================
# UTILITIES
# =========================
def download_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
        extract_nested_zips(EXTRACT_DIR)

def extract_nested_zips(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_dir = os.path.join(root, Path(file).stem)
                os.makedirs(nested_extract_dir, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, "r") as nz:
                    nz.extractall(nested_extract_dir)

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

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

# =========================
# SEMANTIC CHUNKING
# =========================
def semantic_chunk(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):
        chunk = " ".join(sentences[i:i+4])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks

def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS.get(subject, [])
    return [c for c in chunks if topic in c.lower() or any(k in c.lower() for k in keys)]

# =========================
# SUBJECTIVE QUESTIONS
# =========================
GENERIC_PATTERNS = [
    "Explain the concept of {c}.",
    "Describe the importance of {c}.",
    "Discuss the role of {c} in society.",
    "Why is {c} significant?"
]

def generate_subjective_questions(chunks, topic, n):
    questions = []
    seen = set()
    random.shuffle(chunks)
    for chunk in chunks:
        q = random.choice(GENERIC_PATTERNS).format(c=topic)
        if q not in seen:
            questions.append(q)
            seen.add(q)
        if len(questions) >= n:
            break
    while len(questions) < n:
        questions.append(f"Explain the concept of {topic}.")
    return questions

# =========================
# MCQ GENERATION
# =========================
def generate_mcqs_exam_ready(chunks, topic, n, level, subject):
    mcqs = []
    used_sentences = set()
    keywords = [k.lower() for k in SUBJECT_KEYWORDS.get(subject, [])]
    random.shuffle(chunks)

    for chunk in chunks:
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', chunk) if s.strip()]
        if not sentences:
            continue

        for _ in range(n):
            if len(mcqs) >= n:
                break

            # Correct candidate sentences
            correct_candidates = [s for s in sentences if topic.lower() in s.lower() and s not in used_sentences]
            if not correct_candidates:
                correct_candidates = [s for s in sentences if s not in used_sentences]
            if not correct_candidates:
                continue

            correct = random.choice(correct_candidates)
            used_sentences.add(correct)

            # Distractors
            distractors = [s for s in sentences if s != correct]
            distractors = random.sample(distractors, min(3, len(distractors)))
            while len(distractors) < 3:
                distractors.append("Incorrect statement")

            options = [correct] + distractors
            random.shuffle(options)

            # Determine answer index
            if level == "NCERT Level":
                answer_index = options.index(correct)
            else:  # UPSC: 1-2 correct answers
                correct_multi = [correct]
                more_correct = [s for s in sentences if s != correct and topic.lower() in s.lower()]
                if more_correct:
                    correct_multi.append(random.choice(more_correct))
                answer_index = [options.index(c) for c in correct_multi if c in options]

            mcqs.append({
                "question": f"Consider the following statements about {topic} and select the correct option(s):",
                "options": options,
                "answer": answer_index
            })

        if len(mcqs) >= n:
            break

    # Fallback
    while len(mcqs) < n:
        mcqs.append({
            "question": f"What is {topic}?",
            "options": ["It is important", "Not relevant", "Unrelated", "Random fact"],
            "answer": 0
        })

    return mcqs

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    if st.button("📥 Load NCERT Content"):
        download_zip()
        extract_zip()
        st.success("NCERT content loaded")

# =========================
# USER INPUT
# =========================
subject = st.selectbox("Select Subject", SUBJECTS)
topic = st.text_input("Enter Topic / Chapter (e.g. Constitution, Federalism)")
num_q = st.number_input("Number of Questions", 1, 10, 5)
level = st.radio("Select Level", ["NCERT Level", "UPSC Level"])

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["Subjective Questions", "MCQs"])

# =========================
# SUBJECTIVE QUESTIONS TAB
# =========================
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.warning("Enter a topic")
            st.stop()

        texts = load_all_texts()
        if not texts:
            st.error("No readable NCERT PDFs found")
            st.stop()

        chunks = []
        for t in texts:
            chunks.extend(semantic_chunk(t))
        relevant = boolean_filter(chunks, topic, subject)
        if len(relevant) < 5:
            relevant = chunks[:15]

        questions = generate_subjective_questions(relevant, topic, num_q)
        st.success(f"Generated {len(questions)} Subjective Questions")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")

# =========================
# MCQS TAB
# =========================
with tab2:
    if st.button("Generate MCQs"):
        if not topic.strip():
            st.warning("Enter a topic")
            st.stop()

        texts = load_all_texts()
        if not texts:
            st.error("No readable NCERT PDFs found")
            st.stop()

        chunks = []
        for t in texts:
            chunks.extend(semantic_chunk(t))
        relevant = boolean_filter(chunks, topic, subject)
        if len(relevant) < 5:
            relevant = chunks[:15]

        mcqs = generate_mcqs_exam_ready(relevant, topic, num_q, level, subject)
        st.success(f"Generated {len(mcqs)} MCQs ({level})")

        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")

            # Display correct answer(s)
            if isinstance(mcq['answer'], list):
                answers = ", ".join([chr(97 + a) for a in mcq['answer']])
                st.write(f"✅ **Answer(s):** {answers}")
            else:
                st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            st.write("---")
