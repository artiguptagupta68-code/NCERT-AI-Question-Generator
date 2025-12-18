# NCERT & UPSC Context-aware MCQ Generator
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
st.set_page_config(page_title="NCERT & UPSC MCQ Generator", layout="wide")
st.title("📘 NCERT & UPSC MCQ Generator")

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

# =========================
# BOOLEAN FILTER
# =========================
def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [c for c in chunks if topic in c.lower() or any(k in c.lower() for k in keys)]

# =========================
# VALIDATION
# =========================
def is_valid_question(q):
    return len(q.split()) > 5

# =========================
# MCQ GENERATION
# =========================

# =========================
# CONTEXTUAL MCQ GENERATION
# =========================
def generate_mcqs_exam_ready(chunks, topic, n, level, subject):
    """
    Generate meaningful MCQs from text chunks.
    - chunks: semantically chunked text
    - topic: topic string
    - n: number of questions
    - level: "NCERT Level" or "UPSC Level"
    - subject: subject context
    """
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

            # Pick main sentence related to topic as correct answer
            correct_candidates = [s for s in sentences if topic.lower() in s.lower() and s not in used_sentences]
            if not correct_candidates:
                correct_candidates = [s for s in sentences if s not in used_sentences]
            if not correct_candidates:
                continue

            correct = random.choice(correct_candidates)
            used_sentences.add(correct)

            # Distractors: other sentences from chunk not correct
            distractors = [s for s in sentences if s != correct]
            distractors = random.sample(distractors, min(3, len(distractors)))
            while len(distractors) < 3:
                distractors.append("Incorrect statement")

            options = [correct] + distractors
            random.shuffle(options)
            
            if level == "NCERT Level":
                answer_index = options.index(correct)
            else:  # UPSC: multiple correct (1-2)
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

    # Fallback if not enough questions
    while len(mcqs) < n:
        mcqs.append({
            "question": f"What is {topic}?",
            "options": ["It is important", "Not relevant", "Unrelated", "Random fact"],
            "answer": 0
        })

    return mcqs


# =========================
# DISPLAY MCQs
# =========================
# =========================
# TABS SETUP
# =========================
tab1, tab2 = st.tabs(["Subjective Questions", "MCQs"])

# Subjective questions logic
with tab1:
    st.write("Subjective Questions will appear here.")
    # ... your subjective generation code ...

# MCQs logic
with tab2:
    st.write("MCQs will appear here.")
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

            # Correct answer(s)
            if isinstance(mcq['answer'], list):
                answers = ", ".join([chr(97 + a) for a in mcq['answer']])
                st.write(f"✅ **Answer(s):** {answers}")
            else:
                st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            
            st.write("---")
