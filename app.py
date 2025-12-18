# NCERT Question Generator with Subjective & MCQs (STABLE VERSION)

import os
import zipfile
import re
import random
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
    "Polity": ["constitution", "federal", "rights", "judiciary", "parliament"],
    "Sociology": ["society", "caste", "class", "gender"],
    "Psychology": ["learning", "memory", "emotion"],
    "Business Studies": ["management", "planning", "marketing"],
    "Economics": ["economy", "gdp", "inflation", "growth"]
}

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config("NCERT Question Generator", layout="wide")
st.title("📘 NCERT Question Generator (Class XI–XII)")

# =========================
# FILE UTILITIES
# =========================
def download_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)

def extract_zip():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    extract_nested_zips(EXTRACT_DIR)

def extract_nested_zips(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                zp = os.path.join(root, f)
                out = os.path.join(root, Path(f).stem)
                os.makedirs(out, exist_ok=True)
                with zipfile.ZipFile(zp, "r") as nz:
                    nz.extractall(out)

# =========================
# PDF READING (CACHED)
# =========================
@st.cache_data(show_spinner=True)
def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        try:
            reader = PdfReader(str(pdf))
            text = " ".join(p.extract_text() or "" for p in reader.pages)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text.split()) > 50:
                texts.append(text)
        except:
            continue
    return texts

@st.cache_data(show_spinner=True)
def build_chunks(texts):
    chunks = []
    for t in texts:
        sentences = re.split(r'(?<=[.?!])\s+', t)
        for i in range(0, len(sentences), 4):
            chunk = " ".join(sentences[i:i+4])
            if len(chunk.split()) > 30:
                chunks.append(chunk)
    return chunks

# =========================
# SEMANTIC CHUNKING
# =========================
def semantic_chunk(text):
    """
    Splits text into semantic chunks for question generation.
    Each chunk is ~3–5 sentences long, and only included if length > 30 words.
    """
    import re
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), 4):  # 4 sentences per chunk
        chunk = " ".join(sentences[i:i+4])
        if len(chunk.split()) > 30:
            chunks.append(chunk)
    return chunks


# =========================
# FILTERING
# =========================
def boolean_filter(chunks, topic, subject):
    topic = topic.lower()
    keys = SUBJECT_KEYWORDS[subject]
    return [
        c for c in chunks
        if topic in c.lower() or any(k in c.lower() for k in keys)
    ]

# =========================
# QUESTION GENERATION
# =========================
def generate_mcqs_dynamic(topic, chunks, n, subject, level="UPSC Level"):
    """
    Generates UPSC/NCERT standard MCQs from chunks.
    """
    mcqs = []
    used_questions = set()
    
    # Flatten text for distractor pool
    pool_text = " ".join(chunks)
    
    random.shuffle(chunks)
    for chunk in chunks:
        if len(mcqs) >= n:
            break
        sentences = re.split(r'(?<=[.?!])\s+', chunk)
        relevant_sentences = [s for s in sentences if topic.lower() in s.lower()]
        if not relevant_sentences:
            continue
        base_sentence = random.choice(relevant_sentences)
        
        # Question construction
        if level == "NCERT Level":
            question = f"What is/are described in the following statement? \n\n{base_sentence}"
        else:  # UPSC Level
            question = f"Consider the following statement:\n\n{base_sentence}\n\nWhich of the following is correct?"
        
        # Distractors: pick other sentences containing subject keywords but not the answer
        distractors = []
        answer = base_sentence.strip()
        keywords = SUBJECT_KEYWORDS[subject]
        potential_distractors = [s for s in re.split(r'(?<=[.?!])\s+', pool_text)
                                 if any(k in s.lower() for k in keywords)
                                 and s.strip() != answer]
        random.shuffle(potential_distractors)
        for d in potential_distractors:
            if len(distractors) >= 3:
                break
            distractors.append(d.strip())
        
        # Ensure 3 distractors
        while len(distractors) < 3:
            distractors.append("None of the above")  # fallback placeholder
        
        # Shuffle options
        options = [answer] + distractors
        random.shuffle(options)
        correct_idx = options.index(answer)
        
        # Avoid duplicates
        if question in used_questions:
            continue
        used_questions.add(question)
        
        mcqs.append({
            "q": question,
            "options": options,
            "ans": correct_idx
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
topic = st.text_input("Enter Topic (e.g. Federalism, GDP)")
num_q = st.slider("Number of Questions", 3, 10, 5)
level = st.radio("Select Level", ["NCERT Level", "UPSC Level"])

tab1, tab2 = st.tabs(["✍️ Subjective", "📝 MCQs"])

# =========================
# SUBJECTIVE TAB
# =========================
with tab1:
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.warning("Please enter a topic")
            st.stop()

        texts = load_all_texts()
        chunks = build_chunks(texts)
        relevant = boolean_filter(chunks, topic, subject)

        questions = generate_subjective(topic, num_q, level)

        st.success("Questions Generated")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")

# =========================
# MCQ TAB
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

        mcqs = generate_mcqs_dynamic(topic, relevant, num_q, subject, level)
        st.success(f"Generated {len(mcqs)} MCQs ({level})")
        for i, mcq in enumerate(mcqs, 1):
            st.write(f"**Q{i}. {mcq['question']}**")
            for idx, opt in enumerate(mcq["options"]):
                st.write(f"{chr(97+idx)}) {opt}")
            st.write(f"✅ **Answer:** {chr(97 + mcq['answer'])}")
            st.write("---")



