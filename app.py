# ===============================
# NCERT + UPSC Exam-Ready Generator with Chatbot
# ===============================

import os, zipfile, re, random, pickle
from pathlib import Path

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# CONFIG
# -------------------------------
NCERT_FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
UPSC_FILE_ID = "YOUR_UPSC_ZIP_FILE_ID"

NCERT_ZIP = "ncert.zip"
UPSC_ZIP = "upsc.zip"

NCERT_DIR = "ncert_extracted"
UPSC_DIR = "upsc_extracted"

SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator + Chatbot")

# -------------------------------
# UTILITIES
# -------------------------------
def download_and_extract(file_id, zip_path, extract_dir):
    if not os.path.exists(zip_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract main zip
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    
    # Extract nested zips if any
    for zfile in Path(extract_dir).rglob("*.zip"):
        try:
            extract_to = zfile.parent / zfile.stem
            extract_to.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(extract_to)
        except:
            pass

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*",
                  " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

def semantic_chunks(text, size=3):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [" ".join(sentences[i:i+size]) for i in range(0, len(sentences), size)]

# -------------------------------
# LOAD AND EMBED PDF CORPUS
# -------------------------------
def prepare_corpus(extract_dir):
    chunks = []
    for pdf in Path(extract_dir).rglob("*.pdf"):
        text = clean_text(read_pdf(str(pdf)))
        s_chunks = semantic_chunks(text)
        chunks.extend(s_chunks)
    
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    
    return chunks, embeddings

# -------------------------------
# CHATBOT RETRIEVAL
# -------------------------------
def chatbot_answer(query, chunks, embeddings, top_k=5):
    q_emb = embedder.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    retrieved_chunks = [chunks[i] for i in top_indices]
    return "\n\n".join(retrieved_chunks)

# -------------------------------
# SUBJECTIVE / MCQ GENERATORS
# -------------------------------
def is_conceptual(sentence):
    skip_words = ["chapter", "unit", "page", "contents", "glossary", "figure", "table"]
    return not any(word in sentence.lower() for word in skip_words)

def is_topic_relevant(sentence, topic):
    topic_words = topic.lower().split()
    return any(word in sentence.lower() for word in topic_words)

def get_relevant_chunks(chunks, topic):
    good = []
    for ch in chunks:
        if any(word in ch.lower() for word in topic.lower().split()) and is_conceptual(ch):
            good.append(ch)
    return good

def count_possible_mcqs(chunks):
    count = 0
    for ch in chunks:
        sents = [s for s in re.split(r'[.;]', ch) if 8 <= len(s.split()) <= 50]
        if sents:
            count += 1
    return count

def highlight_keywords(sentence):
    keywords = ["constitution", "freedom", "rights", "democracy", "equality", "india", "preamble"]
    for k in keywords:
        sentence = re.sub(fr"\b({k})\b", r"**\1**", sentence, flags=re.I)
    return sentence

def get_dynamic_distractors(chunks, correct, topic, k=3):
    pool = []
    for ch in chunks:
        for s in re.split(r'[.;]', ch):
            s = s.strip()
            if 8 <= len(s.split()) <= 50 and s != correct and is_topic_relevant(s, topic):
                pool.append(s)
    random.shuffle(pool)
    return pool[:k]

def generate_subjective(topic, n, standard="NCERT"):
    if standard == "NCERT":
        templates = [
            f"Explain the concept of {topic}.",
            f"Describe the main features of {topic}.",
            f"What is the importance of {topic} in everyday life?",
            f"Discuss the key points of {topic}.",
            f"Give examples related to {topic}."
        ]
    else:  # UPSC standard
        templates = [
            f"Analyse the role of {topic} in shaping India's constitutional framework.",
            f"Examine the significance of {topic} in promoting democracy and equality.",
            f"Discuss the challenges in implementing {topic} effectively in India.",
            f"Evaluate how {topic} influences socio-economic policies in India.",
            f"Critically assess the impact of {topic} on governance and society."
        ]
    return templates[:n]

def generate_ncert_mcqs_short(chunks, topic, n):
    mcqs = []
    used = set()
    relevant_sentences = [s.strip() for ch in chunks for s in re.split(r'[.;]', ch)
                          if 8 <= len(s.split()) <= 50 and is_conceptual(s) and is_topic_relevant(s, topic)]
    random.shuffle(relevant_sentences)

    for correct in relevant_sentences:
        if correct in used: continue
        used.add(correct)
        distractors = get_dynamic_distractors(chunks, correct, topic)
        if len(distractors) < 2: continue
        options = [highlight_keywords(correct)] + distractors[:3]
        random.shuffle(options)
        mcqs.append({
            "q": f"Which of the following best describes **{topic}**?",
            "options": options,
            "answer": options.index(highlight_keywords(correct))
        })
        if len(mcqs) >= n: break
    return mcqs

def generate_assertion_reason_short(chunks, topic, n):
    arqs, used = [], set()
    relevant_sentences = [s.strip() for ch in chunks for s in re.split(r'[.;]', ch)
                          if 8 <= len(s.split()) <= 50 and is_topic_relevant(s, topic) and is_conceptual(s)]
    random.shuffle(relevant_sentences)
    for s in relevant_sentences:
        if s in used: continue
        used.add(s)
        reason_candidates = [r for r in relevant_sentences if r != s]
        reason = reason_candidates[0] if reason_candidates else s
        arqs.append({
            "A": highlight_keywords(s),
            "R": highlight_keywords(reason),
            "answer": random.choice(["a", "b", "c", "d"])
        })
        if len(arqs) >= n: break
    return arqs

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
# SIDEBAR SETTINGS
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract(NCERT_FILE_ID, NCERT_ZIP, NCERT_DIR)
        st.success("NCERT PDFs loaded successfully")
    if st.button("📥 Load UPSC PDFs"):
        download_and_extract(UPSC_FILE_ID, UPSC_ZIP, UPSC_DIR)
        st.success("UPSC PDFs loaded successfully")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic (e.g. Preamble, Freedom, Equality)")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# -------------------------------
# LOAD CONTENT
# -------------------------------
ncert_chunks, ncert_embeddings = [], []
upsc_chunks, upsc_embeddings = [], []

if os.path.exists(NCERT_DIR):
    ncert_chunks, ncert_embeddings = prepare_corpus(NCERT_DIR)
if os.path.exists(UPSC_DIR):
    upsc_chunks, upsc_embeddings = prepare_corpus(UPSC_DIR)

relevant_chunks = get_relevant_chunks(ncert_chunks, topic)
max_possible = count_possible_mcqs(relevant_chunks)

st.write(f"📄 NCERT PDFs detected: {len(list(Path(NCERT_DIR).rglob('*.pdf')))}")
st.write(f"🧩 Total NCERT chunks: {len(ncert_chunks)}")
st.write(f"🔍 Relevant NCERT chunks found: {len(relevant_chunks)}")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Subjective (NCERT/UPSC)", "🧠 MCQs (NCERT + UPSC)", "💬 Ask Anything (Chatbot)"])

# -------------------------------
# SUBJECTIVE TAB
# -------------------------------
with tab1:
    standard = st.radio("Select Question Standard", ["NCERT", "UPSC"])
    
    if st.button("Generate Subjective Questions"):
        if not topic.strip():
            st.error("Please enter a topic first.")
        else:
            chunks_source = ncert_chunks if standard=="NCERT" else upsc_chunks
            relevant_chunks = get_relevant_chunks(chunks_source, topic)
            max_possible = count_possible_mcqs(relevant_chunks)
            final_n = min(num_q, max_possible)
            if max_possible == 0:
                st.error("No meaningful content found for this topic.")
            else:
                st.info(f"📊 {max_possible} meaningful questions possible. Showing {final_n}.")
                qs = generate_subjective(topic, final_n, standard=standard)
                for i, q in enumerate(qs, 1):
                    st.write(f"{i}. {q}")

# -------------------------------
# MCQ TAB
# -------------------------------
with tab2:
    mcq_type = st.radio("MCQ Type", ["NCERT MCQs", "UPSC PYQ – Statements", "UPSC PYQ – Assertion Reason"])
    
    if st.button("Generate MCQs"):
        if not topic.strip():
            st.error("Please enter a topic first.")
        else:
            chunks_source = ncert_chunks if mcq_type=="NCERT MCQs" else upsc_chunks
            relevant_chunks = get_relevant_chunks(chunks_source, topic)
            max_possible = count_possible_mcqs(relevant_chunks)
            final_n = min(num_q, max_possible)
            if max_possible == 0:
                st.error("❌ No meaningful content found for this topic.")
            else:
                st.info(f"📊 {max_possible} meaningful MCQs possible. Showing {final_n}.")
                
                if mcq_type=="NCERT MCQs":
                    mcqs = generate_ncert_mcqs_short(relevant_chunks, topic, final_n)
                    for i, m in enumerate(mcqs, 1):
                        st.write(f"**Q{i}. {m['q']}**")
                        for j, opt in enumerate(m["options"]):
                            st.write(f"{chr(97+j)}) {opt}")
                        st.write(f"✅ Answer: {chr(97 + m['answer'])}")
                        st.write("---")
                elif mcq_type=="UPSC PYQ – Statements":
                    qs = generate_upsc_statements(topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. With reference to {topic}, consider the following statements:**")
                        for idx, s in enumerate(q["statements"], 1):
                            st.write(f"{idx}. {s}")
                        st.write("Which of the statements given above are correct?")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")
                else:
                    qs = generate_assertion_reason_short(relevant_chunks, topic, final_n)
                    for i, q in enumerate(qs, 1):
                        st.write(f"**Q{i}. Assertion (A):** {q['A']}")
                        st.write(f"**Reason (R):** {q['R']}")
                        st.write("a) Both A and R are true and R is the correct explanation of A")
                        st.write("b) Both A and R are true but R is not the correct explanation of A")
                        st.write("c) A is true but R is false")
                        st.write("d) A is false but R is true")
                        st.write(f"✅ Answer: {q['answer']}")
                        st.write("---")

# -------------------------------
# CHATBOT TAB
# -------------------------------
with tab3:
    source = st.radio("Select Source for Chatbot", ["NCERT", "UPSC"])
    user_query = st.text_input("Ask a question strictly from the selected source:")
    
    if st.button("Get Answer") and user_query.strip():
        chunks_source = ncert_chunks if source=="NCERT" else upsc_chunks
        embeddings_source = ncert_embeddings if source=="NCERT" else upsc_embeddings
        if not chunks_source:
            st.error(f"No {source} content loaded. Please load PDFs first.")
        else:
            answer = chatbot_answer(user_query, chunks_source, embeddings_source)
            st.markdown(f"📘 **{source}-based answer:**\n\n{answer}")
