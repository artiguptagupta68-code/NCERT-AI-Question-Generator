import os
import zipfile
import re
import random
from pathlib import Path
import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# CONFIG
# -------------------------------
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6
SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45
EPOCHS = 15
BATCH_SIZE = 16

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT AI Generator", layout="wide")
st.title("📘 NCERT + UPSC AI Question Generator")

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if 'texts' not in st.session_state:
    st.session_state['texts'] = []
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = np.empty((0, 384))

# -------------------------------
# LOAD EMBEDDER
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedder = load_embedder()

# -------------------------------
# UTILITIES
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    # Extract main zip
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    
    # Extract nested zips if any
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except:
            pass
    
    st.success("✅ NCERT PDFs extracted!")

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

def semantic_chunking(text):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]
    return sentences  # simple chunking, can be enhanced

def embed_chunks(chunks):
    if not chunks:
        return np.empty((0, 384))
    return embedder.encode(chunks, convert_to_numpy=True)

def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    if not chunks.any():
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode == "UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:top_k]

def normalize_text(s):
    s = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().capitalize()

def is_conceptual(s):
    s = s.strip()
    if len(s.split()) < 8:
        return False
    skip = ["phone", "fax", "isbn", "copyright", "page", "address", 
            "reprint", "pd", "bs", "ncertain", "office", "publication division"]
    if any(k in s.lower() for k in skip):
        return False
    keywords = ["right", "law", "constitution", "governance", "democracy",
                "citizen", "freedom", "justice", "equality", "policy"]
    return any(k in s.lower() for k in keywords)

def generate_flashcard(chunks, topic):
    sentences = []
    for ch in chunks:
        parts = re.split(r'(?<=[.?!])\s+', ch)
        for p in parts:
            if len(p.split()) > 8:
                sentences.append(p.strip())
    if not sentences:
        return []
    concept_overview = sentences[0]
    explanation = " ".join(sentences[1:6])
    classification = "The concept relates to democracy, rights, rule of law, and citizen-state relations."
    conclusion = "Overall, the Constitution provides a framework for governance and ensures justice, equality, and accountability."
    points = [" ".join(s.split()[:20]) for s in sentences[1:6]]
    flashcard = {
        "title": topic.capitalize(),
        "content": f"""
Concept Overview:
{concept_overview}

Explanation:
{explanation}

Classification / Types:
{classification}

Conclusion:
{conclusion}

Points to Remember:
- {"\n- ".join(points)}
"""
    }
    return [flashcard]

# -------------------------------
# GLOBALS
# -------------------------------
texts = []
chunks = []
embeddings = np.empty((0, 384))

with st.sidebar:
    if st.button("📥 Load NCERT PDFs", key="load_pdfs"):
        download_and_extract()
        texts = load_all_texts()
        if not texts:
            st.warning("No PDF content found! Check if PDFs are readable.")
        else:
            st.success(f"✅ Loaded {len(texts)} PDFs")
            chunks = [s for t in texts for s in semantic_chunking(t)]
            embeddings = embed_chunks(chunks)
            
            # Save in session_state
            st.session_state['texts'] = texts
            st.session_state['chunks'] = chunks
            st.session_state['embeddings'] = embeddings


# -------------------------------
# MAIN APP
# -------------------------------
subject = st.selectbox("Subject", SUBJECTS, key="subject_select")
topic = st.text_input("Topic", key="topic_input")
num_q = st.number_input("Number of Questions", 1, 10, 5, key="num_q_input")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Subjective", "🧠 MCQs", "💬 Chatbot", "🧠 Flashcards"])

# Subjective
with tab1:
    std1 = st.radio("Standard", ["NCERT", "UPSC"], key="std_sub", horizontal=True)
    if st.button("Generate Subjective", key="btn_subjective"):
        if not topic.strip():
            st.warning("Enter a topic first")
        else:
            rel = retrieve_relevant_chunks(np.array(chunks), embeddings, topic, std1)
            for i, q in enumerate([f"Explain {topic}.", f"Describe {topic}."][:len(rel)], 1):
                st.write(f"{i}. {q}")

# MCQs
with tab2:
    std2 = st.radio("Standard", ["NCERT", "UPSC"], key="std_mcq", horizontal=True)
    if st.button("Generate MCQs", key="btn_mcq"):
        if not topic.strip():
            st.warning("Enter a topic first")
        else:
            rel = retrieve_relevant_chunks(np.array(chunks), embeddings, topic, std2)
            for i, s in enumerate(rel[:num_q], 1):
                opts = random.sample(rel, min(4, len(rel)))
                st.write(f"Q{i}. {s}")
                for j, o in enumerate(opts):
                    st.write(f"{chr(97+j)}) {o}")

# Chatbot
with tab3:
    chatbot_mode = st.radio("Answer Style", ["NCERT", "UPSC"], key="std_chat", horizontal=True)
    user_q = st.text_input("Enter your question", key="user_question")
    if st.button("Ask NCERT", key="btn_chat"):
        if not user_q.strip():
            st.warning("Enter a question")
        else:
            rel = retrieve_relevant_chunks(np.array(chunks), embeddings, user_q, chatbot_mode)
            if not rel:
                st.warning("No answer found in NCERT")
            else:
                ans = " ".join([normalize_text(s) for s in rel[:5]])
                st.write(ans)

# Flashcards
with tab4:
    mode = st.radio("Depth", ["NCERT", "UPSC"], key="std_flash", horizontal=True)
    if st.button("Generate Flashcard", key="btn_flash"):
        if not topic.strip():
            st.warning("Enter a topic first")
        else:
            rel = retrieve_relevant_chunks(np.array(chunks), embeddings, topic, mode)
            cards = generate_flashcard(rel, topic)
            if cards:
                c = cards[0]
                st.markdown(f"### 📌 {c['title']}")
                st.write(c["content"])
            else:
                st.warning("No relevant content found to generate flashcard.")
