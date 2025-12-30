import os, zipfile, re, random
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

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT AI Generator", layout="wide")
st.title("📘 NCERT + UPSC AI Question Generator")

# -------------------------------
# LOAD EMBEDDER
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedder = load_embedder()

# -------------------------------
# PDF DOWNLOAD & EXTRACTION
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    st.success("✅ NCERT PDFs extracted!")

# -------------------------------
# READ & CLEAN PDF
# -------------------------------
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

def load_all_texts():
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
    return texts

# -------------------------------
# SEMANTIC CHUNKING
# -------------------------------
def semantic_chunking(text, max_words=180, sim_threshold=0.65):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]
    if len(sentences) < 2:
        return sentences

    embeddings_arr = embedder.encode(sentences, convert_to_numpy=True)
    chunks = []
    current = [sentences[0]]
    current_emb = embeddings_arr[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_emb], [embeddings_arr[i]])[0][0]
        length = sum(len(s.split()) for s in current)
        if sim < sim_threshold or length > max_words:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_emb = embeddings_arr[i]
        else:
            current.append(sentences[i])
            current_emb = np.mean([current_emb, embeddings_arr[i]], axis=0)
    if current:
        chunks.append(" ".join(current))
    return chunks

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    if not chunks:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode=="UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    retrieved = [c for c, s in ranked if s >= threshold][:top_k]
    st.write(f"🔹 Retrieved {len(retrieved)} chunks for query: '{query}'")
    return retrieved

# -------------------------------
# FLASHCARDS
# -------------------------------
def generate_flashcard(chunks, topic):
    if not chunks:
        return []
    sentences = []
    for ch in chunks:
        parts = re.split(r'(?<=[.?!])\s+', ch)
        sentences.extend([p.strip() for p in parts if len(p.split()) > 8])
    if not sentences:
        return []
    flashcard = {
        "title": topic.capitalize(),
        "content": "\n".join(sentences[:10])
    }
    return [flashcard]

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs", key="load_pdfs"):
        download_and_extract()
        st.session_state['texts'] = load_all_texts()
        st.success(f"Loaded {len(st.session_state['texts'])} PDFs")

# -------------------------------
# MAIN APP
# -------------------------------
subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# Load PDFs & create chunks
if 'texts' not in st.session_state:
    st.session_state['texts'] = []
    st.session_state['chunks'] = []
    st.session_state['embeddings'] = np.empty((0, 384))

if st.session_state['texts']:
    chunks = []
    for t in st.session_state['texts']:
        chunks.extend(semantic_chunking(t))
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    st.session_state['chunks'] = chunks
    st.session_state['embeddings'] = embeddings
    st.write(f"🧩 Created {len(chunks)} chunks from {len(st.session_state['texts'])} PDFs")
else:
    st.warning("❌ No PDF content loaded. Use sidebar to load NCERT PDFs.")

# -------------------------------
# FLASHCARD TAB
# -------------------------------
if topic.strip() and st.session_state['chunks']:
    rel = retrieve_relevant_chunks(st.session_state['chunks'], st.session_state['embeddings'], topic, mode="NCERT")
    cards = generate_flashcard(rel, topic)
    if cards:
        c = cards[0]
        st.markdown(f"### 📌 {c['title']}")
        st.write(c["content"])
    else:
        st.warning("No relevant content found to generate flashcard.")
