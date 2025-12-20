# ===============================
# NCERT RAG System (UPSC Grade)
# ===============================

import os, zipfile, re, pickle
from pathlib import Path

import streamlit as st
import gdown
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------------------------------
# CONFIG
# -------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

VECTOR_DIR = "vectorstore"
FAISS_PATH = f"{VECTOR_DIR}/ncert.index"
META_PATH = f"{VECTOR_DIR}/ncert_meta.pkl"

os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------
# STREAMLIT
# -------------------------------
st.set_page_config("Ask NCERT", layout="wide")
st.title("📘 Ask Anything from NCERT (UPSC-Safe)")

# -------------------------------
# MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# -------------------------------
# PDF UTILITIES
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            ZIP_PATH,
            quiet=False
        )

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            out = zfile.parent / zfile.stem
            out.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(out)
        except:
            pass

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(reprint|isbn|copyright).*", "", text, flags=re.I)
    return text.strip()

# -------------------------------
# CHUNKING
# -------------------------------
def semantic_chunks(text, size=3):
    sentences = re.split(r'(?<=[.])\s+', text)
    return [" ".join(sentences[i:i+size]) for i in range(0, len(sentences), size)]

# -------------------------------
# BUILD / LOAD VECTORSTORE
# -------------------------------
def load_or_build_index():
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(pdf))
        if len(t.split()) > 100:
            texts.append(t)

    chunks = []
    for t in texts:
        chunks.extend(semantic_chunks(t))

    embeddings = embedder.encode(chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve(query, k=6):
    q_emb = embedder.encode([query])
    _, idx = st.session_state.index.search(q_emb, k)
    return [st.session_state.chunks[i] for i in idx[0]]

# -------------------------------
# SESSION STATE
# -------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    if st.button("📥 Load & Index NCERT"):
        with st.spinner("Indexing NCERT (one-time)..."):
            download_and_extract()
            idx, ch = load_or_build_index()
            st.session_state.index = idx
            st.session_state.chunks = ch
        st.success("✅ NCERT Ready")

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📝 Subjective", "🧠 MCQs / AR", "💬 Ask NCERT"])

# -------------------------------
# SUBJECTIVE
# -------------------------------
with tab1:
    topic = st.text_input("Topic (NCERT only)")
    if st.button("Generate Subjective"):
        ctx = retrieve(topic)
        st.write("### NCERT-Based Question")
        st.write(f"Explain **{topic}** with reference to NCERT.")
        st.write("#### Key Points (from NCERT):")
        for c in ctx:
            st.write("- ", c)

# -------------------------------
# MCQs / ASSERTION
# -------------------------------
with tab2:
    topic = st.text_input("Topic for MCQs")
    if st.button("Generate MCQ"):
        ctx = retrieve(topic)
        if not ctx:
            st.error("No NCERT content found.")
        else:
            st.write("**Assertion (A):**", ctx[0])
            st.write("**Reason (R):**", ctx[1] if len(ctx) > 1 else ctx[0])
            st.write("""
a) Both A and R are true and R explains A  
b) Both true but R not explanation  
c) A true, R false  
d) A false, R true
""")

# -------------------------------
# CHATBOT (STRICT NCERT)
# -------------------------------
with tab3:
    q = st.text_input("Ask anything strictly from NCERT")
    if st.button("Ask"):
        ctx = retrieve(q)
        if not ctx:
            st.error("Answer not found in NCERT.")
        else:
            st.write("📘 **NCERT-based answer:**")
            for c in ctx:
                st.write(c)
