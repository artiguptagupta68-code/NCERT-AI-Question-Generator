# app.py
import os
import zipfile
import shutil
from pathlib import Path
import re

import streamlit as st
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = os.getenv("NCERT_DRIVE_FILE_ID", "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G")
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

st.set_page_config(page_title="NCERT → UPSC Question Generator", layout="wide")
st.title("NCERT → UPSC Question Generator (Offline, Rule-Based)")

# ----------------------------
# Utilities
# ----------------------------
def download_drive_file(file_id: str, out_path: str) -> bool:
    if os.path.exists(out_path):
        return True
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception as e:
        st.warning(f"Download failed: {e}")
        return False

def extract_zip_with_nested(src_zip_path: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(src_zip_path, "r") as z:
            z.extractall(dest_dir)
    except Exception as e:
        st.error(f"Zip extraction failed: {e}")
        return []
    # Handle nested zips
    found_any = True
    while found_any:
        found_any = False
        for root, _, files in os.walk(dest_dir):
            for f in list(files):
                if f.lower().endswith(".zip"):
                    found_any = True
                    nested_zip = os.path.join(root, f)
                    folder_target = os.path.join(root, Path(f).stem)
                    os.makedirs(folder_target, exist_ok=True)
                    try:
                        with zipfile.ZipFile(nested_zip, "r") as nz:
                            nz.extractall(folder_target)
                    except Exception:
                        pass
                    try:
                        os.remove(nested_zip)
                    except Exception:
                        pass
    return sorted(os.listdir(dest_dir))

def read_pdf(path: str) -> str:
    try:
        r = PdfReader(path)
        text = ""
        for p in r.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
        return text
    except:
        return ""

def clean_text(text: str) -> str:
    patterns = [
        r"Reprint\s*\d{4}-\d{2}",
        r"ISBN[\s:0-9-]+",
        r"©[\s\S]{0,80}",
        r"Not for commercial use",
        r"Publication Division",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    step = chunk_size - overlap
    i = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk.split()) > 30:
            chunks.append({"chunk_id": f"chunk_{i}", "text": chunk})
            i += 1
        start += step
    return chunks

# ----------------------------
# Build FAISS Index
# ----------------------------
@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_resource
def build_faiss_index(chunks):
    model = load_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, index, model, chunks, top_k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, min(top_k, index.ntotal))
    results = [chunks[i]["text"] for i in I[0]]
    return results

# ----------------------------
# Rule-based Question Generation
# ----------------------------
INTERROGATIVES = ["Explain", "Describe", "Discuss", "State", "Define", "Analyse", "Examine", "Why", "How", "Compare"]

def generate_subjective_questions(text_chunks, n=5):
    questions = []
    for chunk in text_chunks:
        sentences = re.split(r'(?<=[\.\?])\s+', chunk)
        for s in sentences:
            if len(s.split()) > 10:
                q_start = INTERROGATIVES[np.random.randint(len(INTERROGATIVES))]
                question = f"{q_start} the following: {s.strip()} ?"
                if question not in questions:
                    questions.append(question)
            if len(questions) >= n:
                break
        if len(questions) >= n:
            break
    return questions

# ----------------------------
# Streamlit UI
# ----------------------------
with st.sidebar:
    use_drive = st.checkbox("Auto-download NCERT ZIP from Drive", value=True)
    uploaded = st.file_uploader("Or upload NCERT ZIP file", type="zip")

if uploaded:
    with open(ZIP_PATH, "wb") as f:
        f.write(uploaded.getbuffer())
elif use_drive:
    ok = download_drive_file(FILE_ID, ZIP_PATH)
    if not ok:
        st.error("Failed to download ZIP. Upload manually or check FILE_ID.")
        st.stop()
else:
    st.info("Upload a ZIP file or enable auto-download.")
    st.stop()

# Extract ZIP
with st.spinner("Extracting NCERT ZIP..."):
    extract_zip_with_nested(ZIP_PATH, EXTRACT_DIR)
st.success("Extraction done.")

# List PDFs
pdf_files = list(Path(EXTRACT_DIR).rglob("*.pdf"))
if not pdf_files:
    st.error("No PDFs found in extracted folder.")
    st.stop()

subject = st.selectbox("Select Subject", SUBJECTS)
docs = []
for pdf in pdf_files:
    if subject.lower() in pdf.stem.lower() or subject.lower() in str(pdf.parent).lower():
        text = read_pdf(str(pdf))
        if text.strip():
            docs.append({"doc_id": pdf.name, "text": clean_text(text)})

if not docs:
    st.warning(f"No readable PDFs found for {subject}.")
    st.stop()

# Chunk
all_chunks = []
for doc in docs:
    all_chunks.extend(chunk_text(doc["text"]))

if not all_chunks:
    st.error("No chunks created.")
    st.stop()

# Build FAISS
with st.spinner("Building FAISS index..."):
    embed_model = load_embedding_model()
    index, embeddings = build_faiss_index(all_chunks)
st.success("FAISS index ready.")

# User Input
st.subheader("Generate NCERT Subjective Questions")
topic = st.text_input("Enter chapter/topic (example: 'Constitution')")
num_q = st.number_input("Number of questions to generate", min_value=1, max_value=20, value=5)

if st.button("Generate Questions"):
    if not topic:
        st.warning("Enter a topic.")
    else:
        relevant_chunks = retrieve_chunks(topic, index, embed_model, all_chunks, top_k=TOP_K)
        questions = generate_subjective_questions(relevant_chunks, n=num_q)
        st.success(f"Generated {len(questions)} subjective questions")
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
