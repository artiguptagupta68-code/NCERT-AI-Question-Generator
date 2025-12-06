# app.py
import os
import zipfile
import shutil
from pathlib import Path
import streamlit as st
import gdown
import numpy as np
import torch
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
DEFAULT_TOP_K = 5

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Enter any topic; the app will detect the most relevant subject folder and generate long subjective questions.")

# ----------------------------
# Utilities: download & unzip
# ----------------------------
def download_zip_from_drive(file_id: str, out_path: str) -> bool:
    if os.path.exists(out_path):
        return True
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception as e:
        st.warning(f"Download failed: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    shutil.rmtree(extract_to, ignore_errors=True)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    # handle nested zips
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith(".zip"):
                nested = os.path.join(root, f)
                nested_dir = os.path.join(root, Path(f).stem)
                os.makedirs(nested_dir, exist_ok=True)
                try:
                    with zipfile.ZipFile(nested, "r") as nz:
                        nz.extractall(nested_dir)
                except Exception:
                    pass

# ----------------------------
# PDF reading
# ----------------------------
def read_pdf_pypdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            try:
                t = p.extract_text()
                if t:
                    text += t + "\n"
            except Exception:
                continue
        return text
    except Exception:
        return ""

def read_pdf_pymupdf(path):
    if not _HAS_PYMUPDF:
        return ""
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            try:
                t = page.get_text()
                if t:
                    text += t + "\n"
            except Exception:
                continue
        return text
    except Exception:
        return ""

def read_pdf_text(path):
    text = read_pdf_pypdf(path)
    if text.strip():
        return text
    if _HAS_PYMUPDF:
        text = read_pdf_pymupdf(path)
        if text.strip():
            return text
    return ""

# ----------------------------
# Load and chunk PDFs
# ----------------------------
def load_docs_from_folder(folder):
    docs = []
    if not os.path.exists(folder):
        return docs
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                text = read_pdf_text(os.path.join(root, f))
                if text.strip():
                    docs.append({"doc_id": f, "text": text})
    return docs

def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, p in enumerate(parts):
            all_chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{Path(doc['doc_id']).stem}_chunk_{i}",
                "text": p
            })
    return all_chunks

# ----------------------------
# Build FAISS index
# ----------------------------
@st.cache_resource
def build_faiss_index(chunks):
    if not chunks:
        return None, None, None
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    return model, index, metadata

# ----------------------------
# Generator pipeline
# ----------------------------
@st.cache_resource
def load_generator_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    if device == 0:
        model = model.to("cuda")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# ----------------------------
# Dynamic subject detection
# ----------------------------
def detect_best_subject_folder(base_dir, topic, min_matches=1):
    topic_words = re.findall(r"\w+", topic.lower())
    if not topic_words:
        return None
    best_folder, best_score = None, 0
    candidate_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    candidate_folders.append(base_dir)  # fallback
    for folder in candidate_folders:
        score, scanned = 0, 0
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith(".pdf"):
                    continue
                text = read_pdf_text(os.path.join(root, f)).lower()
                for w in topic_words:
                    score += text.count(w)
                scanned += 1
                if scanned >= 5:
                    break
            if scanned >= 5:
                break
        if score > best_score:
            best_score = score
            best_folder = folder
    return best_folder if best_score >= min_matches else None

# ----------------------------
# Retrieve chunks
# ----------------------------
def retrieve_chunks(query, index, metadata, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    retrieved = [metadata[i] for i in indices[0] if i < len(metadata)]
    return retrieved

# ----------------------------
# Build prompt
# ----------------------------
def build_question_prompt(retrieved_chunks, topic, num_questions, max_context_chars=3000):
    ctx, total = [], 0
    for r in retrieved_chunks:
        t = r["text"].strip()
        if not t:
            continue
        remaining = max_context_chars - total
        if remaining <= 0:
            break
        ctx.append(t[:remaining])
        total += len(t)
    context = "\n\n".join(ctx)
    return (
        f"You are an expert NCERT question generator. Based ONLY on the NCERT context below, "
        f"generate {num_questions} HIGH-QUALITY long subjective questions on the topic '{topic}'. "
        "Each question must be at least 3-4 lines long, exam-style, concept-rich, and do NOT include answers. "
        "Do not invent facts beyond the provided context.\n\n"
        f"NCERT Context:\n{context}\n\n"
        "Generate the questions now:"
    )

# ----------------------------
# Orchestration & UI
# ----------------------------
st.write("Preparing NCERT content (may take a while on first run)...")

# Download and extract ZIP
if not download_zip_from_drive(FILE_ID, ZIP_PATH):
    st.error("Failed to download ZIP. Check FILE_ID or permissions.")
    st.stop()
if not os.path.exists(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):
    extract_zip(ZIP_PATH, EXTRACT_DIR)
    st.success(f"ZIP extracted to: {EXTRACT_DIR}")

# User input
topic = st.text_input("Enter chapter name or topic (e.g., 'Constitutional Design', 'Electricity'):")
num_questions = st.slider("Number of long subjective questions", 1, 10, 5)
top_k = st.number_input("Number of chunks to use as context", 1, 20, DEFAULT_TOP_K)

if st.button("Generate Questions") and topic.strip():
    with st.spinner("Detecting subject folder for this topic..."):
        folder = detect_best_subject_folder(EXTRACT_DIR, topic)
    if folder:
        st.success(f"Selected folder: {os.path.relpath(folder, EXTRACT_DIR)}")
    else:
        st.info("No exact folder match. Using full corpus.")
        folder = EXTRACT_DIR

    with st.spinner("Loading and chunking PDFs..."):
        docs = load_docs_from_folder(folder)
        if not docs:
            st.error("No readable PDFs found.")
            st.stop()
        chunks = split_documents(docs)
        st.info(f"Created {len(chunks)} chunks.")

    st.session_state["_faiss_chunks_temp"] = chunks
    with st.spinner("Building embeddings & FAISS index..."):
        _, index, metadata = build_faiss_index(chunks)
    st.session_state.pop("_faiss_chunks_temp", None)
    st.success("FAISS index ready.")

    with st.spinner("Retrieving relevant chunks..."):
        retrieved = retrieve_chunks(topic, index, metadata, top_k=top_k)
    if not retrieved:
        st.warning("No relevant content found.")
        st.stop()

    generator = load_generator_pipeline()
    prompt = build_question_prompt(retrieved, topic, num_questions)

    with st.spinner("Generating long subjective questions..."):
        try:
            output = generator(prompt, max_length=600, do_sample=False)[0]["generated_text"]
        except Exception as e:
            st.error(f"Generation failed: {e}")
            output = ""

    if output:
        st.write("### Generated Questions")
        st.markdown(output)
    else:
        st.error("No output produced.")

    st.write("### Sources used")
    for r in retrieved:
        st.write(f"{r['doc_id']} — {r['chunk_id']}")
