# app.py
import os
import zipfile
import shutil
from pathlib import Path
import streamlit as st
import gdown
import numpy as np
import torch

from pypdf import PdfReader
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # your Drive file id
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small & fast
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Generates NCERT-style subjective questions from topic/chapter (RAG + Transformers)")

# ----------------------------
# Utilities: download, unzip, nested zips
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
                    # ignore nested extract failures
                    pass

# ----------------------------
# Read PDF text robustly
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
    # try pypdf first
    text = read_pdf_pypdf(path)
    if text and text.strip():
        return text
    # fallback to pymupdf if available
    if _HAS_PYMUPDF:
        text = read_pdf_pymupdf(path)
        if text and text.strip():
            return text
    return ""  # empty if cannot extract

# ----------------------------
# Load and chunk documents
# ----------------------------
def load_docs_from_folder(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                p = os.path.join(root, f)
                text = read_pdf_text(p)
                if text and text.strip():
                    docs.append({"doc_id": f, "text": text})
                else:
                    st.warning(f"Unreadable or image-only PDF skipped: {f}")
    return docs

def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        text = doc.get("text", "")
        if not text.strip():
            continue
        parts = splitter.split_text(text)
        for i, p in enumerate(parts):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{Path(doc_id).stem}_chunk_{i}",
                "text": p
            })
    return all_chunks

# ----------------------------
# Embeddings + FAISS build (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_faiss_index(chunks):
    if not chunks:
        return None, None, None
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    return model, index, metadata

# ----------------------------
# Load generator (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_generator_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    if device == 0:
        model = model.to("cuda")
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen

# ----------------------------
# Retrieval and prompt building
# ----------------------------
def retrieve_chunks(query, index, metadata, top_k=TOP_K):
    if index is None or metadata is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, index.ntotal if hasattr(index, "ntotal") else index.ntotal if hasattr(index, "ntotal") else top_k)
    if k <= 0:
        return []
    D, I = index.search(q_emb, k)
    res = []
    for idx in I[0]:
        if idx < len(metadata):
            res.append(metadata[idx])
    return res

def build_question_prompt(retrieved_chunks, topic, max_context_chars=3000):
    ctx_parts = []
    total = 0
    for r in retrieved_chunks:
        t = r.get("text", "").strip()
        if not t:
            continue
        remaining = max_context_chars - total
        if remaining <= 0:
            break
        if len(t) > remaining:
            t = t[:remaining]
        ctx_parts.append(t)
        total += len(t)
    context = "\n\n".join(ctx_parts)
    prompt = (
        "You are an expert NCERT question generator.\n"
"Based ONLY on the following NCERT context, generate:\n"
"- 5 Long Subjective Questions\n"
"Each question MUST be at least 4-5 lines long, descriptive, and based fully on NCERT.\n\n"
f"Topic: {topic}\n\n"
f"NCERT Context:\n{context}\n\n"
"Generate detailed exam-style subjective questions:"

    )
    return prompt

# ----------------------------
# Orchestration
# ----------------------------
# 1) Download & extract if needed
st.text("Preparing NCERT content...")
ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Failed to download ZIP from Google Drive. Please check FILE_ID and permissions or upload ZIP to backend.")
    st.stop()

if not zipfile.is_zipfile(ZIP_PATH):
    st.error(f"{ZIP_PATH} is not a valid ZIP file. Check Google Drive link or permissions.")
    st.stop()

# extract
extract_zip(ZIP_PATH, EXTRACT_DIR)
st.success(f"ZIP extracted to: {EXTRACT_DIR}")

# 2) Load documents
docs = load_docs_from_folder(EXTRACT_DIR)
st.info(f"Loaded {len(docs)} readable PDF documents.")

if not docs:
    st.error("No readable PDF text found. If PDFs are scanned images, OCR is required (not included).")
    st.stop()

# 3) Split into chunks and cache in session
all_chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
st.info(f"Total chunks created: {len(all_chunks)}")

if not all_chunks:
    st.error("No chunks created from PDFs. Cannot proceed.")
    st.stop()

# save into session for later use
st.session_state['all_chunks'] = all_chunks

# 4) Build embeddings + FAISS
embed_model, index, metadata = build_faiss_index(all_chunks)
if index is None:
    st.error("Failed to build FAISS index (no embeddings).")
    st.stop()
st.success("FAISS index built.")

# expose embed_model to retrieval function scope
# (embed_model variable used in retrieve_chunks)
# assign to a module-level name used by functions
embed_model = embed_model  # already returned

# 5) Load generator
generator = load_generator_pipeline()
st.success("Generator model loaded.")

# 6) UI: topic input -> generate subjective questions
st.subheader("Generate NCERT Subjective Questions")
topic = st.text_input("Enter chapter name or topic (example: 'Constitutional Design', 'Electricity', 'Reproduction'):")

if topic:
    with st.spinner("Retrieving relevant NCERT text..."):
        retrieved = retrieve_chunks(topic, index, metadata, top_k=TOP_K)
    if not retrieved:
        st.warning("No relevant NCERT content found for that topic. Try a different keyword (e.g., 'constitution', 'democracy').")
    else:
        with st.spinner("Generating questions..."):
            prompt = build_question_prompt(retrieved, topic)
            try:
                out = generator(prompt, max_length=400, do_sample=False)[0]["generated_text"]
            except Exception as e:
                st.error(f"Generation failed: {e}")
                out = ""
        st.write("### Generated Long Subjective Questions")
        st.write(out)

        st.write("### Sources used")
        for r in retrieved:
            st.write(f"{r['doc_id']} — {r['chunk_id']}")

# End of app.py
