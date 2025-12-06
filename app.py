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

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # Drive file id containing subject folders / zips
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
DEFAULT_TOP_K = 5

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Select a topic; the app will pick the most relevant subject folder and generate long subjective questions.")

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
    text = read_pdf_pypdf(path)
    if text and text.strip():
        return text
    if _HAS_PYMUPDF:
        text = read_pdf_pymupdf(path)
        if text and text.strip():
            return text
    return ""

# ----------------------------
# Load & chunk documents for a given folder
# ----------------------------
def load_docs_from_subject_folder(folder):
    docs = []
    if not os.path.exists(folder):
        return docs
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                p = os.path.join(root, f)
                text = read_pdf_text(p)
                if text and text.strip():
                    docs.append({"doc_id": f, "text": text})
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
# Embeddings + FAISS (cached per subject key)
# ----------------------------
@st.cache_resource
def build_faiss_index_for_chunks(chunks_tuple):
    # streamlit caching forces hashable args; we pass a tuple of (subject_key, num_chunks) or store in session_state
    chunks = st.session_state.get("_faiss_chunks_temp", [])
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
# Generator cache
# ----------------------------
@st.cache_resource
def load_generator_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    if device == 0:
        model = model.to("cuda")
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen

# ----------------------------
# Subject detection: pick best folder that matches topic
# ----------------------------
def detect_best_subject_folder(base_dir, topic, min_matches=1):
    """
    Strategy:
      - For each immediate subfolder of base_dir, count occurrences of topic keywords in its PDFs.
      - Pick folder with highest count. If none found, return None.
    """
    topic_norm = topic.lower().strip()
    words = re.findall(r"\w+", topic_norm)
    if not words:
        return None

    best_folder = None
    best_score = 0

    # consider immediate subfolders and zip-extracted folders
    candidate_folders = []
    # immediate subfolders
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path):
            candidate_folders.append(path)
    # also include base_dir itself
    candidate_folders.append(base_dir)

    for folder in candidate_folders:
        score = 0
        # scan each pdf in folder but limit to first few files for speed
        scanned = 0
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith(".pdf"):
                    continue
                scanned += 1
                p = os.path.join(root, f)
                text = read_pdf_text(p)
                t_lower = text.lower()
                # count word matches
                for w in words:
                    score += t_lower.count(w)
                if scanned >= 5:  # only scan first 5 pdfs per folder for speed
                    break
            if scanned >= 5:
                break
        if score > best_score:
            best_score = score
            best_folder = folder

    if best_score >= min_matches:
        return best_folder
    return None

# ----------------------------
# Build prompt for question generation
# ----------------------------
def build_question_prompt(retrieved_chunks, topic, num_questions, max_context_chars=3000):
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
        "You are an expert NCERT question generator. Based ONLY on the NCERT context below, "
        f"generate {num_questions} HIGH-QUALITY long subjective questions on the topic '{topic}'. "
        "Each question must be at least 3-4 lines long (exam-style, concept-rich), and should NOT include answers. "
        "Do not invent facts beyond the provided context. Produce numbered questions only.\n\n"
        f"NCERT Context:\n{context}\n\n"
        "Generate the questions now:"
    )
    return prompt

# ----------------------------
# Orchestration & UI
# ----------------------------
st.write("Preparing NCERT content (this may take a while on first run)...")

ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Failed to download ZIP from Google Drive. Check FILE_ID/permissions or upload ZIP to backend.")
    st.stop()

if not zipfile.is_zipfile(ZIP_PATH):
    st.error("Downloaded file is not a valid ZIP. Check the file on Drive.")
    st.stop()

# extract once
if not os.path.exists(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):
    extract_zip(ZIP_PATH, EXTRACT_DIR)
    st.success(f"ZIP extracted to: {EXTRACT_DIR}")

# allow user to optionally select a subject folder from detected folders
subfolders = [os.path.join(EXTRACT_DIR, d) for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d))]
subfolders_display = [os.path.relpath(p, EXTRACT_DIR) for p in subfolders]
if not subfolders:
    st.info("No subfolders detected in extracted ZIP — searching whole corpus.")
else:
    st.sidebar.header("Detected subject folders")
    st.sidebar.write(subfolders_display)

# user inputs
topic = st.text_input("Enter chapter name or topic (e.g., 'Constitutional Design', 'Electricity', 'Reproduction'):")
num_questions = st.slider("Number of long subjective questions to generate", min_value=1, max_value=10, value=5)
top_k = st.number_input("Retrieval: number of chunks to use as context", min_value=1, max_value=20, value=DEFAULT_TOP_K)

if st.button("Generate Questions") and topic.strip():
    # detect best subject folder automatically
    with st.spinner("Detecting best subject folder for this topic..."):
        best_folder = detect_best_subject_folder(EXTRACT_DIR, topic, min_matches=1)
    if best_folder:
        st.success(f"Selected subject folder: {os.path.relpath(best_folder, EXTRACT_DIR)}")
    else:
        st.info("No single subject folder strongly matched the topic. Using the entire extracted corpus.")
        best_folder = EXTRACT_DIR

    # load docs only from selected folder
    with st.spinner("Loading and chunking PDFs from the selected subject..."):
        docs = load_docs_from_subject_folder(best_folder)
        st.info(f"Loaded {len(docs)} readable documents from selected folder.")
        if not docs:
            st.error("No readable documents found in the selected folder. Try another topic or fix the PDFs.")
            st.stop()
        chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        st.info(f"Created {len(chunks)} chunks from selected subject.")
        if not chunks:
            st.error("No chunks created. Cannot proceed.")
            st.stop()

    # store chunks in session temporarily (for the cached builder)
    st.session_state['_faiss_chunks_temp'] = chunks

    # build faiss index for these chunks (cached)
    with st.spinner("Building embeddings & FAISS index for this subject (cached)..."):
        embed_model_local, index_local, metadata_local = build_faiss_index_for_chunks("subject_build_key")
    # clear temp
    st.session_state.pop('_faiss_chunks_temp', None)

    if index_local is None:
        st.error("Failed to build FAISS index.")
        st.stop()
    st.success("FAISS index ready for this subject.")

    # retrieve chunks for the topic (using same embed model)
    with st.spinner("Retrieving relevant chunks..."):
        retrieved = retrieve_chunks(topic, index_local, metadata_local, top_k=top_k)
    if not retrieved:
        st.warning("No relevant NCERT content found for that topic in the chosen subject. Try different keyword or increase retrieval chunks.")
    else:
        st.info(f"Retrieved {len(retrieved)} chunks as context.")

        # load generator
        generator = load_generator_pipeline()

    def retrieve_chunks(query, index, metadata, top_k=5):
    """Retrieve the top_k most relevant chunks using FAISS search."""

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Encode the query text
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

    # Search in FAISS
    distances, indices = index.search(query_vec, top_k)

    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            retrieved.append(metadata[idx])

    return retrieved



def retrieve_chunks(query, index, metadata, top_k=5):
    """
    Retrieve the top_k most relevant chunks based on FAISS similarity search.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Encode query text
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

    # Search FAISS index
    distances, indices = index.search(query_vec, top_k)

    retrieved = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            retrieved.append(metadata[idx])

    return retrieved
       
# build prompt and generate questions
    prompt = build_question_prompt(retrieved, topic, num_questions)
    with st.spinner("Generating long subjective questions..."):
        try:
            output = generator(prompt, max_length=600, do_sample=False)[0]["generated_text"]
        except Exception as e:
            st.error(f"Generation failed: {e}")
            output = ""
            if output:
                st.write("### Generated Long Subjective Questions")
                st.markdown(output)
            else:
                st.error("No output produced by the generator.")
                st.write("### Sources used (file — chunk_id)")
                for r in retrieved:
                    st.write(f"{r['doc_id']} — {r['chunk_id']}")

# end of app.py
