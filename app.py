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
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # Drive file id
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
DEFAULT_TOP_K = 5

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Choose a subject, enter a topic; app will use that subject's PDFs to generate long subjective questions.")

# ----------------------------
# Utilities
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
    # nested zips
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

def list_subfolders(folder):
    if not os.path.exists(folder):
        return []
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

# ----------------------------
# Load & chunk
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
    return docs

def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        text = doc.get("text", "")
        if not text.strip():
            continue
        parts = splitter.split_text(text)
        for i, p in enumerate(parts):
            out.append({"doc_id": doc_id, "chunk_id": f"{Path(doc_id).stem}_chunk_{i}", "text": p})
    return out

# ----------------------------
# Cached builders
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_faiss(chunks):
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

@st.cache_resource(show_spinner=True)
def load_generator():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    if device == 0:
        model = model.to("cuda")
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen

# ----------------------------
# Retrieval
# ----------------------------
def retrieve_chunks(query, index, metadata, top_k=DEFAULT_TOP_K):
    if index is None or metadata is None:
        return []
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    q_emb = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, index.ntotal) if hasattr(index, "ntotal") else top_k
    if k <= 0:
        return []
    D, I = index.search(q_emb, k)
    retrieved = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            retrieved.append(metadata[idx])
    return retrieved

# ----------------------------
# Prompt builder
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
# Prepare corpus
# ----------------------------
st.write("Preparing NCERT content (first run may take a while)...")
ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Failed to download ZIP from Google Drive. Check FILE_ID/permissions or upload ZIP to backend.")
    st.stop()

if not zipfile.is_zipfile(ZIP_PATH):
    st.error("Downloaded file is not a valid ZIP. Check the file on Drive.")
    st.stop()

# extract
if not os.path.exists(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):
    extract_zip(ZIP_PATH, EXTRACT_DIR)

# subjects dropdown (manual)
SUBJECT_CHOICES = ["Economics", "Psychology", "Sociology", "Polity", "Business Studies"]
subject_choice = st.sidebar.selectbox("Select subject (manual)", SUBJECT_CHOICES + ["Auto-detect"])

# user inputs
topic = st.text_input("Enter chapter name or topic (e.g., 'Constitutional Design', 'Electricity', 'Reproduction'):")
num_questions = st.slider("Number of long subjective questions", 1, 10, 5)
top_k = st.number_input("Retrieval: number of chunks to use as context", min_value=1, max_value=20, value=DEFAULT_TOP_K)

# helper: map selected subject to actual folder (fuzzy)
def find_folder_for_subject(choice, base_dir=EXTRACT_DIR):
    # normalize
    want = choice.lower().strip().replace(" ", "")
    folders = list_subfolders(base_dir)
    for f in folders:
        fnorm = f.lower().replace(" ", "")
        if want in fnorm or fnorm in want:
            return os.path.join(base_dir, f)
    # try keyword matching
    keywords = {
        "economics": ["econ", "economics"],
        "psychology": ["psych", "psychology"],
        "sociology": ["sociology", "soc"],
        "polity": ["polity", "political", "politics", "politicalscience", "political_science"],
        "businessstudies": ["business", "businessstudies", "bst", "business_studies"]
    }
    for key, kws in keywords.items():
        if key in want or any(k in want for k in kws):
            # search folders for any folder with those keywords
            for f in folders:
                fnorm = f.lower().replace(" ", "")
                if any(k in fnorm for k in kws):
                    return os.path.join(base_dir, f)
    return None

if st.button("Generate Questions") and topic.strip():
    # determine folder
    if subject_choice == "Auto-detect":
        # simple auto-detect: scan all subject folders for highest keyword counts
        best_folder = None
        best_score = 0
        words = re.findall(r"\w+", topic.lower())
        for f in list_subfolders(EXTRACT_DIR):
            folder_path = os.path.join(EXTRACT_DIR, f)
            scanned = 0
            score = 0
            for root, _, files in os.walk(folder_path):
                for ff in files:
                    if not ff.lower().endswith(".pdf"):
                        continue
                    scanned += 1
                    p = os.path.join(root, ff)
                    txt = read_pdf_text(p).lower()
                    for w in words:
                        score += txt.count(w)
                    if scanned >= 5:
                        break
                if scanned >= 5:
                    break
            if score > best_score:
                best_score = score
                best_folder = folder_path
        if best_folder:
            selected_folder = best_folder
            subject_name = os.path.relpath(best_folder, EXTRACT_DIR)
            st.success(f"Detected subject: **{subject_name}**")
        else:
            selected_folder = EXTRACT_DIR
            subject_name = "Full Corpus (no single subject detected)"
            st.info("No strong subject match; using full corpus.")
    else:
        # manual selection mapping
        mapped = find_folder_for_subject(subject_choice, EXTRACT_DIR)
        if mapped:
            selected_folder = mapped
            subject_name = os.path.relpath(mapped, EXTRACT_DIR)
            st.success(f"Selected subject: **{subject_name}**")
        else:
            selected_folder = EXTRACT_DIR
            subject_name = subject_choice + " (folder not found; using full corpus)"
            st.warning(f"No exact folder found for '{subject_choice}'. Using full corpus instead.")

    # load docs from only the selected folder
    with st.spinner("Loading PDFs from selected folder..."):
        docs = load_docs_from_folder(selected_folder)
        st.info(f"Loaded {len(docs)} readable PDF documents from: {subject_name}")
        if not docs:
            st.error("No readable documents found in selected folder. Try another subject or upload correct ZIP.")
            st.stop()

    with st.spinner("Creating chunks..."):
        chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        st.info(f"Created {len(chunks)} chunks from selected subject.")

    if not chunks:
        st.error("No text chunks created. Cannot proceed.")
        st.stop()

    # build faiss for these chunks
def build_faiss_index(chunks):
    """Build a FAISS index from text chunks."""

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # If chunks are strings, use them directly
    if isinstance(chunks[0], str):
        texts = chunks
    else:
        # If chunks are dicts
        texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks



    
    st.session_state["_faiss_chunks_temp"] = chunks
    with st.spinner("Building embeddings & FAISS index (cached)..."):
        emb_model_local, index_local, metadata_local = build_faiss("_faiss_chunks_temp") if False else build_faiss("dummy_key")
        # Note: build_faiss reads chunks from session_state key _faiss_chunks_temp
    st.session_state.pop("_faiss_chunks_temp", None)

    if index_local is None or metadata_local is None:
        st.error("Failed to build FAISS index.")
        st.stop()
    st.success("FAISS index ready for selected subject.")

    # retrieve chunks relevant to topic
    with st.spinner("Retrieving relevant chunks..."):
        retrieved = retrieve_chunks(topic, index_local, metadata_local, top_k=top_k)
    if not retrieved:
        st.warning("No relevant content found in this subject for that topic. Try different keyword or increase retrieval size.")
        st.stop()
    st.info(f"Retrieved {len(retrieved)} chunks as context.")

    # load generator
    generator = load_generator()

    # build prompt & generate
    prompt = build_question_prompt(retrieved, topic, num_questions)
    with st.spinner("Generating long subjective questions..."):
        try:
            out = generator(prompt, max_length=600, do_sample=False)[0]["generated_text"]
        except Exception as e:
            st.error(f"Generation failed: {e}")
            out = ""

    if out:
        st.write("### Generated Long Subjective Questions")
        st.markdown(out)
        st.write(f"### 📌 Subject Used: **{subject_name}**")
        st.write("### Sources used (file — chunk_id)")
        for r in retrieved:
            st.write(f"{r['doc_id']} — {r['chunk_id']}")
    else:
        st.error("No output produced by the generator.")
