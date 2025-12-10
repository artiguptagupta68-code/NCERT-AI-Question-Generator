# app.py
"""
NCERT RAG Question Generator — Offline LoRA-powered
- Downloads ncrt.zip from Google Drive (FILE_ID)
- Extracts nested zips, finds PDFs recursively
- Filters PDFs by subject keyword
- Reads PDF text, chunks
- Embeds with SentenceTransformers
- Builds FAISS index and retrieves top-K chunks
- Generates 1/2/5 mark questions offline using a LoRA-fine-tuned model
"""

import os
import zipfile
import shutil
from pathlib import Path
import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Optional faster text extraction
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

# -----------------------
# CONFIG
# -----------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncrt_extracted"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SUBJECTS = ["Economics", "Polity", "Business Studies", "Psychology", "Sociology"]
DEFAULT_TOP_K = 6
DEFAULT_NUM_Q = 6

# LoRA-fine-tuned model (offline)
BASE_MODEL = "./base_model"  # path to your local base model
LORA_WEIGHTS = "./ncert_lora_model"  # path to LoRA weights

st.set_page_config(page_title="NCERT RAG — Offline LoRA Question Generator", layout="wide")
st.title("NCERT RAG — Offline LoRA-powered Question Generator (1/2/5 mark)")

faiss.omp_set_num_threads(1)

# -----------------------
# Utilities
# -----------------------
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

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
    safe_mkdir(dest_dir)
    with zipfile.ZipFile(src_zip_path, "r") as z:
        z.extractall(dest_dir)
    found_any = True
    while found_any:
        found_any = False
        for root, _, files in os.walk(dest_dir):
            for f in list(files):
                if f.lower().endswith(".zip"):
                    found_any = True
                    nested_zip = os.path.join(root, f)
                    folder_target = os.path.join(root, Path(f).stem)
                    safe_mkdir(folder_target)
                    try:
                        with zipfile.ZipFile(nested_zip, "r") as nz:
                            nz.extractall(folder_target)
                    except Exception as e:
                        st.warning(f"Failed to extract nested zip {nested_zip}: {e}")
                    try:
                        os.remove(nested_zip)
                    except Exception:
                        pass
    return sorted(os.listdir(dest_dir))

# -----------------------
# PDF extraction
# -----------------------
def read_pdf_pypdf(path):
    try:
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""

def read_pdf_pymupdf(path):
    if not _HAS_PYMUPDF:
        return ""
    try:
        doc = fitz.open(path)
        parts = []
        for page in doc:
            t = page.get_text()
            if t:
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""

def read_pdf_text(path):
    t = read_pdf_pypdf(path)
    if t and t.strip():
        return t
    if _HAS_PYMUPDF:
        t2 = read_pdf_pymupdf(path)
        if t2 and t2.strip():
            return t2
    return ""

# -----------------------
# Embedding model cache
# -----------------------
@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

# -----------------------
# File discovery & filtering
# -----------------------
def list_all_pdfs(base_dir):
    out = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(root, f))
    return sorted(out)

def filter_pdfs_for_subject(all_pdf_paths, subject_keyword):
    kw = subject_keyword.lower().strip()
    matched = []
    for p in all_pdf_paths:
        parts = [part.lower() for part in Path(p).parts]
        file_name = Path(p).name.lower()
        if any(kw in part for part in parts) or (kw in file_name):
            matched.append(p)
    return sorted(list(dict.fromkeys(matched)))

# -----------------------
# Read docs & chunk
# -----------------------
def load_docs_from_paths(pdf_paths):
    docs = []
    for path in pdf_paths:
        txt = read_pdf_text(path)
        if txt and txt.strip():
            docs.append({"path": path, "doc_id": Path(path).name, "text": txt})
        else:
            st.warning(f"Unreadable or image PDF skipped: {path}")
    return docs

def chunk_text_simple(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk.split()) > 15:
            chunks.append(chunk)
        i += step
    return chunks

def docs_to_chunks(docs):
    all_chunks = []
    for d in docs:
        parts = chunk_text_simple(d["text"])
        for i, p in enumerate(parts):
            all_chunks.append({
                "doc_id": d["doc_id"],
                "path": d["path"],
                "chunk_id": f"{Path(d['doc_id']).stem}_chunk_{i}",
                "text": p
            })
    return all_chunks

# -----------------------
# FAISS & retrieval
# -----------------------
@st.cache_data(show_spinner=True)
def build_faiss_for_subject(subject_name, chunks_texts):
    if not chunks_texts:
        return None, []
    embed_model = load_embedding_model()
    texts = [c["text"] for c in chunks_texts]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    metadata = [{"doc_id": c["doc_id"], "path": c["path"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks_texts]
    return index, metadata

def retrieve_topk(index, embed_model, query, top_k=DEFAULT_TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    k = min(top_k, int(index.ntotal)) if index is not None else 0
    if k <= 0:
        return []
    D, I = index.search(q_emb, k)
    ids = I[0].tolist()
    return ids, D[0].tolist()

# -----------------------
# LoRA Model Loading
# -----------------------
@st.cache_resource
def load_lora_model(base_model_path=BASE_MODEL, lora_weights_path=LORA_WEIGHTS):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", load_in_8bit=True)
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    return generator

# -----------------------
# Question Generation
# -----------------------
def generate_questions_loRA(generator, topic, retrieved_meta, num_total=6):
    questions = []
    topic_tokens = [t.lower() for t in re.split(r'\W+', topic) if t]
    for r in retrieved_meta:
        text = r.get("text", "")
        # split into sentences
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            s_clean = re.sub(r'\s+', ' ', s).strip()
            if not s_clean:
                continue
            # generate a question from LoRA model
            prompt = f"Instruction: Generate 1/2/5 mark question.\nInput: {s_clean}\nTopic: {topic}\nOutput:"
            out = generator(prompt, max_new_tokens=120)
            q_text = out[0]["generated_text"].split("Output:")[-1].strip()
            if q_text and len(questions) < num_total:
                questions.append(q_text)
            if len(questions) >= num_total:
                break
        if len(questions) >= num_total:
            break
    # simple split into 1/2/5 marks
    c1 = min(2, num_total)
    c2 = min(2, max(0, num_total-c1))
    c5 = max(0, num_total-c1-c2)
    return {
        "1": questions[:c1],
        "2": questions[c1:c1+c2],
        "5": questions[c1+c2:c1+c2+c5]
    }

# -----------------------
# Streamlit UI
# -----------------------
col1, col2 = st.columns([3,1])
with col1:
    st.write("Google Drive FILE_ID used to download main ZIP.")
    st.write(f"`FILE_ID = {FILE_ID}`")
with col2:
    if st.button("Redownload & Re-extract ZIP"):
        try:
            if os.path.exists(EXTRACT_DIR):
                shutil.rmtree(EXTRACT_DIR)
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
        except Exception:
            pass
        st.experimental_rerun()

# download & extract
if not os.path.exists(ZIP_PATH):
    with st.spinner("Downloading ZIP from Google Drive..."):
        ok = download_drive_file(FILE_ID, ZIP_PATH)
    if not ok:
        st.error("Failed to download ZIP. Check FILE_ID or upload manually.")
        st.stop()
    st.success("ZIP downloaded.")

if os.path.exists(EXTRACT_DIR):
    try:
        shutil.rmtree(EXTRACT_DIR)
    except Exception:
        pass

with st.spinner("Extracting ZIP and nested ZIPs..."):
    entries = extract_zip_with_nested(ZIP_PATH, EXTRACT_DIR)
st.info(f"Top-level entries: {entries}")

all_pdfs = list_all_pdfs(EXTRACT_DIR)
st.info(f"Total PDFs found recursively: {len(all_pdfs)}")
if len(all_pdfs) == 0:
    st.error("No PDFs found in extracted ZIP. Check contents.")
    st.stop()

subject = st.selectbox("Select subject", SUBJECTS)
matched_pdfs = filter_pdfs_for_subject(all_pdfs, subject)
st.info(f"PDFs matched for '{subject}': {len(matched_pdfs)}")
if len(matched_pdfs) == 0:
    st.warning("No auto-matched files. Showing all discovered PDFs:")
    for p in all_pdfs[:200]:
        st.write(p)
    st.stop()

with st.spinner("Reading matched PDFs..."):
    docs = load_docs_from_paths(matched_pdfs)
st.success(f"Readable PDFs loaded for '{subject}': {len(docs)}")
if len(docs) == 0:
    st.error("No readable text. Consider OCR.")
    st.stop()

with st.spinner("Chunking documents..."):
    chunks = docs_to_chunks(docs)
st.info(f"Total chunks: {len(chunks)}")
if len(chunks) == 0:
    st.error("No chunks created. Reduce CHUNK_SIZE.")
    st.stop()

with st.spinner("Building FAISS index..."):
    index, metadata = build_faiss_for_subject(subject, chunks)
st.success("FAISS ready.")

topic = st.text_input("Enter topic/chapter (keyword) to generate questions (e.g., 'constitution'):")
num_q = st.number_input("Number of total questions", min_value=1, max_value=20, value=DEFAULT_NUM_Q)
top_k = st.number_input("Top K chunks to retrieve", min_value=1, max_value=50, value=DEFAULT_TOP_K)

generator = load_lora_model(BASE_MODEL, LORA_WEIGHTS)

if st.button("Generate Questions (Offline LoRA)"):
    if not topic.strip():
        st.warning("Enter a topic or keyword.")
    else:
        with st.spinner("Retrieving top chunks..."):
            ids, _ = retrieve_topk(index, load_embedding_model(), topic, top_k=top_k)
            retrieved_meta = [metadata[idx] for idx in ids if 0<=idx<len(metadata)]
        if not retrieved_meta:
            st.warning("No relevant chunks found. Try a broader keyword.")
        else:
            st.write("### Retrieved snippets:")
            for m in retrieved_meta:
                st.write(f"- {m['doc_id']} — {m['chunk_id']}")
                st.write(m['text'][:350]+"...")
                st.write("---")

            fallback = generate_questions_loRA(generator, topic, retrieved_meta, int(num_q))
            st.markdown("### 1-Mark Questions")
            for q in fallback["1"]:
                st.write(f"- {q}")

            st.markdown("### 2-Mark Questions")
            for q in fallback["2"]:
                st.write(f"- {q}")

            st.markdown("### 5-Mark Questions")
            for q in fallback["5"]:
                st.write(f"- {q}")
