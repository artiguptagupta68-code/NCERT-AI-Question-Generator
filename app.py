
import streamlit as st
import zipfile
import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

st.set_page_config(page_title="📚 AI NCERT Question Generator", layout="wide")
st.title("📚 AI NCERT Question Generator (Offline)")

SUBJECTS = ["Economics", "Polity", "Business Studies", "Psychology", "Sociology"]

# ---------------- Upload ZIP ----------------
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

        # ---------------- Embeddings & FAISS ----------------
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedding_model.embed_documents(chunks)

        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        embedding_matrix = np.array(embeddings).astype("float32")
        index.add(embedding_matrix)

        faiss_index = {
            "index": index,
            "chunks": chunks
        }
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



        # ---------------- User input ----------------
        query_topic = st.text_input("Enter topic/chapter (keyword) to generate questions:")

        if st.button("Generate Questions"):
            if not query_topic.strip():
                st.warning("Enter a topic keyword.")
            else:
                # Retrieve top K relevant chunks
                model_embed = SentenceTransformer("all-MiniLM-L6-v2")
                query_embedding = model_embed.encode([query_topic], convert_to_numpy=True)
                D, I = index.search(query_embedding.astype("float32"), k=min(5, len(chunks)))
                retrieved_chunks = [chunks[i] for i in I[0]]
                context = "\n\n".join(retrieved_chunks)

                # ---------------- Transformer-based offline question generator ----------------
                # CPU-friendly model
                llm_model_name = "facebook/opt-125m"
                tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
                llm_model.to("cpu")

                # Build prompt for multiple questions
                prompt = f"""
You are an expert NCERT question setter.

Topic: {query_topic}

Based on the context below, generate {num_q} NCERT-style questions (1-mark, 2-mark, 5-mark mixed). 
Each question must start with an interrogative or command verb and end with '?'

Context:
{context}
"""
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = llm_model.generate(
                        **inputs,
                        max_new_tokens=600,
                        temperature=0.25,
                        do_sample=True
                    )
                result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # ---------------- Clean & format questions ----------------
                lines = [ln.strip() for ln in result_text.splitlines() if ln.strip()]
                questions = []
                for ln in lines:
                    if re.match(r'^\d+\.\s+', ln) and not ln.endswith('?'):
                        ln = ln.rstrip(' .') + '?'
                    questions.append(ln)

                st.subheader("Generated Questions:")
                for q in questions:
                    st.write(q)
