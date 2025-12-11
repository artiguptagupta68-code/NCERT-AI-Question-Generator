# app.py
import os
import zipfile
import shutil
from pathlib import Path
import re

import streamlit as st
import gdown
import torch
import numpy as np
from pypdf import PdfReader
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Generate NCERT-style subjective questions from chapters (RAG + Transformers)")

# ----------------------------
# UTILITIES
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
# PDF READING
# ----------------------------
def read_pdf_pypdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
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
            t = page.get_text()
            if t:
                text += t + "\n"
        return text
    except Exception:
        return ""

def read_pdf_text(path):
    text = read_pdf_pypdf(path)
    if text.strip():
        return text
    return read_pdf_pymupdf(path)

# ----------------------------
# LOAD PDFs BY SUBJECT
# ----------------------------
def load_docs_by_subject(folder, subject_keyword):
    subject_keyword = subject_keyword.lower()
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                path_lower = Path(root).name.lower() + " " + f.lower()
                if subject_keyword in path_lower:
                    p = os.path.join(root, f)
                    text = read_pdf_text(p)
                    if text.strip():
                        docs.append({"doc_id": f, "text": text})
                    else:
                        st.warning(f"Unreadable or image-only PDF skipped: {f}")
    return docs

# ----------------------------
# CHUNKING
# ----------------------------
def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    all_chunks = []
    for doc in docs:
        text = doc["text"]
        doc_id = doc["doc_id"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{Path(doc_id).stem}_{start}",
                "text": chunk_text
            })
            start += chunk_size - chunk_overlap
    return all_chunks

# ----------------------------
# EMBEDDINGS + FAISS
# ----------------------------
@st.cache_resource
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
# GENERATOR PIPELINE
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
# RETRIEVE CHUNKS
# ----------------------------
def retrieve_chunks(query, index, metadata, embed_model, top_k=TOP_K):
    if index is None or metadata is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, index.ntotal)
    D, I = index.search(q_emb, k)
    results = [metadata[idx] for idx in I[0] if idx < len(metadata)]
    return results

# ----------------------------
# EXTRACT NUMBERED QUESTIONS
# ----------------------------
def extract_numbered_questions(text, num_questions):
    """
    Extract exactly num_questions from numbered list in model output
    """
    lines = re.split(r'\n?\d+\.\s', text)
    questions = []
    for q in lines[1:]:
        q_clean = q.strip()
        if q_clean and not q_clean.endswith("?"):
            q_clean += "?"
        questions.append(q_clean)
        if len(questions) >= num_questions:
            break
    return questions

# ----------------------------
# ORCHESTRATION
# ----------------------------

# 1️⃣ Prepare NCERT
st.text("Preparing NCERT content...")
ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Failed to download NCERT ZIP. Upload manually or check FILE_ID.")
    st.stop()
if not zipfile.is_zipfile(ZIP_PATH):
    st.error(f"{ZIP_PATH} is not a valid ZIP file.")
    st.stop()
extract_zip(ZIP_PATH, EXTRACT_DIR)
st.success(f"NCERT ZIP extracted to: {EXTRACT_DIR}")

# 2️⃣ Select subject
subject = st.selectbox("Select Subject", SUBJECTS, key="subject_select")
docs = load_docs_by_subject(EXTRACT_DIR, subject)
st.info(f"Loaded {len(docs)} PDFs for {subject}")
if not docs:
    st.warning(f"No readable PDFs found for {subject}.")
    st.stop()

# 3️⃣ Chunk documents
all_chunks = chunk_documents(docs)
st.info(f"Total chunks created: {len(all_chunks)}")

# 4️⃣ Build FAISS
embed_model, index, metadata = build_faiss_index(all_chunks)
if index is None:
    st.error("Failed to build FAISS index.")
    st.stop()
st.success("FAISS index ready.")

# 5️⃣ Load generator
generator = load_generator_pipeline()
st.success("Generator model loaded.")

# 6️⃣ User input
st.subheader("Generate NCERT Questions")
topic = st.text_input("Enter chapter/topic (example: 'Constitution', 'Electricity')", key="topic_input")
num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=20, value=5, key="num_questions_input")

# 7️⃣ Generate button
if st.button("Generate Questions", key="generate_btn"):
    if not topic.strip():
        st.warning("Please enter a valid topic.")
    else:
        retrieved_chunks = retrieve_chunks(topic, index, metadata, embed_model)
        if not retrieved_chunks:
            st.warning(f"No relevant content found for '{topic}' in {subject}.")
        else:
            # Combine top chunks as context
            context_text = "\n\n".join([r["text"][:1200] for r in retrieved_chunks])

            # Build prompt
            prompt = (
                f"You are an NCERT expert. Generate exactly {num_questions} distinct exam-style questions "
                f"based ONLY on the following context.\n"
                f"Rules:\n"
                f"- Start each question with What, Why, How, Explain, Describe, State, Define, Discuss, Examine, Evaluate.\n"
                f"- End each question with '?'.\n"
                f"- Do NOT invent facts; use only NCERT content.\n"
                f"- Number each question: 1. ... 2. ... etc.\n\n"
                f"Topic: {topic}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Generate exactly {num_questions} questions."
            )

            with st.spinner("Generating questions..."):
                try:
                    out = generator(prompt, max_length=1500, do_sample=True, temperature=0.3)[0]["generated_text"]
                except Exception as e:
                    st.error(f"Question generation failed: {e}")
                    out = ""

            # Extract exactly N questions
            final_questions = extract_numbered_questions(out, num_questions)

            if final_questions and len(final_questions) == num_questions:
                st.success(f"Generated exactly {len(final_questions)} Questions")
                for i, q in enumerate(final_questions, 1):
                    st.write(f"{i}. {q}")

                # Show sources used
                st.write("### Sources used")
                for r in retrieved_chunks:
                    st.write(f"{r['doc_id']} — {r['chunk_id']}")
            else:
                st.warning(f"Model generated only {len(final_questions)} questions. Try reducing the number or simplifying the topic.")
