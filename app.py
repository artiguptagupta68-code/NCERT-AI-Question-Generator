# app.py
import os
import zipfile
import shutil
from pathlib import Path
import streamlit as st
import gdown
import torch
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-large"
TOP_K = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Generate high-quality NCERT-style subjective questions using RAG + Transformers")

# ----------------------------
# DOWNLOAD ZIP
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

# ----------------------------
# EXTRACT ZIP + NESTED ZIPS
# ----------------------------
def extract_zip(zip_path: str, extract_to: str):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)

    shutil.rmtree(extract_to, ignore_errors=True)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith(".zip"):
                nested = os.path.join(root, f)
                nested_dir = os.path.join(root, Path(f).stem)
                os.makedirs(nested_dir, exist_ok=True)
                try:
                    with zipfile.ZipFile(nested, "r") as nz:
                        nz.extractall(nested_dir)
                except:
                    pass

# ----------------------------
# PDF READERS
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
    return docs

# ----------------------------
# ADVANCED NCERT CLEANER
# ----------------------------
def clean_ncert_text(text):
    patterns = [
        r"Reprint\s*\d{4}-\d{2}",
        r"ISBN[\s:0-9-]+",
        r"NCERT[\s\S]{0,50}Publication",
        r"Not for commercial use",
        r"Government of India",
        r"Publication Division",
        r"©[\s\S]{0,30}",
        r"[A-Za-z ]+ Rao",
        r"[A-Za-z ]+ Singh",
        r"[A-Za-z ]+ Kumar",
        r"[A-Za-z ]+ Devi",
        r"[A-Za-z ]+ Sharma"
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

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
# FAISS INDEX
# ----------------------------
@st.cache_resource
def build_faiss_index(chunks):
    if not chunks:
        return None, None, None
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
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
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# ----------------------------
# RETRIEVAL
# ----------------------------
def retrieve_chunks(query, index, metadata, top_k=TOP_K):
    if index is None or metadata is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, index.ntotal)
    D, I = index.search(q_emb, k)
    return [metadata[idx] for idx in I[0] if idx < len(metadata)]

# ----------------------------
# QUESTION IMPROVEMENTS
# ----------------------------
def sanitize_question(q):
    q = q.strip()
    q = q.replace("passage", "text").replace("Passage", "Text")
    return q

def build_question_prompt(context_text, topic, num):
    return f"""
You are an expert NCERT question setter.

Generate EXACTLY {num} HIGH-QUALITY and DISTINCT questions based ONLY on the following NCERT text:

📌 RULES:
- Use ONLY the NCERT text; do NOT invent facts.
- All questions MUST be different.
- Start with: What, Why, How, Explain, Describe, Discuss, Define, Examine, Evaluate.
- MUST end with a question mark.
- NEVER use word “passage”; ALWAYS use “text”.
- Output ONLY numbered questions.

Topic: {topic}

NCERT Text:
{context_text}

Generate EXACTLY {num} questions:
1.
2.
3.
"""

def extract_good_questions(text, n):
    qs = re.findall(r"\d+\.\s*(.+?\?)", text)

    cleaned = []
    seen = set()

    for q in qs:
        q = sanitize_question(q)
        if len(q) < 10:
            continue
        if any(bad in q.lower() for bad in ["shweta", "reprint", "isbn", "publication"]):
            continue
        if q.lower() not in seen:
            cleaned.append(q)
            seen.add(q.lower())
        if len(cleaned) == n:
            break
    return cleaned

def generate_n_distinct_questions(generator, topic, context_text, num):
    for _ in range(3):
        prompt = build_question_prompt(context_text, topic, num)
        out = generator(prompt, max_length=650, do_sample=True, temperature=0.4)[0]["generated_text"]
        qs = extract_good_questions(out, num)
        if len(qs) == num:
            return qs
    return []

# ----------------------------
# STREAMLIT APP FLOW
# ----------------------------
st.text("Preparing NCERT content...")
ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Failed to download NCERT ZIP.")
    st.stop()

extract_zip(ZIP_PATH, EXTRACT_DIR)
st.success(f"Extracted to {EXTRACT_DIR}")

subject = st.selectbox("Select Subject", SUBJECTS)
docs = load_docs_by_subject(EXTRACT_DIR, subject)
st.info(f"Loaded {len(docs)} PDFs")

if not docs:
    st.warning("No readable PDFs found.")
    st.stop()

all_chunks = chunk_documents(docs)
embed_model, index, metadata = build_faiss_index(all_chunks)

generator = load_generator_pipeline()

st.subheader("Generate NCERT Questions")
topic = st.text_input("Enter topic", key="topic_input")

num_questions = st.number_input(
    "Number of questions",
    min_value=1, max_value=20, value=5
)

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Enter a valid topic.")
    else:
        retrieved_chunks = retrieve_chunks(topic, index, metadata)
        if not retrieved_chunks:
            st.warning("No relevant NCERT content found.")
        else:
            context_text = "\n\n".join([clean_ncert_text(r["text"][:1200]) for r in retrieved_chunks])
            final_questions = generate_n_distinct_questions(generator, topic, context_text, num_questions)

            if final_questions:
                st.success(f"Generated {len(final_questions)} questions")
                for i, q in enumerate(final_questions, 1):
                    st.write(f"{i}. {q}")

                st.write("### Sources Used")
                for r in retrieved_chunks:
                    st.write(f"{r['doc_id']} — {r['chunk_id']}")
            else:
                st.error("Failed to generate meaningful questions. Try reducing the number.")
