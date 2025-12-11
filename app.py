# app.py
import os
import zipfile
import shutil
from pathlib import Path
import json
import re
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

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

GEN_MODEL_NAME = "google/flan-t5-base"

TOP_K = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

st.set_page_config(page_title="NCERT AI UPSC Question Generator", layout="wide")
st.title("📘 NCERT AI UPSC Question Generator")
st.caption("Generate UPSC-style NCERT questions using RAG + Transformers")

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
    # Handle nested zips
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
    return docs

# ----------------------------
# CLEAN NCERT TEXT
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
        r"[A-Za-z ]+ Singh"
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

# ----------------------------
# CHUNK DOCUMENTS
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
# RETRIEVE TOP CHUNKS
# ----------------------------
def retrieve_chunks(query, index, metadata, top_k=TOP_K):
    if index is None or metadata is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, index.ntotal)
    D, I = index.search(q_emb, k)
    results = [metadata[idx] for idx in I[0] if idx < len(metadata)]
    return results

# ----------------------------
# EXTRACT QUESTIONS
# ----------------------------
QUESTION_START_WORDS = ["What", "Why", "How", "Explain", "Describe", "State", "Define", "Discuss", "Examine", "Evaluate", "Tell"]
def extract_questions(text, num_questions):
    pattern = r'(?:' + '|'.join(QUESTION_START_WORDS) + r').*?\?'
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    seen = set()
    final = []
    for q in matches:
        q_clean = q.strip()
        if q_clean not in seen:
            seen.add(q_clean)
            final.append(q_clean)
        if len(final) >= num_questions:
            break
    return final

# ----------------------------
# UPSC REFERENCE QUESTIONS
# ----------------------------
def pre_generate_upsc_questions(generator, docs, subject, n_questions=500):
    all_text = "\n\n".join([clean_ncert_text(d["text"])[:1500] for d in docs])
    reference_questions = []
    for i in range(n_questions):
        prompt = f"""
You are an expert UPSC question setter.

Generate ONE UPSC-style exam question strictly based on the following NCERT context.
Use only the NCERT facts.

Context (subject: {subject}):
{all_text}

Output:
One UPSC-style question.
"""
        out = generator(prompt, max_length=200, do_sample=True, temperature=0.6, top_p=0.9)[0]["generated_text"]
        qs = extract_questions(out, 1)
        if qs:
            reference_questions.append(qs[0])
    os.makedirs("upsc_reference", exist_ok=True)
    path = f"upsc_reference/{subject.replace(' ','_')}_questions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reference_questions, f, indent=2, ensure_ascii=False)
    return path

# 3️⃣ Limit reference questions generation
def load_upsc_reference(subject):
    path = f"/tmp/upsc_reference/{subject.replace(' ','_')}_questions.json"
    os.makedirs("/tmp/upsc_reference", exist_ok=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def generate_upsc_questions_with_reference(generator, topic, context_text, reference_questions, num_questions):
    reference_sample = "\n".join(reference_questions[:min(20, len(reference_questions))])
    prompt = f"""
You are an expert UPSC question setter.

Based on the NCERT context below AND using the style and phrasing from the following UPSC reference questions, generate {num_questions} exam-style questions for the topic: {topic}.

NCERT Context:
{context_text}

Reference UPSC Questions:
{reference_sample}

Output {num_questions} distinct questions numbered 1., 2., 3., ... 
Each question must:
- Start with interrogative words (What, Why, How, Explain, etc.)
- End with a question mark (?)
- Be meaningful, based ONLY on NCERT context
"""
    out = generator(prompt, max_length=600, do_sample=True, temperature=0.5, top_p=0.9)[0]["generated_text"]
    return extract_questions(out, num_questions)

# ----------------------------
# ORCHESTRATION
# ----------------------------
st.text("Preparing NCERT content...")
ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Failed to download NCERT ZIP.")
    st.stop()
if not zipfile.is_zipfile(ZIP_PATH):
    st.error(f"{ZIP_PATH} is not a valid ZIP file.")
    st.stop()
extract_zip(ZIP_PATH, EXTRACT_DIR)
st.success(f"NCERT ZIP extracted to: {EXTRACT_DIR}")

# Subject selection
subject = st.selectbox("Select Subject", SUBJECTS)
docs = load_docs_by_subject(EXTRACT_DIR, subject)
st.info(f"Loaded {len(docs)} PDFs for {subject}")
if not docs:
    st.warning(f"No readable PDFs found for {subject}.")
    st.stop()

# Chunking
all_chunks = chunk_documents(docs)
st.info(f"Total chunks created: {len(all_chunks)}")

# FAISS
embed_model, index, metadata = build_faiss_index(all_chunks)
if index is None:
    st.error("Failed to build FAISS index.")
    st.stop()
st.success("FAISS index ready.")

# Generator
generator = load_generator_pipeline()
st.success("Generator model loaded.")

# Load UPSC reference questions
if not os.path.exists(f"upsc_reference/{subject.replace(' ','_')}_questions.json"):
    st.text(f"Generating reference UPSC questions for {subject} (one-time)...")
    pre_generate_upsc_questions(generator, docs, subject)
reference_questions = load_upsc_reference(subject)

# User input
st.subheader("Generate NCERT UPSC Questions")
topic = st.text_input("Enter chapter/topic (example: 'Constitution', 'Electricity')", key="topic_input")
num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=20, value=5, key="num_questions_input")

# 1️⃣ Move retrieved_chunks and context_text inside button
if st.button("Generate Questions"):
    retrieved_chunks = retrieve_chunks(topic, index, metadata, top_k=TOP_K)
    if not retrieved_chunks:
        st.warning(f"No relevant NCERT content found for '{topic}' in {subject}.")
    else:
        context_text = "\n\n".join([r["text"][:1200] for r in retrieved_chunks])
        context_text = clean_ncert_text(context_text)

            final_questions = generate_upsc_questions_with_reference(generator, topic, context_text, reference_questions, num_questions)

            if final_questions:
                st.success(f"Generated {len(final_questions)} Questions")
                for i, q in enumerate(final_questions, 1):
                    st.write(f"{i}. {q}")
                st.write("### Sources used")
                for r in retrieved_chunks:
                    st.write(f"{r['doc_id']} — {r['chunk_id']}")
            else:
                st.warning("Model could not generate distinct questions. Reduce number or simplify topic.")
