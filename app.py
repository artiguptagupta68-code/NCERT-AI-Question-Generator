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
    import fitz
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
st.caption("Generate NCERT-style questions using RAG (FAISS + Transformers)")

# ----------------------------
# Utilities
# ----------------------------
def download_zip_from_drive(file_id: str, out_path: str):
    if os.path.exists(out_path):
        return True
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except:
        return False


def extract_zip(zip_path, extract_dir):
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # extract nested zips
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                nested = os.path.join(root, f)
                sub = os.path.join(root, Path(f).stem)
                os.makedirs(sub, exist_ok=True)
                try:
                    with zipfile.ZipFile(nested, "r") as nz:
                        nz.extractall(sub)
                except:
                    pass


# ----------------------------
# PDF reading
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
    except:
        return ""


def read_pdf_pymupdf(path):
    if not _HAS_PYMUPDF:
        return ""
    try:
        doc = fitz.open(path)
        text = ""
        for p in doc:
            t = p.get_text()
            if t:
                text += t + "\n"
        return text
    except:
        return ""


def read_pdf_text(path):
    text = read_pdf_pypdf(path)
    if text.strip():
        return text
    return read_pdf_pymupdf(path)


# ----------------------------
# Load PDFs by subject
# ----------------------------
def load_docs_by_subject(folder, subject):
    docs = []
    subject = subject.lower()
    for root, _, files in os.walk(folder):
        folder_name = Path(root).name.lower()
        if subject in folder_name:
            for f in files:
                if f.lower().endswith(".pdf"):
                    path = os.path.join(root, f)
                    text = read_pdf_text(path)
                    if text.strip():
                        docs.append({"doc_id": f, "text": text})
    return docs


# ----------------------------
# Clean NCERT noisy metadata
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
# Chunking
# ----------------------------
def chunk_documents(docs):
    all_chunks = []
    for d in docs:
        text = d["text"]
        doc_id = d["doc_id"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{Path(doc_id).stem}_{start}",
                "text": chunk
            })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks


# ----------------------------
# Building FAISS
# ----------------------------
@st.cache_resource
def build_faiss_index(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    metadata = chunks.copy()
    return model, index, metadata


# ----------------------------
# Generator model
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
# Retrieve top-k chunks
# ----------------------------
def retrieve_chunks(query, index, metadata):
    if index is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(TOP_K, index.ntotal)
    D, I = index.search(q_emb, k)
    return [metadata[i] for i in I[0] if i < len(metadata)]


# ----------------------------
# Extract well-formed questions
# ----------------------------
QUESTION_START_WORDS = [
    "What", "Why", "How", "Explain", "Describe", "State",
    "Define", "Discuss", "Examine", "Evaluate"
]

def extract_questions(text):
    pattern = r"(?:{}).*?\?".format("|".join(QUESTION_START_WORDS))
    qs = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return [q.strip() for q in qs]


# ----------------------------
# Main generator: ensure distinct, meaningful questions
# ----------------------------
def generate_distinct_questions(generator, topic, context, n):
    questions = set()
    attempts = 0
    max_attempts = n * 6

    while len(questions) < n and attempts < max_attempts:

        prompt = f"""
You are an NCERT expert question setter.

Generate ONE meaningful question only from the NCERT context below.
Rules:
- Must be related to the topic: {topic}
- Must be meaningful and fact-based.
- Must NOT contain words like “text”, “passage”, “author”, “reprint”, “ISBN”.
- Must end with a question mark (?).
- Must use interrogative words like What, Why, How, Explain, Describe, Discuss.

NCERT CONTEXT:
{context}

Generate ONE high-quality question:
"""

        out = generator(
            prompt,
            max_length=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )[0]["generated_text"]

        qs = extract_questions(out)
        if qs:
            q = qs[0]

            # Filter out unwanted noisy hallucinations
            BAD = ["Reprint", "ISBN", "Shweta", "Publication", "Government of India"]
            if any(b in q for b in BAD):
                attempts += 1
                continue

            if len(q.split()) > 5:  
                questions.add(q)

        attempts += 1

    questions = list(questions)
    while len(questions) < n:
        questions.append("Model could not generate a distinct question. Please reduce the number.")

    return questions[:n]


# ----------------------------
# ORCHESTRATION
# ----------------------------
st.text("Preparing NCERT content...")

ok = download_zip_from_drive(FILE_ID, ZIP_PATH)
if not ok:
    st.error("Download failed.")
    st.stop()

extract_zip(ZIP_PATH, EXTRACT_DIR)
st.success("NCERT content extracted.")

subject = st.selectbox("Select Subject", SUBJECTS)

docs = load_docs_by_subject(EXTRACT_DIR, subject)
st.info(f"Loaded: {len(docs)} PDFs")

if not docs:
    st.error("No PDFs found for this subject.")
    st.stop()

all_chunks = chunk_documents(docs)
st.info(f"Total chunks created: {len(all_chunks)}")

embed_model, index, metadata = build_faiss_index(all_chunks)
st.success("FAISS Ready.")

generator = load_generator_pipeline()
st.success("Question Generator Model Loaded.")


# ----------------------------
# UI: Generate questions
# ----------------------------
st.subheader("Generate NCERT Questions")

topic = st.text_input("Enter topic (example: Constitution, Federalism, Motion)")
num_questions = st.number_input("Number of questions", 1, 20, 5)

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Enter a topic.")
    else:
        retrieved = retrieve_chunks(topic, index, metadata)

        if not retrieved:
            st.warning("No relevant NCERT content found for this topic.")
        else:
            context_text = "\n\n".join([r["text"][:1200] for r in retrieved])
            context_text = clean_ncert_text(context_text)

            final_qs = generate_distinct_questions(
                generator, topic, context_text, num_questions
            )

            st.success(f"Generated {len(final_qs)} Questions")
            for i, q in enumerate(final_qs, 1):
                st.write(f"{i}. {q}")

            st.write("### Sources used:")
            for r in retrieved:
                st.write(f"- {r['doc_id']} — {r['chunk_id']}")
