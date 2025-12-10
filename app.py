%%writefile app.py
import streamlit as st
import zipfile
import os
import shutil
from pathlib import Path
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import numpy as np
import faiss
import gdown
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="📘 NCERT Question Generator", layout="wide")
st.title("📘 AI NCERT Question Generator (Transformer-based, No LLM API)")


# ----------------------------
# Flags
# ----------------------------
_HAS_PYMUPDF = True


# ----------------------------
# Utilities: download
# ----------------------------
def download_zip_from_drive(file_id: str, out_path: str) -> bool:
    try:
        if os.path.exists(out_path):
            return True
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False


# ----------------------------
# Unzip + nested unzip
# ----------------------------
def extract_zip(zip_path: str, extract_to: str):
    shutil.rmtree(extract_to, ignore_errors=True)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower().endswith(".zip"):
                nested_path = os.path.join(root, f)
                nested_out = os.path.join(root, Path(f).stem)
                os.makedirs(nested_out, exist_ok=True)
                try:
                    with zipfile.ZipFile(nested_path, "r") as nz:
                        nz.extractall(nested_out)
                except:
                    pass


# ----------------------------
# PDF reading utilities
# ----------------------------
def read_pdf_pypdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except:
        return ""


def read_pdf_pymupdf(path):
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

    if _HAS_PYMUPDF:
        text = read_pdf_pymupdf(path)
        if text.strip():
            return text

    return ""


# ----------------------------
# Load all documents
# ----------------------------
def load_documents(folder):
    texts = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                txt = read_pdf_text(path)
                if txt.strip():
                    texts.append(txt)
            elif f.lower().endswith(".txt"):
                with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                    texts.append(fp.read())
    return texts


# ----------------------------
# Input fields
# ----------------------------
file_id = st.text_input("Google Drive File ID for NCERT ZIP")

subject = st.selectbox("Select Subject", [
    "Physics", "Chemistry", "Maths", "Biology", "History",
    "Geography", "Political Science", "Economics", "English", "Hindi"
])

num_q = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)

start_btn = st.button("Generate Questions")


# ----------------------------
# Main logic
# ----------------------------
if start_btn:

    if not file_id:
        st.error("Please enter a Google Drive File ID")
        st.stop()

    zip_path = "/tmp/ncert.zip"
    extract_folder = "/tmp/ncert_data"

    st.info("Downloading ZIP from Google Drive...")
    if not download_zip_from_drive(file_id, zip_path):
        st.error("Failed to download ZIP.")
        st.stop()

    st.info("Extracting ZIP...")
    extract_zip(zip_path, extract_folder)

    st.info("Reading PDFs/TXT...")
    docs = load_documents(extract_folder)

    if not docs:
        st.error("No readable files found in ZIP.")
        st.stop()

    # ------------------------
    # Filter by subject keyword
    # ------------------------
    subject_texts = [t for t in docs if subject.lower() in t.lower()]

    if not subject_texts:
        st.warning(f"No documents matched subject '{subject}'. Using all documents instead.")
        subject_texts = docs

    full_text = " ".join(subject_texts)

    # ------------------------
    # Split into chunks
    # ------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(full_text)

    # ------------------------
    # Embeddings + FAISS
    # ------------------------
    st.info("Generating embeddings...")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embed_model.encode(chunks)

    vec_dim = vectors.shape[1]
    index = faiss.IndexFlatL2(vec_dim)
    index.add(np.array(vectors).astype("float32"))

    # ------------------------
    # Build fake queries to retrieve random/chosen contexts
    # ------------------------
    query_embedding = embed_model.encode([subject], convert_to_numpy=True).astype("float32")
    _, I = index.search(query_embedding, k=num_q)

    selected_chunks = [chunks[i] for i in I[0]]

    # ------------------------
    # Load transformers model for question generation
    # ------------------------
    st.info("Generating questions using OPT-125m...")

    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cpu")

    # ------------------------
    # Generate Questions
    # ------------------------
    questions = []

    for i, ctx in enumerate(selected_chunks):
        prompt = (
            f"Read the context and generate ONE exam-style question.\n"
            f"Context:\n{ctx}\n\n"
            f"Generated Question:"
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7
            )

        q = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        questions.append(q.split("Generated Question:")[-1].strip())

    # ------------------------
    # Display Output
    # ------------------------
    st.subheader(f"📘 Generated {num_q} Questions ({subject})")
    for i, q in enumerate(questions, 1):
        st.write(f"**Q{i}. {q}**")
