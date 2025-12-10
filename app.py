
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

# --------------------------------------------------------------------
# PDF TEXT EXTRACTORS
# --------------------------------------------------------------------
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
        for page in doc:
            t = page.get_text()
            if t:
                text += t + "\n"
        return text
    except:
        return ""

def read_pdf_text(path):
    txt = read_pdf_pypdf(path)
    if txt.strip():
        return txt
    return read_pdf_pymupdf(path)

# --------------------------------------------------------------------
# LOAD PDFS
# --------------------------------------------------------------------
def load_docs(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, f)
                text = read_pdf_text(pdf_path)
                if text.strip():
                    docs.append({"doc_id": f, "text": text})
                else:
                    st.warning(f"Unreadable PDF skipped: {f}")
    return docs

# --------------------------------------------------------------------
# CHUNKING
# --------------------------------------------------------------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for doc in docs:
        parts = splitter.split_text(doc["text"])
        for i, p in enumerate(parts):
            chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "text": p
            })
    return chunks

# --------------------------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------------------------
st.subheader("Upload NCERT ZIP")
uploaded_zip = st.file_uploader("Upload NCERT PDFs (ZIP)", type=["zip"])

if uploaded_zip:
    with open("ncert.zip", "wb") as f:
        f.write(uploaded_zip.read())

    EXTRACT_DIR = "ncert_data"
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    with zipfile.ZipFile("ncert.zip", "r") as z:
        z.extractall(EXTRACT_DIR)

    # load all PDFs
    docs = load_docs(EXTRACT_DIR)
    st.success(f"Loaded {len(docs)} PDFs")

    # split
    chunks = split_docs(docs)

    # ----------------------------------------------------------------
    # EMBEDDINGS + FAISS INDEX
    # ----------------------------------------------------------------
    st.info("Embedding chunks...")

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents([c["text"] for c in chunks])
    embedding_dim = len(embeddings[0])

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype="float32"))

    st.success("FAISS index built.")

    # ----------------------------------------------------------------
    # QUESTION GENERATION SECTION
    # ----------------------------------------------------------------
    st.subheader("Generate Questions")

    topic = st.text_input("Enter topic/chapter keyword:")
    num_q = st.number_input("How many questions?", 1, 20, 5)

    if st.button("Generate Questions"):
        if not topic.strip():
            st.warning("Enter a topic / chapter name.")
        else:
            # Retrieve relevant chunks
            model_embed = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model_embed.encode([topic], convert_to_numpy=True)
            D, I = index.search(q_emb.astype("float32"), k=min(5, len(chunks)))

            context = "\n\n".join(chunks[i]["text"] for i in I[0])

            # ------------------------------------------------------------
            # LOAD OFFLINE TRANSFORMER MODEL
            # ------------------------------------------------------------
            gen_model_name = "facebook/opt-125m"
            tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
            model = AutoModelForCausalLM.from_pretrained(gen_model_name)
            model.to("cpu")

            # ------------------------------------------------------------
            # PROMPT FOR HIGH-QUALITY COMPETITIVE QUESTIONS
            # ------------------------------------------------------------
            prompt = f"""
You are an expert NCERT teacher and UPSC-grade question maker.

Generate {num_q} **high-quality, complete, competitive-exam style questions** 
from the topic: **{topic}**

Rules for the questions:
- Must start with an interrogative word (What, Why, How, Explain, Describe…)
- Must end with a **question mark**
- Must be meaningful, complete and based ONLY on the context
- Level: Competitive exam (UPSC, State PSC, CBSE Boards)

Context:
{context}

Now generate the questions:
"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.4,
                    do_sample=True,
                )

            raw = tokenizer.decode(output[0], skip_special_tokens=True)

            # ------------------------------------------------------------
            # EXTRACT CLEAN QUESTIONS
            # ------------------------------------------------------------
            lines = raw.split("\n")
            final_q = []

            for ln in lines:
                ln = ln.strip()
                if len(ln.split()) < 4:
                    continue

                if re.match(r"^(What|Why|How|Explain|Describe|Discuss|Define|Evaluate|Examine|Analyze)\b", ln, re.I):
                    if not ln.endswith("?"):
                        ln += "?"
                    final_q.append(ln)

            # Limit to required count
            final_q = final_q[:num_q]

            # ------------------------------------------------------------
            # DISPLAY
            # ------------------------------------------------------------
            st.subheader("Generated Questions")
            for i, q in enumerate(final_q, 1):
                st.write(f"**{i}. {q}**")

