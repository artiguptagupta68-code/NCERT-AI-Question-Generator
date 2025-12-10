import streamlit as st
import os
import zipfile
import shutil
from pathlib import Path
import numpy as np
import torch
import re
import fitz  # PyMuPDF
from pypdf import PdfReader

import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------------
st.set_page_config(page_title="📘 NCERT Offline Question Generator", layout="wide")
st.title("📘 NCERT Offline Question Generator (Transformer Based)")
st.caption("Generates competitive-exam style questions using RAG + Transformers (offline).")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --------------------------------------------------------------------
# PDF TEXT PARSERS
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
                fpath = os.path.join(root, f)
                text = read_pdf_text(fpath)
                if text.strip():
                    docs.append({"doc_id": f, "text": text})
                else:
                    st.warning(f"Skipped unreadable PDF: {f}")
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
st.subheader("Upload NCERT ZIP File")
uploaded_zip = st.file_uploader("Upload a ZIP containing NCERT PDFs", type=["zip"])

if uploaded_zip:
    # save ZIP
    with open("ncert.zip", "wb") as f:
        f.write(uploaded_zip.read())

    # extract
    EXTRACT_DIR = "ncert_data"
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    with zipfile.ZipFile("ncert.zip", "r") as z:
        z.extractall(EXTRACT_DIR)

    st.success("ZIP extracted successfully.")

    # load PDFs
    docs = load_docs(EXTRACT_DIR)
    st.info(f"Loaded {len(docs)} PDFs.")

    # chunking
    chunks = split_docs(docs)
    st.info(f"Created {len(chunks)} text chunks.")

    # ----------------------------------------------------------------
    # EMBEDDINGS + FAISS
    # ----------------------------------------------------------------
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chunk_texts = [c["text"] for c in chunks]

    st.info("Embedding chunks...")
    embeddings = embedder.embed_documents(chunk_texts)

    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(embeddings, dtype="float32"))

    st.success("FAISS index ready.")

    # ----------------------------------------------------------------
    # QUESTION GENERATION
    # ----------------------------------------------------------------
    st.subheader("Generate Questions")

    topic = st.text_input("Enter topic/chapter keyword:")
    num_q = st.number_input("Number of questions:", 1, 20, 5)

    if st.button("Generate Questions"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            # find relevant chunks
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = embed_model.encode([topic], convert_to_numpy=True)
            D, I = index.search(q_emb.astype("float32"), k=min(5, len(chunks)))

            context = "\n\n".join(chunks[i]["text"] for i in I[0])

            # load transformer
            gen_model_name = "facebook/opt-125m"
            tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
            model = AutoModelForCausalLM.from_pretrained(gen_model_name)
            model.to("cpu")

            # prompt
            prompt = f"""
You are an expert NCERT teacher and UPSC-level examiner.

Generate {num_q} competitive-exam style questions based on the topic: {topic}

Requirements:
- Each question must start with an interrogative word (What, Why, How, Explain, Describe, Discuss…)
- Each question must end with a question mark
- Questions must be complete, meaningful, and based on the context
- Difficulty: UPSC / State PSC / CBSE Board level

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

            # extract clean questions
            lines = raw.split("\n")
            final_q = []

            for ln in lines:
                ln = ln.strip()

                # avoid useless lines
                if len(ln.split()) < 4:
                    continue

                # must start with interrogative
                if re.match(r"^(What|Why|How|Explain|Describe|Discuss|Define|Evaluate|Examine|Analyze)\b", ln, re.I):
                    if not ln.endswith("?"):
                        ln += "?"
                    final_q.append(ln)

            final_q = final_q[:num_q]

            # display
            st.subheader("Generated Questions")
            if len(final_q) == 0:
                st.warning("Model could not extract proper questions. Try a different topic.")
            else:
                for i, q in enumerate(final_q, 1):
                    st.write(f"**{i}. {q}**")
