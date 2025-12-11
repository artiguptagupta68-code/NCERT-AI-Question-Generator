# app.py
import os
import zipfile
import shutil
import re
from pathlib import Path
from typing import List, Dict

import streamlit as st
import gdown
import numpy as np
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import tempfile

# -----------------------
# CONFIG
# -----------------------
FILE_ID = os.getenv("NCERT_DRIVE_FILE_ID", "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G")
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
SUBJECTS = ["Economics", "Polity", "Business Studies", "Psychology", "Sociology"]
DEFAULT_TOP_K = 6

# Transformer generator (offline). Use a CPU-friendly model if GPU not available.
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "facebook/opt-125m")

# UI
st.set_page_config(page_title="NCERT → UPSC Question Generator (Offline)", layout="wide")
st.title("NCERT → UPSC Question Generator (Offline)")

faiss.omp_set_num_threads(1)

# -----------------------
# Helpers: zip download / extraction
# -----------------------
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

def extract_zip_with_nested(src_zip_path: str, dest_dir: str) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(src_zip_path, "r") as z:
            z.extractall(dest_dir)
    except Exception as e:
        st.warning(f"Main zip extraction failed: {e}")
        return []
    found_any = True
    while found_any:
        found_any = False
        for root, _, files in os.walk(dest_dir):
            for f in list(files):
                if f.lower().endswith(".zip"):
                    found_any = True
                    nested_zip = os.path.join(root, f)
                    folder_target = os.path.join(root, Path(f).stem)
                    os.makedirs(folder_target, exist_ok=True)
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
# PDF reading
# -----------------------
def read_pdf_pypdf(path: str) -> str:
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

def read_pdf_text(path: str) -> str:
    return read_pdf_pypdf(path)

# -----------------------
# Document loading & chunking
# -----------------------
def list_all_pdfs(base_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(root, f))
    return sorted(out)

def filter_pdfs_for_subject(all_pdf_paths: List[str], subject_keyword: str) -> List[str]:
    kw = subject_keyword.lower().strip()
    matched = []
    for p in all_pdf_paths:
        parts = [part.lower() for part in Path(p).parts]
        file_name = Path(p).name.lower()
        if any(kw in part for part in parts) or (kw in file_name):
            matched.append(p)
    return sorted(list(dict.fromkeys(matched)))

def load_docs_from_paths(pdf_paths: List[str]) -> List[Dict]:
    docs = []
    for path in pdf_paths:
        txt = read_pdf_text(path)
        if txt and txt.strip():
            docs.append({"path": path, "doc_id": Path(path).name, "text": txt})
        else:
            st.warning(f"Unreadable or image PDF skipped: {path}")
    return docs

def chunk_text_simple(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def docs_to_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []
    for d in docs:
        parts = chunk_text_simple(d["text"])
        for i, p in enumerate(parts):
            if len(p.split()) < 20:
                continue
            all_chunks.append({
                "doc_id": d["doc_id"],
                "path": d["path"],
                "chunk_id": f"{Path(d['doc_id']).stem}_chunk_{i}",
                "text": p
            })
    return all_chunks

# -----------------------
# FAISS index build & retrieve
# -----------------------
@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=True)
def build_faiss_for_subject(subject_name: str, chunks_texts: List[Dict]):
    if not chunks_texts:
        return None, []
    embed_model = load_embedding_model()
    texts = [c["text"] for c in chunks_texts]
    embeddings = embed_model.encode(texts, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    metadata = [{"doc_id": c["doc_id"], "path": c["path"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks_texts]
    return index, metadata

def retrieve_topk(index, embed_model, query: str, top_k: int = DEFAULT_TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    k = min(top_k, int(index.ntotal)) if index is not None else 0
    if k <= 0:
        return [], []
    D, I = index.search(q_emb, k)
    ids = I[0].tolist()
    return ids, D[0].tolist()

# -----------------------
# Generator loading (transformer, offline)
# -----------------------
@st.cache_resource
def load_generator_model(model_name: str = GEN_MODEL_NAME):
    # CPU fallback if GPU not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# -----------------------
# Prompt builder & postprocessing
# -----------------------
def build_prompt_for_questions(context_snippets: List[str], topic: str, num_total: int):
    """
    Build a strict prompt that asks the model to output numbered questions grouped into marks.
    """
    context = "\n\n".join(context_snippets)
    # distribute counts c1/c2/c5 (default priorities: more 1/2, then 5)
    c1 = min(2, num_total)
    c2 = min(2, max(0, num_total - c1))
    c5 = max(0, num_total - c1 - c2)
    prompt = f"""You are an expert UPSC-style question setter who uses only the NCERT context provided.
Produce exactly {c1} 1-mark, {c2} 2-mark and {c5} 5-mark questions based ONLY on the context below.
Rules:
- Each question must be a single line, begin with an interrogative or command (What, Why, How, Explain, Describe, State briefly, Define, Discuss, Examine, Evaluate) and end with a single question mark '?'.
- Use NCERT phrasing and do NOT invent facts.
- Keep 1-mark questions very short (recall/definition), 2-mark short explanatory/application, 5-mark analytical with example and significance.
Output exactly in this format (no extra text):

1-Mark Questions
1. ...
2. ...

2-Mark Questions
1. ...
2. ...

5-Mark Questions
1. ...
2. ...

NCERT CONTEXT (use only this):
Topic: {topic}

{context}

Now generate the questions.
"""
    return prompt, c1, c2, c5

def postprocess_generated_text(raw: str, c1:int, c2:int, c5:int, num_total:int):
    """
    Extract numbered questions from raw text returned by model.
    Ensure each question ends with '?' and starts with interrogative/command.
    """
    # find lines that look like numbered items
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # collect lines that contain a leading number like '1.' optionally with heading
    q_lines = []
    for ln in lines:
        # keep numbered lines or lines that start with interrogative
        if re.match(r'^\d+\.\s+', ln):
            q_lines.append(ln)
        elif re.match(r'^(What|Why|How|Explain|Describe|State|Define|Discuss|Examine|Evaluate)\b', ln, flags=re.I):
            q_lines.append(ln)
    # normalize: drop headings like '1-Mark Questions'
    filtered = []
    for ln in q_lines:
        # remove leading numbering
        ln2 = re.sub(r'^\d+\.\s*', '', ln).strip()
        # ensure ends with '?'
        if not ln2.endswith('?'):
            ln2 = ln2.rstrip('.!?') + '?'
        # ensure starts with interrogative or command - if not, prepend 'Explain: '
        if not re.match(r'^(What|Why|How|Explain|Describe|State|Define|Discuss|Examine|Evaluate)\b', ln2, flags=re.I):
            ln2 = "Explain briefly: " + ln2
            if not ln2.endswith('?'):
                ln2 = ln2.rstrip('.!?') + '?'
        filtered.append(ln2)
    # de-duplicate keeping order
    seen = set()
    final = []
    for q in filtered:
        if q not in seen:
            seen.add(q)
            final.append(q)
        if len(final) >= num_total:
            break
    return final

# -----------------------
# Export jsonl for fine-tuning (context -> question pairs)
# -----------------------
def export_jsonl_for_finetune(questions: List[str], context_snippets: List[str], path_out: str):
    pairs = []
    ctx = "\n\n".join(context_snippets)
    for q in questions:
        pairs.append({"prompt": ctx + "\n\nGenerate question:", "completion": " " + q})
    with open(path_out, "w", encoding="utf-8") as fo:
        for p in pairs:
            fo.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path_out

# -----------------------
# UI flow
# -----------------------
with st.sidebar:
    st.header("Settings")
    use_drive_zip = st.checkbox("Auto-download NCERT ZIP from Drive (FILE_ID)", value=False)
    file_upload = st.file_uploader("Or upload NCERT ZIP file", type="zip")
    st.write("Generator model (local):")
    st.write(f"`{GEN_MODEL_NAME}` (change via env var GEN_MODEL_NAME)")

col1, col2 = st.columns([3,1])
with col1:
    st.write("NCERT ZIP source (Drive FILE_ID):")
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

# obtain ZIP: either uploaded or download from Drive if user checked
if use_drive_zip:
    ok = download_drive_file(FILE_ID, ZIP_PATH)
    if not ok:
        st.error("Failed to download ZIP from Drive. Upload manually or check FILE_ID.")
        st.stop()
elif file_upload:
    with open(ZIP_PATH, "wb") as fo:
        fo.write(file_upload.getbuffer())
else:
    st.info("Either check 'Auto-download' on the left or upload a ZIP file to proceed.")
    st.stop()

# extract
if os.path.exists(EXTRACT_DIR):
    try:
        shutil.rmtree(EXTRACT_DIR)
    except Exception:
        pass

with st.spinner("Extracting ZIP and nested zips..."):
    entries = extract_zip_with_nested(ZIP_PATH, EXTRACT_DIR)
st.success("Extraction complete.")

# discover PDFs
all_pdfs = list_all_pdfs(EXTRACT_DIR)
st.info(f"Total PDFs found: {len(all_pdfs)}")
if len(all_pdfs) == 0:
    st.error("No PDFs found in extracted content.")
    st.stop()

# subject selection
subject = st.selectbox("Select subject", SUBJECTS)
matched_pdfs = filter_pdfs_for_subject(all_pdfs, subject)
st.info(f"PDFs matched to '{subject}': {len(matched_pdfs)}")
if len(matched_pdfs) == 0:
    st.warning("No matched files; using all PDFs")
    matched_pdfs = all_pdfs

# read & chunk
with st.spinner("Reading matched PDFs..."):
    docs = load_docs_from_paths(matched_pdfs)
st.success(f"Loaded {len(docs)} readable documents.")
if not docs:
    st.error("No readable text found.")
    st.stop()

with st.spinner("Chunking documents..."):
    chunks = docs_to_chunks(docs)
st.info(f"Total chunks created: {len(chunks)}")
if not chunks:
    st.error("No chunks created; adjust CHUNK_SIZE.")
    st.stop()

# build faiss
with st.spinner("Building FAISS index..."):
    index, metadata = build_faiss_for_subject(subject, chunks)
st.success("FAISS index ready.")

# user inputs
topic = st.text_input("Enter topic/chapter keyword (e.g., 'constitution'):")
num_q = st.number_input("Total questions to generate (will be divided into 1/2/5 mark buckets)", min_value=1, max_value=20, value=6)
top_k = st.number_input("Top-K chunks to retrieve", min_value=1, max_value=50, value=DEFAULT_TOP_K)

if st.button("Generate Questions (offline transformer)"):
    if not topic.strip():
        st.warning("Enter a topic/keyword.")
    else:
        with st.spinner("Retrieving top chunks..."):
            ids, _ = retrieve_topk(index, load_embedding_model(), topic, top_k=top_k)
            retrieved_meta = [metadata[idx] for idx in ids if 0 <= idx < len(metadata)]
        if not retrieved_meta:
            st.warning("No relevant chunks found. Try a broader keyword.")
        else:
            st.markdown("### Retrieved context snippets")
            context_snippets = []
            for m in retrieved_meta:
                snippet = m["text"][:500].strip()
                st.write(f"- **{m['doc_id']} — {m['chunk_id']}**")
                st.write(snippet + "...")
                st.write("---")
                context_snippets.append(m["text"])

            # load generator model
            st.info("Loading generator model (this may take a moment)...")
            tokenizer, gen_model, device = load_generator_model(GEN_MODEL_NAME)

            # build prompt
            prompt, c1, c2, c5 = build_prompt_for_questions(context_snippets[:top_k], topic, num_q)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                out_ids = gen_model.generate(**inputs,
                                             max_new_tokens=600,
                                             temperature=0.25,
                                             do_sample=False,
                                             num_beams=2,
                                             eos_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # raw may include prompt; try to remove prompt prefix
            # find substring after 'Now generate the questions.' or after 'NCERT CONTEXT'
            if "Now generate the questions" in raw:
                raw_out = raw.split("Now generate the questions")[-1]
            elif "1-Mark Questions" in raw:
                raw_out = raw.split("1-Mark Questions", 1)[-1]
                raw_out = "1-Mark Questions\n" + raw_out
            else:
                raw_out = raw

            # postprocess to pick final questions
            final_questions = postprocess_generated_text(raw_out, c1, c2, c5, int(num_q))
            if not final_questions:
                st.warning("Model didn't produce recognizable questions. Showing raw output below for debugging.")
                st.text(raw)
            else:
                # bucket them: first c1 -> 1-mark, next c2 -> 2-mark, next c5 -> 5-mark
                st.success("Questions generated (offline).")
                st.markdown("### 1-Mark Questions")
                for i, q in enumerate(final_questions[:c1], 1):
                    st.write(f"{i}. {q}")

                st.markdown("### 2-Mark Questions")
                for i, q in enumerate(final_questions[c1:c1+c2], 1):
                    st.write(f"{i}. {q}")

                st.markdown("### 5-Mark Questions")
                for i, q in enumerate(final_questions[c1+c2:c1+c2+c5], 1):
                    st.write(f"{i}. {q}")

                # Export JSONL for finetuning
                if st.button("Export JSONL for fine-tuning (context -> question pairs)"):
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
                    path_out = tmpf.name
                    export_jsonl_for_finetune(final_questions, context_snippets[:top_k], path_out)
                    st.success("JSONL created.")
                    with open(path_out, "rb") as fo:
                        st.download_button("Download JSONL", fo, file_name=f"{topic}_questions.jsonl")
