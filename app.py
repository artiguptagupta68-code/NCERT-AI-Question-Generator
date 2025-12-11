# app.py
"""
Streamlit-Cloud optimized NCERT -> UPSC Question Generator (offline)
- Upload or auto-download NCERT ZIP (contains PDFs)
- Choose subject (folder/name keyword)
- Enter topic/chapter + number of questions
- Retrieve topical context using SentenceTransformer + FAISS
- Generate N exam-style questions using an on-device transformer (lazy loaded)
- If transformer fails to produce distinct questions, deterministic fallback used
Notes:
- Heavy models are loaded lazily (only on Generate button click)
- Avoids langchain and other optional libs that caused import issues
"""
import os
import zipfile
import shutil
from pathlib import Path
import io
import re
import tempfile

import streamlit as st
import gdown
import numpy as np
import torch

from pypdf import PdfReader
try:
    import fitz  # PyMuPDF (optional faster extractor)
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------
# Config / defaults
# -----------------------
FILE_ID = os.getenv("NCERT_DRIVE_FILE_ID", "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G")
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"        # small & fast (sentence-transformers alias)
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-base")  # seq2seq generator
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

# interrogative start tokens (first word)
QUESTION_START_WORDS = [
    "What", "Why", "How", "Explain", "Describe", "State",
    "Define", "Discuss", "Examine", "Evaluate", "Compare"
]

# Streamlit UI setup
st.set_page_config(page_title="NCERT → UPSC Question Generator (Offline)", layout="wide")
st.title("NCERT → UPSC Question Generator (Offline)")
st.caption("Upload NCERT ZIP or auto-download, select subject, provide topic and generate exam-style questions.")

# -----------------------
# Utility functions
# -----------------------
def download_drive_file(file_id: str, out_path: str) -> bool:
    if os.path.exists(out_path):
        return True
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=True)
        return os.path.exists(out_path)
    except Exception as e:
        st.warning(f"Download failed: {e}")
        return False

def extract_zip_with_nested(zip_path: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
    except Exception as e:
        st.error(f"Failed to extract ZIP: {e}")
        return []
    # extract nested zips if any
    for root, _, files in os.walk(dest_dir):
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
    return sorted(os.listdir(dest_dir))

def read_pdf_text_pypdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""

def read_pdf_text_pymupdf(path: str) -> str:
    if not _HAS_PYMUPDF:
        return ""
    try:
        doc = fitz.open(path)
        texts = []
        for page in doc:
            try:
                texts.append(page.get_text())
            except Exception:
                continue
        return "\n".join(t for t in texts if t).strip()
    except Exception:
        return ""

def read_pdf_text(path: str) -> str:
    t = read_pdf_text_pypdf(path)
    if t and t.strip():
        return t
    if _HAS_PYMUPDF:
        t2 = read_pdf_text_pymupdf(path)
        if t2 and t2.strip():
            return t2
    return ""

def simple_clean_text(text: str) -> str:
    """Remove common NCERT noise and multiple blank lines"""
    text = re.sub(r"(Reprint|Re-?print)\s*\d{4}[-–]\d{2}", "", text, flags=re.I)
    text = re.sub(r"ISBN[\s:0-9\-]+", "", text, flags=re.I)
    text = re.sub(r"©[\s\S]{0,40}", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def chunk_text_simple_by_chars(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk long text by characters (simple deterministic splitter)"""
    chunks = []
    start = 0
    L = len(text)
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += step
    return chunks

def list_all_pdfs(base_dir: str):
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
        fname = Path(p).name.lower()
        if any(kw in part for part in parts) or (kw in fname):
            matched.append(p)
    return matched

# -----------------------
# Lazy model loaders (cached in-session)
# -----------------------
@st.cache_resource
def load_embedding_model():
    # sentence-transformers model name (small & fast)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def build_faiss_index(chunks_texts):
    # chunks_texts: list of dict {"doc_id","chunk_id","text"}
    if not chunks_texts:
        return None, []
    model = load_embedding_model()
    texts = [c["text"] for c in chunks_texts]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks_texts]
    return index, metadata

@st.cache_resource
def load_generator_model(model_name=GEN_MODEL_NAME):
    """Load seq2seq generator pipeline lazily (CPU on Streamlit Cloud)."""
    device = -1  # CPU default (Streamlit Cloud typically doesn't provide GPU)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen

# -----------------------
# Retrieval
# -----------------------
def retrieve_topk(index, embed_model, query, top_k=TOP_K_DEFAULT):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    k = min(top_k, index.ntotal) if index is not None else 0
    if k <= 0:
        return []
    D, I = index.search(q_emb, k)
    ids = I[0].tolist()
    return ids, D[0].tolist()

# -----------------------
# Prompt building + postprocessing
# -----------------------
def build_prompt_for_n_questions(context_snippets, topic, n):
    # Keep prompt strict: ask for numbered questions, start with interrogative words, end with '?'
    context = "\n\n".join(context_snippets).strip()
    prompt = (
        f"You are an expert NCERT + UPSC-style question setter.\n"
        f"Using ONLY the NCERT context below, generate exactly {n} distinct exam-style questions on the topic '{topic}'.\n"
        f"Rules:\n"
        f"- Each question must begin with What/Why/How/Explain/Describe/State/Define/Discuss/Examine/Evaluate/Compare.\n"
        f"- Each question must be a single line and end with a question mark '?'.\n"
        f"- Do NOT invent facts or names. Use only information present in the context.\n"
        f"- Produce numbered output 1. ... 2. ... etc. No extra explanations.\n\n"
        f"NCERT CONTEXT:\n{context}\n\n"
        f"Generate the questions now:\n"
    )
    return prompt

def extract_questions_from_text(raw_text, n):
    """
    Extract up to n distinct questions that begin with interrogatives and end with '?'
    """
    # Ensure only single-line questions captured
    # Pattern: start with interrogative word and capture until '?'
    pattern = r'(?:^|\n)\s*(?:' + '|'.join([re.escape(w) for w in QUESTION_START_WORDS]) + r')\b.*?\?'
    matches = re.findall(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = []
    seen = set()
    for m in matches:
        q = " ".join(m.strip().split())
        # remove leading numbering like "1. " if present
        q = re.sub(r'^\d+\.\s*', '', q)
        # Ensure single ending '?'
        q = q.rstrip(' .!?') + '?'
        # Normalize capitalization of first word (preserve rest)
        if q not in seen:
            seen.add(q)
            cleaned.append(q)
        if len(cleaned) >= n:
            break
    return cleaned

# -----------------------
# Deterministic fallback question generator (guarantees output)
# -----------------------
def fallback_generate_questions(context_snippets, topic, n):
    """
    If model fails, produce structured questions by selecting salient sentences
    and converting them into question templates (1/2/5 style).
    This produces meaningful, NCERT-grounded questions without LLM.
    """
    # join and split into sentences, prefer sentences containing topic tokens
    ctx = "\n\n".join(context_snippets)
    sents = re.split(r'(?<=[.!?])\s+', ctx)
    sents = [simple_clean_text(s) for s in sents if len(s.strip()) > 40]
    # pick top unique sentences, prefer those with topic tokens
    tokens = [t.lower() for t in re.split(r'\W+', topic) if t]
    scored = []
    for s in sents:
        s_low = s.lower()
        score = sum(1 for t in tokens if t and t in s_low)
        scored.append((score, len(s), s))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    questions = []
    i = 0
    while len(questions) < n and i < len(scored):
        sent = scored[i][2]
        # attempt to make a question
        q = None
        # definitional pattern
        if re.search(r'\bis\b|\bare\b|\brefers to\b|\bmeans\b', sent, flags=re.I):
            # get the part before 'is'
            m = re.match(r'^(.*?)\s+(is|are|refers to|means)\b', sent, flags=re.I)
            subj = m.group(1).strip() if m else None
            if subj and len(subj.split()) <= 10:
                q = f"What is {subj}?"
        if q is None:
            # cause/effect -> why
            if re.search(r'\bbecause\b|\bsince\b|\bdue to\b|\btherefore\b', sent, flags=re.I):
                short = " ".join(sent.split()[:20]) + "..."
                q = f"Why {short}?"
        if q is None:
            # list or includes
            if re.search(r'\bincludes\b|\bcomprise\b|\bconsists of\b', sent, flags=re.I):
                head = " ".join(sent.split()[:8]) + "..."
                q = f"What does {head} refer to?"
        if q is None:
            # default: state briefly (recall)
            short = " ".join(sent.split()[:14]) + ("..." if len(sent.split()) > 14 else "")
            q = f"State briefly: {short}?"
        # normalize
        if q:
            q = q.strip()
            if not q.endswith('?'):
                q = q.rstrip(' .!?') + '?'
            if q not in questions:
                questions.append(q)
        i += 1
    # pad if needed
    while len(questions) < n:
        questions.append("Model could not generate a distinct question. Try reducing the number or simplifying the topic.")
    return questions

# -----------------------
# UI: sidebar & file handling
# -----------------------
with st.sidebar:
    st.header("Options")
    use_drive = st.checkbox("Auto-download NCERT ZIP from Drive (FILE_ID)", value=False)
    uploaded_file = st.file_uploader("Or upload NCERT ZIP file", type="zip")
    st.write("Generator model (env var GEN_MODEL_NAME). Default:", GEN_MODEL_NAME)
    st.markdown("**Notes:** Models are loaded only when you click *Generate Questions*.")

col1, col2 = st.columns([3,1])
with col1:
    st.write("NCERT ZIP source (Drive FILE_ID):")
    st.write(f"`FILE_ID = {FILE_ID}`")
with col2:
    if st.button("Re-extract / Reset"):
        try:
            if os.path.exists(EXTRACT_DIR):
                shutil.rmtree(EXTRACT_DIR)
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
        except Exception:
            pass
        st.experimental_rerun()

# obtain ZIP
if use_drive:
    ok = download_drive_file(FILE_ID, ZIP_PATH)
    if not ok:
        st.warning("Auto-download failed — please upload ZIP manually.")
        uploaded_file = st.file_uploader("Upload NCERT ZIP file", type="zip")
elif uploaded_file:
    # save uploaded to ZIP_PATH
    with open(ZIP_PATH, "wb") as fo:
        fo.write(uploaded_file.getbuffer())
else:
    st.info("Either check 'Auto-download' in the sidebar or upload a ZIP to proceed.")
    st.stop()

# extract ZIP (lazy but done before subject selection)
if os.path.exists(EXTRACT_DIR):
    try:
        shutil.rmtree(EXTRACT_DIR)
    except Exception:
        pass

with st.spinner("Extracting NCERT ZIP (this may take a few seconds)..."):
    entries = extract_zip_with_nested(ZIP_PATH, EXTRACT_DIR)
st.success("Extraction complete.")

# discover PDFs and subject selection
all_pdfs = list_all_pdfs(EXTRACT_DIR)
st.info(f"Total PDFs discovered: {len(all_pdfs)}")

subject = st.selectbox("Select Subject (checks folder/name)", SUBJECTS)
matched_pdfs = filter_pdfs_for_subject(all_pdfs, subject)
if not matched_pdfs:
    st.warning(f"No auto-matched PDFs found for subject '{subject}'. Showing all PDFs (first 50).")
    # show first 50 paths so user can see where files are
    for p in all_pdfs[:50]:
        st.write(p)
    st.stop()

st.success(f"Found {len(matched_pdfs)} PDFs for subject '{subject}'")

# read and chunk matched PDFs lazily
with st.spinner("Reading matched PDFs (may take a few seconds)..."):
    docs = []
    for p in matched_pdfs:
        txt = read_pdf_text(p)
        if txt and txt.strip():
            docs.append({"doc_id": Path(p).name, "text": simple_clean_text(txt)})
# quick check
if not docs:
    st.error("No readable PDF text found for the chosen subject. Try uploading a different ZIP.")
    st.stop()

# chunk docs into chunks (list of dicts)
chunks = []
for d in docs:
    parts = chunk_text_simple_by_chars(d["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    for i, part in enumerate(parts):
        chunks.append({"doc_id": d["doc_id"], "chunk_id": f"{Path(d['doc_id']).stem}_chunk_{i}", "text": part})
st.info(f"Total chunks created for subject: {len(chunks)}")

# build FAISS index (lazy cached)
with st.spinner("Building embeddings + FAISS (cached) ..."):
    index, metadata = build_faiss_index(chunks)
st.success("FAISS index ready.")

# UI: topic and number of questions
st.header("Generate NCERT Questions")
topic = st.text_input("Enter chapter/topic (example: 'Constitution', 'Electricity')", "")
num_questions = st.number_input("Number of questions to generate", min_value=1, max_value=20, value=6)

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Please enter a chapter/topic.")
    else:
        with st.spinner("Retrieving top relevant chunks..."):
            ids, _ = retrieve_topk(index, load_embedding_model(), topic, top_k=TOP_K_DEFAULT)
            retrieved = [metadata[i] for i in ids if 0 <= i < len(metadata)]
        if not retrieved:
            st.warning("No relevant content found for this topic. Try a broader keyword.")
        else:
            st.markdown("### Retrieved snippets (top chunks):")
            context_snippets = []
            for r in retrieved:
                st.write(f"**{r['doc_id']} — {r['chunk_id']}**")
                st.write(r["text"][:600] + ("..." if len(r["text"]) > 600 else ""))
                st.write("---")
                context_snippets.append(r["text"])

            # Build smaller combined context for prompt
            context_for_prompt = "\n\n".join([cs[:1200] for cs in context_snippets])

            # Clean metadata-like tokens
            context_for_prompt = simple_clean_text(context_for_prompt)

            # Try generator model (lazy load)
            st.info("Loading generator model (this may take ~10-30s depending on environment)...")
            try:
                generator = load_generator_model()
            except Exception as e:
                st.warning(f"Failed to load generator model: {e}")
                generator = None

            questions = []
            if generator:
                prompt = build_prompt_for_n_questions(context_snippets[:TOP_K_DEFAULT], topic, num_questions)
                try:
                    # pipeline returns list of dicts
                    out = generator(prompt, max_length=512, do_sample=True, top_p=0.9, temperature=0.4)[0]["generated_text"]
                    questions = extract_questions_from_text(out, num_questions)
                except Exception as e:
                    st.warning(f"Model generation failed: {e}")
                    questions = []

            # If generator failed or produced too few distinct questions -> fallback deterministic generator
            if not questions or len(questions) < num_questions:
                st.info("Using fallback deterministic generator to ensure full set of questions.")
                fallback = fallback_generate_questions(context_snippets, topic, num_questions)
                # merge unique questions - prefer model-produced ones first
                merged = []
                seen = set()
                for q in (questions + fallback):
                    if q not in seen:
                        seen.add(q)
                        merged.append(q)
                    if len(merged) >= num_questions:
                        break
                questions = merged

            # Ensure final list length == num_questions
            if len(questions) < num_questions:
                # pad with message
                while len(questions) < num_questions:
                    questions.append("Model could not generate a distinct question. Try simplifying the topic or reducing the number.")

            # Display
            st.success(f"Generated {len(questions)} questions")
            for i, q in enumerate(questions, start=1):
                # Replace 'passage' with 'text' if present
                q = re.sub(r'\b[Pp]assage\b', 'text', q)
                st.write(f"**{i}. {q}**")

            st.markdown("### Sources used")
            for r in retrieved:
                st.write(f"- {r['doc_id']} — {r['chunk_id']}")
