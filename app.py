# app.py
import os
import zipfile
import shutil
from pathlib import Path
import re

import streamlit as st
import gdown
import torch

from pypdf import PdfReader
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    _HAS_PYMUPDF = False

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = os.getenv("NCERT_DRIVE_FILE_ID", "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G")
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-base")
TOP_K = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]

st.set_page_config(page_title="NCERT → UPSC Question Generator", layout="wide")
st.title("NCERT → UPSC Question Generator (Offline)")

# ----------------------------
# Utilities
# ----------------------------
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

def extract_zip_with_nested(src_zip_path: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(src_zip_path, "r") as z:
            z.extractall(dest_dir)
    except Exception as e:
        st.error(f"Zip extraction failed: {e}")
        return []
    # nested zips
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
                    except Exception:
                        pass
                    try:
                        os.remove(nested_zip)
                    except Exception:
                        pass
    return sorted(os.listdir(dest_dir))

# ----------------------------
# Read PDFs
# ----------------------------
def read_pdf_pypdf(path: str) -> str:
    try:
        r = PdfReader(path)
        parts = []
        for p in r.pages:
            try:
                t = p.extract_text()
                if t:
                    parts.append(t)
            except Exception:
                continue
        return "\n".join(parts).strip()
    except Exception:
        return ""

def read_pdf_pymupdf(path: str) -> str:
    if not _HAS_PYMUPDF:
        return ""
    try:
        import fitz
        doc = fitz.open(path)
        text = []
        for page in doc:
            try:
                t = page.get_text()
                if t:
                    text.append(t)
            except Exception:
                continue
        return "\n".join(text).strip()
    except Exception:
        return ""

def read_pdf_text(path: str) -> str:
    t = read_pdf_pypdf(path)
    if t and t.strip():
        return t
    return read_pdf_pymupdf(path)

# ----------------------------
# Load documents by subject
# ----------------------------
def load_docs_by_subject(base_dir: str, subject_keyword: str):
    subject_keyword = subject_keyword.lower()
    docs = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                combined = (Path(root).name + " " + f).lower()
                if subject_keyword in combined:
                    p = os.path.join(root, f)
                    t = read_pdf_text(p)
                    if t and t.strip():
                        docs.append({"doc_id": f, "text": t})
    return docs

# ----------------------------
# Chunking
# ----------------------------
def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    for doc in docs:
        text = doc["text"]
        doc_id = doc["doc_id"]
        i = 0
        start = 0
        step = chunk_size - chunk_overlap
        while start < len(text):
            end = min(start + chunk_size, len(text))
            segment = text[start:end].strip()
            if len(segment.split()) > 30:
                chunks.append({"doc_id": doc_id, "chunk_id": f"{Path(doc_id).stem}_{i}", "text": segment})
                i += 1
            start += step
    return chunks

# ----------------------------
# Embeddings + FAISS
# ----------------------------
@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_resource
def build_faiss(chunks):
    if not chunks:
        return None, None, None
    model = load_embedding_model()
    texts = [c["text"] for c in chunks]
    embeds = model.encode(texts, convert_to_numpy=True).astype("float32")
    dim = embeds.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)
    metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    return model, index, metadata

def retrieve_topk(index, embed_model, query, top_k=TOP_K):
    if index is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    k = min(top_k, int(index.ntotal))
    D, I = index.search(q_emb, k)
    return [I[0][i] for i in range(k)], D[0].tolist()

# ----------------------------
# Generator
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_generator(model_name=GEN_MODEL_NAME):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == 0:
        model = model.to("cuda")
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen, tokenizer

# ----------------------------
# Clean NCERT text
# ----------------------------
def clean_ncert_text(text: str) -> str:
    patterns = [
        r"Reprint\s*\d{4}-\d{2}",
        r"ISBN[\s:0-9-]+",
        r"©[\s\S]{0,80}",
        r"Not for commercial use",
        r"Publication Division",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Extract questions from generated text
# ----------------------------
QUESTION_START = r"(What|Why|How|Explain|Describe|State|Define|Discuss|Examine|Evaluate|Analyse|Analyze|Compare|Differentiate)\b"

def extract_questions_from_text(text: str, needed: int):
    candidates = []
    regex = re.compile(QUESTION_START + r".*?\?", flags=re.I | re.S)
    for m in regex.finditer(text):
        q = m.group(0).strip()
        if len(q.split()) > 3:
            candidates.append(q)
        if len(candidates) >= needed:
            break
    # fallback
    if len(candidates) < needed:
        sents = re.split(r'(?<=[\.\?\!])\s+', text)
        for s in sents:
            s_clean = s.strip()
            if re.match(QUESTION_START, s_clean, flags=re.I) and s_clean.endswith('?') and len(s_clean.split()) > 3:
                if s_clean not in candidates:
                    candidates.append(s_clean)
            if len(candidates) >= needed:
                break
    seen = set()
    final = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            final.append(q)
        if len(final) >= needed:
            break
    return final

# ----------------------------
# Generate subjective questions
# ----------------------------
def generate_n_subjective_questions(gen_pipeline, tokenizer, topic, context_text, n):
    max_new_tokens = 128
    max_input = getattr(tokenizer, "model_max_length", 512)
    reserve = 64
    max_context_chars = int(max_input - reserve)
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars]

    questions = []
    tries = 0
    max_tries = n * 8

    while len(questions) < n and tries < max_tries:
        prompt = (
            "You are an expert NCERT/UPSC-style question setter. "
            "Using ONLY the NCERT context provided, generate ONE **subjective/descriptive** exam-quality question.\n"
            "Strict rules:\n"
            "- Use only facts present in the context.\n"
            "- Begin with an interrogative or command word.\n"
            "- Make it suitable for a descriptive answer.\n"
            "- End the question with '?' and make it complete.\n"
            f"Topic: {topic}\n\n"
            f"Context:\n{context_text}\n\n"
            "Output exactly one question and nothing else:\n"
        )
        try:
            out = gen_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.6)[0]["generated_text"]
        except Exception as e:
            st.warning(f"Generation call failed: {e}")
            break

        extracted = extract_questions_from_text(out, 1)
        if extracted:
            q = extracted[0].replace("passage", "text").strip()
            if q.endswith('?') and len(q.split()) > 4 and q not in questions:
                questions.append(q)
        tries += 1

    while len(questions) < n:
        questions.append("Model could not generate a distinct question. Try reducing N or simplifying topic.")

    return questions

# ----------------------------
# Streamlit UI
# ----------------------------
with st.sidebar:
    st.header("Settings")
    use_drive = st.checkbox("Auto-download NCERT ZIP from Drive (FILE_ID)", value=True)
    uploaded = st.file_uploader("Or upload NCERT ZIP file", type="zip")
    st.write("Generator model (env GEN_MODEL_NAME):", GEN_MODEL_NAME)

col1, col2 = st.columns([3,1])
with col1:
    st.write("NCERT Drive FILE_ID:")
    st.write(f"`{FILE_ID}`")
with col2:
    if st.button("Redownload & Re-extract ZIP"):
        for p in (EXTRACT_DIR, ZIP_PATH):
            try:
                if os.path.exists(p):
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
            except Exception:
                pass
        st.experimental_rerun()

# Obtain ZIP
if uploaded:
    with open(ZIP_PATH, "wb") as fo:
        fo.write(uploaded.getbuffer())
elif use_drive:
    ok = download_drive_file(FILE_ID, ZIP_PATH)
    if not ok:
        st.error("Failed to download ZIP. Upload manually or check FILE_ID.")
        st.stop()
else:
    st.info("Upload a ZIP file or enable auto-download from Drive.")
    st.stop()

# Extract
with st.spinner("Extracting ZIP..."):
    extract_zip_with_nested(ZIP_PATH, EXTRACT_DIR)
st.success("Extraction done.")

# List PDFs
all_pdfs = [str(p) for p in Path(EXTRACT_DIR).rglob("*.pdf")]
st.info(f"Total PDFs discovered: {len(all_pdfs)}")
if len(all_pdfs) == 0:
    st.error("No PDFs found in extracted folder.")
    st.stop()

# Select subject
subject = st.selectbox("Select Subject", SUBJECTS)
docs = load_docs_by_subject(EXTRACT_DIR, subject)
st.info(f"Loaded {len(docs)} PDFs for subject '{subject}'")
if not docs:
    st.warning(f"No readable PDFs found for {subject}.")
    st.stop()

# Chunk documents
with st.spinner("Chunking documents..."):
    all_chunks = chunk_documents(docs)
st.info(f"Total chunks created: {len(all_chunks)}")
if not all_chunks:
    st.error("No chunks created.")
    st.stop()

# Build FAISS index
with st.spinner("Building FAISS index..."):
    embed_model, index, metadata = build_faiss(all_chunks)
st.success("FAISS index ready.")

# Load generator
with st.spinner("Loading generator model..."):
    generator_pipeline, tokenizer = load_generator(GEN_MODEL_NAME)
st.success("Generator ready.")

# User inputs
st.subheader("Generate NCERT Subjective Questions")
topic = st.text_input("Enter chapter/topic (example: 'Constitution')").strip()
num_q = st.number_input("Number of questions to generate", min_value=1, max_value=20, value=5)

if st.button("Generate Questions"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        # Retrieve top-k chunks
        ids, _ = retrieve_topk(index, embed_model, topic, top_k=TOP_K)
        retrieved = [metadata[i] for i in ids if i < len(metadata)]
        if not retrieved:
            st.warning("No relevant chunks found. Try a broader keyword.")
        else:
            st.markdown("### Sources (top chunks):")
            context_parts = []
            for r in retrieved:
                st.write(f"- {r['doc_id']} — {r['chunk_id']}")
                st.write(r['text'][:600] + "...")
                context_parts.append(r["text"])
            context_text = "\n\n".join(context_parts)
            context_text = clean_ncert_text(context_text)

            with st.spinner("Generating subjective questions..."):
                questions = generate_n_subjective_questions(generator_pipeline, tokenizer, topic, context_text, int(num_q))

            st.success(f"Generated {len(questions)} questions")
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q}")
