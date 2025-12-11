# app.py
import os, zipfile, shutil, re
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

# ----------------- CONFIG -----------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"  # Cloud-safe
TOP_K = 4
SUBJECTS = ["Polity", "Sociology", "Psychology", "Business Studies", "Economics"]
QUESTION_START_WORDS = ["What","Why","How","Explain","Describe","State","Define","Discuss","Examine","Evaluate","Tell"]

st.set_page_config(page_title="NCERT AI Question Generator", layout="wide")
st.title("📘 NCERT AI Question Generator")
st.caption("Generate UPSC-style NCERT questions (RAG + Transformers)")

# ----------------- UTILS -----------------
def download_zip(file_id, out_path):
    if os.path.exists(out_path): return True
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", out_path, quiet=False)
        return os.path.exists(out_path)
    except:
        return False

def extract_zip(zip_path, extract_to):
    shutil.rmtree(extract_to, ignore_errors=True)
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path,"r") as z:
        z.extractall(extract_to)

def read_pdf_text(path):
    text = ""
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            t = p.extract_text()
            if t: text += t + "\n"
    except:
        pass
    if not text.strip() and _HAS_PYMUPDF:
        try:
            doc = fitz.open(path)
            for page in doc:
                t = page.get_text()
                if t: text += t + "\n"
        except: pass
    return text

def load_docs_by_subject(folder, subject):
    subject = subject.lower()
    docs = []
    for root,_,files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                path_lower = Path(root).name.lower() + " " + f.lower()
                if subject in path_lower:
                    text = read_pdf_text(os.path.join(root,f))
                    if text.strip():
                        docs.append({"doc_id":f,"text":text})
    return docs

def clean_ncert_text(text):
    patterns = [r"Reprint\s*\d{4}-\d{2}", r"ISBN[\s:0-9-]+", r"NCERT[\s\S]{0,50}Publication",
                r"Not for commercial use", r"Government of India", r"Publication Division",
                r"©[\s\S]{0,30}", r"[A-Za-z ]+ Rao", r"[A-Za-z ]+ Singh"]
    for p in patterns: text = re.sub(p,"",text,flags=re.IGNORECASE)
    return re.sub(r"\n\s*\n","\n",text).strip()

def chunk_documents(docs):
    chunks=[]
    for doc in docs:
        start=0
        text=doc["text"]
        doc_id=doc["doc_id"]
        while start<len(text):
            end=min(start+CHUNK_SIZE,len(text))
            chunks.append({"doc_id":doc_id,"chunk_id":f"{Path(doc_id).stem}_{start}","text":text[start:end]})
            start += CHUNK_SIZE-CHUNK_OVERLAP
    return chunks

@st.cache_resource
def build_faiss_index(chunks):
    if not chunks: return None,None,None
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode([c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    metadata = [{"doc_id":c["doc_id"],"chunk_id":c["chunk_id"],"text":c["text"]} for c in chunks]
    return model,index,metadata

@st.cache_resource
def load_generator_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    if device==0: model=model.to("cuda")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

def retrieve_chunks(query,index,metadata,top_k=TOP_K):
    if index is None or metadata is None: return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D,I = index.search(q_emb, min(top_k,index.ntotal))
    return [metadata[idx] for idx in I[0] if idx<len(metadata)]

def extract_questions(text,num_questions):
    pattern=r'(?:'+'|'.join(QUESTION_START_WORDS)+r').*?\?'
    matches = re.findall(pattern,text,flags=re.IGNORECASE|re.DOTALL)
    seen,set_final=set(),[]
    final=[]
    for q in matches:
        q_clean=q.strip()
        if q_clean not in seen:
            seen.add(q_clean)
            final.append(q_clean)
        if len(final)>=num_questions: break
    return final

def generate_n_questions(generator,topic,context_text,num_questions):
    questions=[]
    attempts,max_attempts=0,num_questions*5
    while len(questions)<num_questions and attempts<max_attempts:
        prompt=f"""
You are an expert UPSC/NCERT question setter.
Generate ONE meaningful question strictly based on the following NCERT context.
Topic: {topic}
Context:
{context_text}
Rules:
- Use only NCERT facts
- Do not use generic words like 'passage' or 'text'
- End with '?'
Output:
"""
        try:
            out=generator(prompt,max_length=200,do_sample=True,temperature=0.6,top_p=0.9)[0]["generated_text"]
        except: out=""
        new_qs=extract_questions(out,1)
        if new_qs and new_qs[0] not in questions:
            questions.append(new_qs[0])
        attempts+=1
    while len(questions)<num_questions:
        questions.append("Could not generate distinct question. Try reducing number.")
    return questions

# ----------------- ORCHESTRATION -----------------
st.text("Preparing NCERT content...")
if not download_zip(FILE_ID, ZIP_PATH):
    st.error("Failed to download NCERT ZIP.")
    st.stop()
if not zipfile.is_zipfile(ZIP_PATH):
    st.error("ZIP file invalid.")
    st.stop()
extract_zip(ZIP_PATH, EXTRACT_DIR)
st.success(f"NCERT extracted to {EXTRACT_DIR}")

subject=st.selectbox("Select Subject",SUBJECTS)
docs=load_docs_by_subject(EXTRACT_DIR,subject)
st.info(f"Loaded {len(docs)} PDFs")
if not docs: st.stop()
all_chunks=chunk_documents(docs)
st.info(f"Total chunks: {len(all_chunks)}")
embed_model,index,metadata=build_faiss_index(all_chunks)
st.success("FAISS ready")
generator=load_generator_pipeline()
st.success("Generator loaded")

# ----------------- USER INTERFACE -----------------
st.subheader("Generate NCERT Questions")
topic = st.text_input("Enter chapter/topic (example: 'Constitution', 'Electricity')")
num_questions = st.number_input("Number of questions", min_value=1,max_value=10,value=5)

if st.button("Generate Questions"):
    if not topic.strip():
        st.warning("Enter a valid topic")
    else:
        retrieved_chunks = retrieve_chunks(topic,index,metadata)
        if not retrieved_chunks:
            st.warning(f"No content found for {topic}")
        else:
            context_text = "\n\n".join([clean_ncert_text(r["text"][:1200]) for r in retrieved_chunks])
            final_questions = generate_n_questions(generator,topic,context_text,num_questions)
            st.success(f"Generated {len(final_questions)} questions")
            for i,q in enumerate(final_questions,1):
                st.write(f"{i}. {q}")
            st.write("### Sources used")
            for r in retrieved_chunks:
                st.write(f"{r['doc_id']} — {r['chunk_id']}")
