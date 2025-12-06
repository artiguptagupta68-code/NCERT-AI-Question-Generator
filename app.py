import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# OCR libraries
from pdf2image import convert_from_path
import pytesseract

# ----------------------------------
# IMPORTANT FAISS FIX
# ----------------------------------
faiss.omp_set_num_threads(1)

# SUBJECT FOLDERS
SUBJECT_FOLDERS = {
    "Economics": "ncert_extracted/economics",
    "Psychology": "ncert_extracted/psychology",
    "Sociology": "ncert_extracted/sociology",
    "Polity": "ncert_extracted/polity",
    "Business Studies": "ncert_extracted/businessstudies"
}

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# --------------------------------------
# LOAD PDF TEXT (with OCR fallback)
# --------------------------------------
def load_pdfs(subject_path):
    documents = []

    for root, _, files in os.walk(subject_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, f)
                text = extract_text_from_pdf(pdf_path)
                if text.strip():
                    documents.append(text)
    return documents

def extract_text_from_pdf(pdf_path):
    """
    Tries normal PDF text extraction first, then falls back to OCR if empty.
    """
    # 1️⃣ Try PyPDF2
    try:
        pdf = PdfReader(pdf_path)
        text = "".join([page.extract_text() or "" for page in pdf.pages])
        if text.strip():
            return text
    except:
        pass

    # 2️⃣ Fallback to OCR
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except:
        return ""  # If all fails

# --------------------------------------
# CHUNKING
# --------------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def clean_chunks(chunks):
    return [c for c in chunks if c.strip() != "" and len(c.split()) > 3]

# --------------------------------------
# EMBEDDING + FAISS INDEX
# --------------------------------------
def embed_and_build_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)

    cleaned_chunks = []
    cleaned_embeddings = []

    for emb, ch in zip(embeddings, chunks):
        if np.linalg.norm(emb) > 0.1:
            cleaned_chunks.append(ch)
            cleaned_embeddings.append(emb)

    embeddings = np.array(cleaned_embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, cleaned_chunks

# --------------------------------------
# RETRIEVAL
# --------------------------------------
def retrieve(query, index, chunks, top_k=5):
    query_emb = model.encode(query, convert_to_numpy=True)

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    query_emb = query_emb.astype("float32")

    distances, indices = index.search(query_emb, top_k)

    retrieved = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
    return retrieved

# --------------------------------------
# QUESTION GENERATION
# --------------------------------------
def generate_questions(context_list, topic, num_q=5):
    if not context_list:
        return "No relevant content found to generate questions."

    questions = []
    for i in range(num_q):
        chunk = context_list[i % len(context_list)]
        q = f"Q{i+1}. Based on the topic '{topic}', explain in detail: {chunk[:200]}..."
        questions.append(q)

    return "\n\n".join(questions)

# --------------------------------------
# STREAMLIT UI
# --------------------------------------
st.title("NCERT AI Question Generator 🔍")
st.write("Generate subjective questions from NCERT context (supports scanned PDFs).")

subject = st.selectbox("Select Subject", list(SUBJECT_FOLDERS.keys()))
subject_path = SUBJECT_FOLDERS[subject]

query_topic = st.text_input("Enter Topic or Chapter Name:")
num_questions = st.number_input("Number of Questions", 1, 20, 5)
top_k = st.number_input("Retrieval Chunk Count", 1, 20, 5)

if st.button("Generate Questions"):
    st.write(f"### Selected Subject: {subject}")

    # Load PDF Data
    documents = load_pdfs(subject_path)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(clean_chunks(chunk_text(doc)))

    if not all_chunks:
        st.error("No valid text found in PDFs! (Check if PDFs are scanned images)")
    else:
        index, chunks = embed_and_build_index(all_chunks)
        retrieved = retrieve(query_topic, index, chunks, top_k)

        st.write("### Retrieved Context Chunks:")
        st.write(retrieved)

        final_questions = generate_questions(retrieved, query_topic, num_questions)

        st.write("### Generated Questions:")
        st.write(final_questions)
