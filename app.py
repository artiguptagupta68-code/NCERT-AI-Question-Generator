
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
uploaded_file = st.file_uploader("Upload NCERT ZIP file", type="zip")

subject = st.selectbox("Select Subject", SUBJECTS)
num_q = st.number_input("Number of total questions", min_value=1, max_value=20, value=6)

if uploaded_file:
    zip_path = os.path.join("/tmp", uploaded_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------- Extract ZIP ----------------
    extract_folder = "/tmp/ncert_extracted"
    os.makedirs(extract_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Extract nested ZIPs
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_path = os.path.join(root, file.replace(".zip", ""))
                os.makedirs(nested_extract_path, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(nested_extract_path)

    # ---------------- Load documents ----------------
    def load_documents(folder, subject_keyword):
        texts = []
        for root, dirs, files in os.walk(folder):
            if subject_keyword.lower() in root.lower():
                for file in files:
                    path = os.path.join(root, file)
                    if file.lower().endswith(".pdf"):
                        try:
                            doc = fitz.open(path)
                            text = ""
                            for page in doc:
                                text += page.get_text()
                            texts.append(text)
                        except Exception as e:
                            st.warning(f"Failed to read PDF {path}: {e}")
                    elif file.lower().endswith(".txt"):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                texts.append(f.read())
                        except Exception as e:
                            st.warning(f"Failed to read TXT {path}: {e}")
        return texts

    texts = load_documents(extract_folder, subject)
    
    if len(texts) == 0:
        st.warning("No PDF/TXT files found for this subject!")
    else:
        # ---------------- Split documents ----------------
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(" ".join(texts))

        # ---------------- Embeddings & FAISS ----------------
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedding_model.embed_documents(chunks)

        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        embedding_matrix = np.array(embeddings).astype("float32")
        index.add(embedding_matrix)

        faiss_index = {
            "index": index,
            "chunks": chunks
        }

        # ---------------- User input ----------------
        query_topic = st.text_input("Enter topic/chapter (keyword) to generate questions:")

        if st.button("Generate Questions"):
            if not query_topic.strip():
                st.warning("Enter a topic keyword.")
            else:
                # Retrieve top K relevant chunks
                model_embed = SentenceTransformer("all-MiniLM-L6-v2")
                query_embedding = model_embed.encode([query_topic], convert_to_numpy=True)
                D, I = index.search(query_embedding.astype("float32"), k=min(5, len(chunks)))
                retrieved_chunks = [chunks[i] for i in I[0]]
                context = "\n\n".join(retrieved_chunks)

                # ---------------- Transformer-based offline question generator ----------------
                # CPU-friendly model
                llm_model_name = "facebook/opt-125m"
                tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
                llm_model.to("cpu")

                # Build prompt for multiple questions
                prompt = f"""
You are an expert NCERT question setter.

Topic: {query_topic}

Based on the context below, generate {num_q} NCERT-style questions (1-mark, 2-mark, 5-mark mixed). 
Each question must start with an interrogative or command verb and end with '?'

Context:
{context}
"""
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = llm_model.generate(
                        **inputs,
                        max_new_tokens=600,
                        temperature=0.25,
                        do_sample=True
                    )
                result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # ---------------- Clean & format questions ----------------
                lines = [ln.strip() for ln in result_text.splitlines() if ln.strip()]
                questions = []
                for ln in lines:
                    if re.match(r'^\d+\.\s+', ln) and not ln.endswith('?'):
                        ln = ln.rstrip(' .') + '?'
                    questions.append(ln)

                st.subheader("Generated Questions:")
                for q in questions:
                    st.write(q)
