import streamlit as st
import zipfile
import os
import fitz
import numpy as np
import faiss
import torch
import re

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="📚 AI NCERT UPSC Question Generator", layout="wide")
st.title("📚 AI NCERT UPSC Question Generator (Offline, OPT-125M)")


# -----------------------------------------------------------
# 1. Upload ZIP
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload NCERT ZIP file", type="zip")

if uploaded_file:

    zip_path = os.path.join("/tmp", uploaded_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_dir = "/tmp/ncert_extracted"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Extract nested ZIPs
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".zip"):
                nested = os.path.join(root, file)
                out_dir = nested.replace(".zip", "")
                os.makedirs(out_dir, exist_ok=True)
                with zipfile.ZipFile(nested, "r") as z2:
                    z2.extractall(out_dir)


    # -----------------------------------------------------------
    # 2. Load Documents

    def load_documents(folder):
        texts = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    try:
                        doc = fitz.open(path)
                        text = ""
                        for pg in doc:
                            text += pg.get_text()
                        texts.append(text)
                    except:
                        pass
                elif file.endswith(".txt"):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            texts.append(f.read())
                    except:
                        pass
        return texts


    docs = load_documents(extract_dir)

    if len(docs) == 0:
        st.error("No PDFs/Text found!")
        st.stop()


    # -----------------------------------------------------------
    # 3. Split into chunks
    -----------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(" ".join(docs))


    # -----------------------------------------------------------
    # 4. Embeddings + FAISS
    -----------------------------------------------------------
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    emb = embedder.embed_documents(chunks)

    dim = len(emb[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(emb).astype("float32"))


    # -----------------------------------------------------------
    # 5. Query Input
    -----------------------------------------------------------
    query = st.text_input("Enter a topic or question to generate UPSC-style questions:")

    if query:

        # 5a. Retrieve chunks
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = encoder.encode([query], convert_to_numpy=True).astype("float32")
        D,I = index.search(q_emb, k=5)

        retrieved = [chunks[i] for i in I[0]]

        # -----------------------------------------------------------
        # 6. Load OPT-125M model (offline, CPU-friendly)
        -----------------------------------------------------------
        @st.cache_resource
        def load_opt():
            tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
            model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

            # Pad token fix
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            return tok, model, device

        tokenizer, gen_model, device = load_opt()

        # -----------------------------------------------------------
        # 7. Build strict UPSC generation prompt
        -----------------------------------------------------------
        context = "\n\n".join(retrieved)

        prompt = f"""
You are an expert UPSC question maker.  
Use ONLY the NCERT context below to create questions.

Context:
{context}

Generate questions in EXACT format:

1-Mark Questions
1.
2.

2-Mark Questions
1.
2.

5-Mark Questions
1.
2.

Rules:
- MUST end with '?'
- MUST start with What/Why/How/Explain/Describe/Discuss/Define/Evaluate
- 1-Mark: short definition/recall
- 2-Mark: short explanation
- 5-Mark: analytical, conceptual depth
"""


        # -----------------------------------------------------------
        # 8. Generate raw output
        -----------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            output_ids = gen_model.generate(
                inputs["input_ids"],
                max_new_tokens=300,
                num_beams=3,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)


        # -----------------------------------------------------------
        # 9. Extract questions cleanly
        -----------------------------------------------------------
        def extract_questions(text):
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            qs = []
            for l in lines:
                if re.match(r"^\d+\.\s+", l):
                    q = re.sub(r"^\d+\.\s*", "", l)
                    if not q.endswith("?"):
                        q += "?"
                    qs.append(q)
            return qs

        questions = extract_questions(raw)

        # If OPT output useless → fallback rule-based
        if len(questions) < 6:

            def rule_based(context):
                sentences = re.split(r'(?<=[.!?])\s+', context)
                out = []
                for s in sentences:
                    s = s.strip()
                    if not s or len(s.split()) < 6:
                        continue

                    if " is " in s or " are " in s:
                        subj = s.split(" is ")[0]
                        out.append(f"What is {subj}?")

                    elif " because " in s:
                        out.append(f"Why {s}?")

                    else:
                        out.append(f"Explain briefly: {s[:60]}?")

                    if len(out) >= 6:
                        break
                return out

            questions = rule_based(context)


        # -----------------------------------------------------------
        # 10. Display questions formatted
        -----------------------------------------------------------
        st.subheader("1-Mark Questions")
        for q in questions[:2]:
            st.write("• " + q)

        st.subheader("2-Mark Questions")
        for q in questions[2:4]:
            st.write("• " + q)

        st.subheader("5-Mark Questions")
        for q in questions[4:6]:
            st.write("• " + q)
