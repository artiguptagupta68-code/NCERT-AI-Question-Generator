import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# CONFIG
# -------------------------------
FILE_ID = "1l5tGbJcvwsPiiOyi-MZYoWzoTDGjTWOz"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"
SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

SUBJECT_KEYWORDS = {
    "Polity": ["polity", "political", "constitution", "civics"],
    "Economics": ["economics", "economic"],
    "Sociology": ["sociology", "society"],
    "Psychology": ["psychology"],
    "Business Studies": ["business", "management"]
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 6
SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45
EPOCHS = 15
BATCH_SIZE = 16

# -------------------------------
# STREAMLIT SETUP
# -------------------------------
st.set_page_config(page_title="NCERT AI Generator", layout="wide")
st.title("📘 NCERT + UPSC AI Question Generator")

# -------------------------------
# DOWNLOAD & EXTRACT PDFs
# -------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    st.success("✅ NCERT PDFs extracted!")

# -------------------------------
# READ & CLEAN PDF
# -------------------------------
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(exercise|summary|table|figure|copyright).*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_all_texts(subject=None, extract_dir=EXTRACT_DIR):
    texts = []
    for pdf in Path(extract_dir).rglob("*.pdf"):
        if subject:
            if not any(k in pdf.stem.lower() for k in SUBJECT_KEYWORDS.get(subject, [])):
                continue
        t = clean_text(read_pdf(pdf))
        if len(t.split()) > 100:
            texts.append(t)
    return texts

# -------------------------------
# SEMANTIC CHUNKING
# -------------------------------
def semantic_chunking(text, embedder, max_words=180, sim_threshold=0.65):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 6]
    if len(sentences) < 2:
        return sentences
    embeddings = embedder.encode(sentences, convert_to_numpy=True)
    chunks = []
    current = [sentences[0]]
    current_emb = embeddings[0]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_emb], [embeddings[i]])[0][0]
        length = sum(len(s.split()) for s in current)
        if sim < sim_threshold or length > max_words:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_emb = embeddings[i]
        else:
            current.append(sentences[i])
            current_emb = np.mean([current_emb, embeddings[i]], axis=0)
    if current:
        chunks.append(" ".join(current))
    return chunks

# -------------------------------
# LOAD EMBEDDER
# -------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedder = load_embedder()

# -------------------------------
# TRAINING
# -------------------------------
def prepare_training_data(chunks):
    examples = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.?!])\s+", chunk)
        sentences = [s for s in sentences if len(s.split()) > 6]
        for i in range(len(sentences)-1):
            examples.append(InputExample(texts=[sentences[i], sentences[i+1]]))
    return examples

def train_embedding_model(chunks, embedder, epochs=EPOCHS):
    train_examples = prepare_training_data(chunks)
    if not train_examples:
        st.warning("❌ Not enough data to fine-tune embeddings.")
        return embedder
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(embedder)
    embedder.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=50,
        show_progress_bar=True
    )
    st.success("✅ Embedding model fine-tuned!")
    embedder.save("trained_ncert_model")
    return embedder

# -------------------------------
# EMBEDDINGS & RETRIEVAL
# -------------------------------
@st.cache_data
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    if len(chunks) == 0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode=="UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:top_k]

# -------------------------------
# CONCEPTUAL FILTER
# -------------------------------
def is_conceptual(s):
    s = s.strip()
    if len(s.split()) < 8:
        return False
    skip = ["phone", "fax", "isbn", "copyright", "page", "address", 
            "reprint", "pd", "bs", "ncertain", "office", "publication division"]
    if any(k in s.lower() for k in skip):
        return False
    keywords = ["right", "law", "constitution", "governance", "democracy",
                "citizen", "freedom", "justice", "equality", "policy"]
    if not any(k in s.lower() for k in keywords):
        return False
    return True

def normalize_text(s):
    s = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().capitalize()

# -------------------------------
# FLASHCARD GENERATOR
# -------------------------------
def generate_flashcard(chunks, topic):
    sentences = []
    for ch in chunks:
        for s in re.split(r'(?<=[.?!])\s+', ch):
            if len(s.split()) > 8:
                sentences.append(s.strip())
    if not sentences:
        return []
    overview = sentences[0]
    explanation = " ".join(sentences[1:6])
    conclusion = "Overall, this concept is important in NCERT context."
    points = [" ".join(s.split()[:20]) for s in sentences[1:6]]
    return [{"title": topic.capitalize(),
             "content": f"Concept Overview:\n{overview}\n\nExplanation:\n{explanation}\n\nConclusion:\n{conclusion}\n\nPoints:\n- {'\n- '.join(points)}"}]

# -------------------------------
# MCQ GENERATOR
# -------------------------------
def generate_mcqs(chunks, topic, n, standard="NCERT"):
    sents = [s for ch in chunks for s in re.split(r"[.;]", ch) if is_conceptual(s)]
    if not sents:
        return []
    random.shuffle(sents)
    mcqs = []
    for s in sents[:n]:
        opts = random.sample(sents, min(4, len(sents)))
        if s not in opts:
            opts[0] = s
        random.shuffle(opts)
        question_text = f"Explain {topic}." if standard=="NCERT" else f"Analyse the importance of {topic}."
        mcqs.append({"q": question_text, "options": opts, "answer": opts.index(s)})
    return mcqs

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
    if st.button("🔹 Fine-tune Embeddings"):
        if not chunks:
            st.warning("Load PDFs first to create chunks.")
        else:
            train_embedding_model(chunks, embedder, epochs=EPOCHS)

# -------------------------------
# AUTO-LOAD PDFs & CHUNKS
# -------------------------------
texts, chunks = [], []
if not os.path.exists(EXTRACT_DIR) or not list(Path(EXTRACT_DIR).rglob("*.pdf")):
    download_and_extract()

texts = load_all_texts(subject)
for t in texts:
    chunks.extend(semantic_chunking(t, embedder))

if not chunks:
    st.warning("No PDF content loaded. Use sidebar to load NCERT PDFs.")
else:
    embeddings = embed_chunks(chunks)
    st.write(f"📄 PDFs used: {len(texts)}")
    st.write(f"🧩 Chunks created: {len(chunks)}")

# -------------------------------
# MAIN TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Subjective", "🧠 MCQs", "💬 Chatbot", "🧠 Flashcards"])
topic = st.text_input("Topic")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# SUBJECTIVE
with tab1:
    std1 = st.radio("Standard", ["NCERT", "UPSC"], key="sub_std", horizontal=True)
    if st.button("Generate Subjective", key="sub_btn"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, std1)
        if not rel:
            st.warning("No relevant content found.")
        else:
            for i, q in enumerate([f"Explain {topic}."]*min(num_q, len(rel)), 1):
                st.write(f"{i}. {q}")

# MCQs
with tab2:
    std2 = st.radio("Standard", ["NCERT", "UPSC"], key="mcq_std", horizontal=True)
    if st.button("Generate MCQs", key="mcq_btn"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, std2)
        mcqs = generate_mcqs(rel, topic, num_q, std2)
        if not mcqs:
            st.warning("No MCQs could be generated.")
        else:
            for i, m in enumerate(mcqs, 1):
                st.markdown(f"**Q{i}. {m['q']}**")
                for j, o in enumerate(m["options"]):
                    st.write(f"{chr(97+j)}) {o}")
                st.write(f"✅ Answer: {chr(97+m['answer'])}")
                st.write("---")

# CHATBOT
with tab3:
    chatbot_mode = st.radio("Answer Style", ["NCERT", "UPSC"], horizontal=True)
    user_q = st.text_input("Enter your question", key="chat_q")
    if st.button("Ask NCERT", key="chat_btn"):
        if not user_q.strip():
            st.error("Please enter a question.")
        else:
            retrieved = retrieve_relevant_chunks(chunks, embeddings, user_q, mode=chatbot_mode)
            if not retrieved:
                st.warning("No relevant content found in NCERT PDFs.")
            else:
                answer_sentences = []
                for r in retrieved:
                    for s in re.split(r"(?<=[.?!])\s+", r):
                        if is_conceptual(s):
                            answer_sentences.append(normalize_text(s))
                st.write(" ".join(answer_sentences[:6]))

# FLASHCARDS
with tab4:
    mode = st.radio("Depth", ["NCERT", "UPSC"], key="flash_std", horizontal=True)
    if topic.strip():
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, mode, top_k=10)
        cards = generate_flashcard(rel, topic)
        if cards:
            c = cards[0]
            st.markdown(f"### 📌 {c['title']}")
            st.write(c["content"])
        else:
            st.warning("No relevant content found for flashcards.")
