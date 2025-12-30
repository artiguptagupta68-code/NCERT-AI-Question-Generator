
import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------




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
EPOCHS = 15  # Fine-tuning epochs
BATCH_SIZE = 16

# =========================================
# STREAMLIT CONFIG
# =========================================
st.set_page_config(page_title="NCERT AI Generator", layout="wide")
st.title("📘 NCERT + UPSC AI Question Generator with Training")

# =========================================
# DOWNLOAD & EXTRACT NCERT ZIP
# =========================================
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading NCERT ZIP...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    st.success("✅ NCERT content extracted")

# =========================================
# READ PDF & CLEAN TEXT
# =========================================
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

def load_all_texts(extract_dir=EXTRACT_DIR):
    texts = []
    for pdf in Path(extract_dir).rglob("*.pdf"):
        t = clean_text(read_pdf(pdf))
        if len(t.split()) > 100:
            texts.append(t)
    return texts


# =========================================
# SEMANTIC CHUNKING
# =========================================
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

# =========================================
# LOAD EMBEDDER
# =========================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# =========================================
# PREPARE TRAINING DATA
# =========================================
def prepare_training_data(chunks):
    examples = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.?!])\s+", chunk)
        sentences = [s for s in sentences if len(s.split()) > 6]
        for i in range(len(sentences)-1):
            examples.append(InputExample(texts=[sentences[i], sentences[i+1]]))
    return examples

# =========================================
# TRAIN MODEL
# =========================================
def train_embedding_model(chunks, embedder, epochs=EPOCHS):
    st.info(f"🔹 Fine-tuning embeddings for {epochs} epochs...")
    train_examples = prepare_training_data(chunks)
    if not train_examples:
        st.warning("Not enough data to fine-tune embeddings.")
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

# =========================================
# EMBEDDINGS
# =========================================
@st.cache_data
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

# =========================================
# RETRIEVAL
# =========================================
def retrieve_relevant_chunks(chunks, embeddings, query, mode="NCERT", top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    threshold = SIMILARITY_THRESHOLD_UPSC if mode=="UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked if s >= threshold][:top_k]

# --------------------------------------------
# QUESTION GENERATORS
# --------------------------------------------
def generate_subjective(topic, n, standard):
    if standard == "NCERT":
        qs = [
            f"Explain the concept of {topic}.",
            f"Describe the main features of {topic}.",
            f"Write a short note on {topic}.",
            f"Explain {topic} with examples."
        ]
    else:
        qs = [
            f"Analyse the significance of {topic}.",
            f"Critically examine {topic}.",
            f"Discuss the relevance of {topic} in contemporary India."
        ]
    return qs[:n]

def generate_ncert_mcqs(chunks, topic, n):
    sents = [s for ch in chunks for s in re.split(r"[.;]", ch) if is_conceptual(s)]
    random.shuffle(sents)
    mcqs = []
    for s in sents[:n]:
        opts = random.sample(sents, min(4, len(sents)))
        if s not in opts:
            opts[0] = s
        random.shuffle(opts)
        mcqs.append({
            "q": f"Which statement best explains {topic}?",
            "options": opts,
            "answer": opts.index(s)
        })
    return mcqs

def normalize_text(s):
    s = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().capitalize()

def is_conceptual_sentence(s):
    s = s.strip()
    if len(s.split()) < 8:
        return False
    # Skip OCR artifacts or admin info
    skip = ["phone", "fax", "isbn", "copyright", "page", "address", 
            "reprint", "pd", "bs", "ncertain", "office", "publication division"]
    if any(k in s.lower() for k in skip):
        return False
    # Keep only sentences that contain key conceptual words
    keywords = ["right", "law", "constitution", "governance", "democracy",
                "citizen", "freedom", "justice", "equality", "policy"]
    if not any(k in s.lower() for k in keywords):
        return False
    return True

def generate_flashcard(chunks, topic):
    """
    Generates ONE high-quality summarized flashcard from all relevant chunks.
    """

    # ---------- STEP 1: CLEAN & FILTER USEFUL SENTENCES ----------
    def is_valid_sentence(s):
        s = s.lower()
        garbage_words = [
            "isbn", "price", "publication", "reprint", "copyright",
            "printed", "office", "phone", "address", "editor",
            "publisher", "page", "ncertain", "pd", "bs"
        ]
        return (
            len(s.split()) > 8 and
            not any(g in s for g in garbage_words)
        )

    sentences = []
    for ch in chunks:
        parts = re.split(r'(?<=[.?!])\s+', ch)
        for p in parts:
            if is_valid_sentence(p):
                sentences.append(p.strip())

    if not sentences:
        return []

    # ---------- STEP 2: BUILD LOGICAL SECTIONS ----------

    concept_overview = sentences[0]

    explanation = " ".join(sentences[1:6])

    classification = (
        "The concept relates to constitutional values such as democracy, "
        "rights and duties, rule of law, and the relationship between the "
        "state and citizens."
    )

    conclusion = (
        "Overall, the Constitution provides a framework for governance, "
        "protects individual freedoms, and ensures justice, equality, "
        "and accountability in a democratic society."
    )

    points = []
    for s in sentences[1:10]:
        short = " ".join(s.split()[:20])
        points.append(short)
        if len(points) == 5:
            break

    flashcard = {
        "title": topic.capitalize(),
        "content": f"""
Concept Overview:
{concept_overview}

Explanation:
{explanation}

Classification / Types:
{classification}

Conclusion:
{conclusion}

Points to Remember:
- {"\n- ".join(points)}
"""
    }

    return [flashcard]
# --------------------------------------------
# SIDEBAR
# --------------------------------------------
with st.sidebar:
    if st.button("📥 Load NCERT PDFs"):
        download_and_extract()
        st.success("NCERT PDFs loaded")

subject = st.selectbox("Subject", SUBJECTS)
topic = st.text_input("Topic")
num_q = st.number_input("Number of Questions", 1, 10, 5)

# --------------------------------------------
# LOAD DATA
# --------------------------------------------
texts, chunks = [], []
if os.path.exists(EXTRACT_DIR):
    texts = load_all_texts()
    for t in texts:
        chunks.extend(semantic_chunks(t))

embeddings = embed_chunks(chunks) if chunks else np.empty((0, 384))

st.write(f"📄 PDFs used: {len(texts)}")
st.write(f"🧩 Chunks created: {len(chunks)}")

# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Subjective", "🧠 MCQs", "💬 Ask NCERT", "🧠 Flashcards"])

# SUBJECTIVE
with tab1:
    std1 = st.radio("Standard", ["NCERT", "UPSC"], key="sub_std", horizontal=True)
    if st.button("Generate Subjective"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, std1)
        for i, q in enumerate(generate_subjective(topic, min(num_q, len(rel)), std1), 1):
            st.write(f"{i}. {q}")

# MCQs
with tab2:
    std2 = st.radio("Standard", ["NCERT", "UPSC"], key="mcq_std", horizontal=True)
    if st.button("Generate MCQs"):
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, std2)
        for i, m in enumerate(generate_ncert_mcqs(rel, topic, num_q), 1):
            st.write(f"**Q{i}. {m['q']}**")
            for j, o in enumerate(m["options"]):
                st.write(f"{chr(97+j)}) {o}")
            st.write(f"✅ Answer: {chr(97+m['answer'])}")
            st.write("---")

# CHATBOT
with tab3:
    st.subheader("Ask anything strictly from NCERT")
    chatbot_mode = st.radio("Answer Style", ["NCERT", "UPSC"], horizontal=True)
    user_q = st.text_input("Enter your question")

    if st.button("Ask NCERT"):
        if not user_q.strip():
            st.error("Please enter a question.")
        else:
            retrieved = retrieve_relevant_chunks(chunks, embeddings, user_q, standard=chatbot_mode, top_k=6)
            if not retrieved:
                st.error("❌ Answer not found in NCERT textbooks.")
            else:
                st.markdown("### 📘 NCERT-based answer:")
                # Take first few sentences of retrieved chunks as answer
                answer_sentences = []
                for r in retrieved:
                    sents = re.split(r"(?<=[.?!])\s+", r)
                    for s in sents:
                        if is_conceptual(s):
                            answer_sentences.append(normalize_text(s))
                st.write(" ".join(answer_sentences[:6]))

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
            st.warning("No relevant content found to generate flashcard.")
