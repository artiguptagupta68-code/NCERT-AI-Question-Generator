# ============================================
# NCERT + UPSC Exam-Ready Generator (RAG-based)
# Streamlit-safe | Summarized Flashcards
# ============================================

import os, zipfile, re, random
from pathlib import Path
import streamlit as st
import gdown
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# CONFIG
# --------------------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
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

SIMILARITY_THRESHOLD_NCERT = 0.35
SIMILARITY_THRESHOLD_UPSC = 0.45

# --------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------
st.set_page_config(page_title="NCERT & UPSC Generator", layout="wide")
st.title("📘 NCERT & UPSC Exam-Ready Question Generator")

# --------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --------------------------------------------
# UTILITIES
# --------------------------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH)

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)

    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            out = zfile.parent / zfile.stem
            out.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(out)
        except:
            pass

def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(
        r"(activity|exercise|project|editor|isbn|copyright).*",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()

def pdf_matches_subject(path, subject):
    return any(k in path.lower() for k in SUBJECT_KEYWORDS[subject])

def load_all_texts(subject):
    texts = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        if pdf_matches_subject(str(pdf), subject):
            t = clean_text(read_pdf(str(pdf)))
            if len(t.split()) > 50:
                texts.append(t)
    return texts

def semantic_chunks(text):
    sents = re.split(r"(?<=[.?!])\s+", text)
    return [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]

def is_conceptual(s):
    skip = ["chapter", "unit", "page", "contents", "figure", "table"]
    return 8 <= len(s.split()) <= 60 and not any(k in s.lower() for k in skip)

# --------------------------------------------
# EMBEDDINGS
# --------------------------------------------
@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

# --------------------------------------------
# SAFE RETRIEVAL
# --------------------------------------------
def retrieve_relevant_chunks(chunks, embeddings, query, standard="NCERT", top_k=10):
    if not chunks or embeddings is None or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return []

    q_vec = embedder.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_vec, embeddings)[0]

    threshold = SIMILARITY_THRESHOLD_UPSC if standard == "UPSC" else SIMILARITY_THRESHOLD_NCERT
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)

    return [ch for ch, sc in ranked if sc >= threshold and is_conceptual(ch)][:top_k]

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

def estimate_flashcards(chunks, max_sentences_per_card=6):
    """
    Estimates the number of flashcards that can be generated.
    """
    all_text = " ".join(chunks)
    sentences = re.split(r'(?<=[.?!])\s+', all_text)
    conceptual_sentences = [s.strip() for s in sentences if is_conceptual_sentence(s)]
    
    if not conceptual_sentences:
        return 0
    num_cards = len(conceptual_sentences) // max_sentences_per_card
    if len(conceptual_sentences) % max_sentences_per_card != 0:
        num_cards += 1
    return num_cards

def generate_flashcard(chunks, topic, mode="NCERT"):
    """
    Generate a single high-quality summarized flashcard for a topic.
    Combines all relevant text, filters conceptual content, summarizes it,
    and formats it with overview, explanation, classification, conclusion, and points to remember.
    """
    if not chunks:
        return []

    # Combine all chunks into one text
    all_text = " ".join(chunks)

    # Remove excessive newlines, multiple spaces
    all_text = re.sub(r'\s+', ' ', all_text).strip()

    # Split into sentences (naive regex splitter)
    sentences = re.split(r'(?<=[.?!])\s+', all_text)

    # Filter: keep meaningful sentences
    def is_meaningful(s):
        skip_keywords = ["phone", "fax", "isbn", "copyright", "page", "address",
                         "reprint", "pd", "bs", "exercise", "activity"]
        return all(k not in s.lower() for k in skip_keywords) and len(s.split()) > 5

    meaningful_sentences = [s.strip() for s in sentences if is_meaningful(s)]

    if not meaningful_sentences:
        return []

    # Concept Overview: first meaningful sentence
    concept_overview = meaningful_sentences[0]

    # Explanation: combine next 10 sentences (or less if fewer exist)
    explanation = " ".join(meaningful_sentences[1:11]) if len(meaningful_sentences) > 1 else concept_overview

    # Classification / Types: generic placeholder
    classification = "These ideas relate to key concepts, constitutional principles, and practical applications."

    # Conclusion: generic placeholder
    conclusion = "Overall, this concept is essential for understanding the topic and its practical relevance."

    # Points to Remember: top 5 short sentences from meaningful_sentences
    points_to_remember = []
    for s in meaningful_sentences[1:15]:
        words = s.split()
        bullet = " ".join(words[:25]) + ("..." if len(words) > 25 else "")
        points_to_remember.append(bullet)
        if len(points_to_remember) >= 5:
            break

    # Construct final flashcard
    card = {
        "title": f"{topic.capitalize()} - Summarized Flashcard",
        "content": (
            f"Concept Overview: {concept_overview}\n\n"
            f"Explanation: {explanation}\n\n"
            f"Classification / Types: {classification}\n\n"
            f"Conclusion: {conclusion}\n\n"
            f"Points to Remember:\n- " + "\n- ".join(points_to_remember)
        )
    }

    return [card]






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
    texts = load_all_texts(subject)
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

# FLASHCARDS
with tab4:
    mode = st.radio("Depth", ["NCERT", "UPSC"], key="flash_std", horizontal=True)
    if topic.strip():
        # Retrieve relevant chunks
        rel = retrieve_relevant_chunks(chunks, embeddings, topic, mode, top_k=10)
        
        # Generate **one summarized flashcard**
        cards = generate_flashcard(rel, topic, mode)
        
        if cards:
            c = cards[0]
            st.markdown(f"### 📌 {c['title']}")
            st.write(c["content"])
        else:
            st.warning("No relevant content found to generate flashcard.")
