import os
from pathlib import Path
import zipfile
import re

import streamlit as st
import gdown
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

SUBJECTS = ["Polity", "Economics", "Sociology", "Psychology", "Business Studies"]

# -----------------------------
# STREAMLIT SETUP
# -----------------------------
st.set_page_config(page_title="NCERT Learning Assistant", layout="wide")
st.title("📘 NCERT Learning Assistant")
st.write("Select topics to get chapter, book, and video recommendations for Class 11–12 students")

# -----------------------------
# DOWNLOAD & EXTRACT NCERT PDFs
# -----------------------------
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    # extract nested zips
    for zfile in Path(EXTRACT_DIR).rglob("*.zip"):
        try:
            target = zfile.parent / zfile.stem
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(zfile, "r") as inner:
                inner.extractall(target)
        except:
            pass

download_and_extract()

# -----------------------------
# PDF Utilities
# -----------------------------
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def clean_text(text):
    text = re.sub(r"(activity|let us|exercise|project|editor|reprint|copyright|isbn).*", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()

def load_all_texts():
    texts = []
    paths = []
    for pdf in Path(EXTRACT_DIR).rglob("*.pdf"):
        t = clean_text(read_pdf(str(pdf)))
        if len(t.split()) > 50:
            texts.append(t)
            paths.append(pdf)
    return texts, paths

# -----------------------------
# EXTRACT TOPICS
# -----------------------------
def extract_topics_from_text(text):
    lines = text.split("\n")
    topics = []
    for line in lines:
        line = line.strip()
        if 3 <= len(line.split()) <= 8 and line[0].isupper():
            line = re.sub(r"^(chapter\s*\d+:?|[\d.]+\s*)", "", line, flags=re.I)
            topics.append(line.strip())
    return list(set(topics))

@st.cache_data
def get_all_topics():
    texts, _ = load_all_texts()
    all_topics = []
    for text in texts:
        all_topics.extend(extract_topics_from_text(text))
    return sorted(list(set(all_topics)))

all_topics = get_all_topics()

# -----------------------------
# Sentence Transformer
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# -----------------------------
# Books & Videos
# -----------------------------
books = {
    "Sociology": {
        "Beginner": ["NCERT Sociology Class 11", "Understanding Society by Haralambos (Intro)"],
        "Intermediate": ["Introduction to Sociology by Bottomore", "Sociology: Themes and Perspectives by Haralambos"],
        "Advanced": ["Advanced Sociology by Giddens", "Sociology: Concepts and Theories by Macionis"]
    },
    "Polity": {
        "Beginner": ["NCERT Political Science Class 11", "Indian Constitution Made Easy"],
        "Intermediate": ["Indian Polity by M. Laxmikanth", "Governance in India by D.D. Basu"],
        "Advanced": ["Introduction to the Constitution of India by D.D. Basu", "Indian Constitutional Law Advanced"]
    },
    "Psychology": {
        "Beginner": ["NCERT Psychology Class 11", "Psychology Basics Intro"],
        "Intermediate": ["Psychology: An Exploration by Saundra K. Ciccarelli", "Understanding Psychology by Feldman"],
        "Advanced": ["Advanced Psychology by Baron & Misra", "Handbook of Psychology by Weiner"]
    },
    "Economics": {
        "Beginner": ["NCERT Economics Class 11", "Principles of Economics Intro by Mankiw"],
        "Intermediate": ["Principles of Economics by N. Gregory Mankiw", "Economic Theory by Samuelson"],
        "Advanced": ["Advanced Economic Theory by H.L. Ahuja", "Micro and Macro Economics Advanced"]
    },
    "Business Studies": {
        "Beginner": ["NCERT Business Studies Class 11", "Introduction to Management Basics"],
        "Intermediate": ["Business Studies by Poonam Gandhi", "Business Management Concepts"],
        "Advanced": ["Advanced Business Management by Robbins", "Strategic Management Advanced"]
    }
}

videos = {
    subject: {
        level: [f"https://www.youtube.com/watch?v={subject}_{level}_{i+1}" for i in range(3)]
        for level in ["Beginner", "Intermediate", "Advanced"]
    } for subject in SUBJECTS
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.sidebar.header("🎓 Select Options")
selected_subject = st.sidebar.selectbox("Select Subject", SUBJECTS)
selected_level = st.sidebar.selectbox("Select Level", ["Beginner", "Intermediate", "Advanced"])
selected_topics = st.sidebar.multiselect("Select Topic(s)", all_topics)

# -----------------------------
# Recommend chapters using embeddings
# -----------------------------
@st.cache_data
def get_chapter_recommendations(topics, top_n=5):
    if not topics:
        return pd.DataFrame({"Message":["Please select topics"]})
    
    texts, paths = load_all_texts()
    text_embeddings = embedder.encode(texts, convert_to_tensor=True)
    topic_embedding = embedder.encode([" ".join(topics)], convert_to_tensor=True)
    
    sims = cosine_similarity(topic_embedding.cpu().numpy(), text_embeddings.cpu().numpy()).flatten()
    
    top_idx = sims.argsort()[-top_n:][::-1]
    results = []
    for i in top_idx:
        results.append({
            "Chapter PDF": str(paths[i].name),
            "Similarity Score": sims[i]
        })
    return pd.DataFrame(results)

# -----------------------------
# Recommend books & videos
# -----------------------------
def recommend_materials(subject, level, topics):
    if not topics:
        return pd.DataFrame({"Message":["Please select at least one topic"]})
    recommended_books = books.get(subject, {}).get(level, [])
    recommended_videos = videos.get(subject, {}).get(level, [])
    return pd.DataFrame({
        "Topics": [", ".join(topics)],
        "Recommended Books": [", ".join(recommended_books)],
        "Recommended Videos": [", ".join(recommended_videos)]
    })

# -----------------------------
# Display recommendations
# -----------------------------
st.subheader("📚 Chapter Recommendations")
if st.button("Get Chapter Recommendations"):
    chapter_recs = get_chapter_recommendations(selected_topics)
    st.dataframe(chapter_recs)

st.subheader("📖 Book & Video Recommendations")
if st.button("Get Books & Videos"):
    materials = recommend_materials(selected_subject, selected_level, selected_topics)
    st.dataframe(materials)
