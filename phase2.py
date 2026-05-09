import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# Data Loading

folder_path = "dataset"
df = pd.read_excel("metadata.xlsx")

documents = []
titles = []
links = []
doc_ids = []

for _, row in df.iterrows():
    doc_id = str(row["docid"])
    file_name = str(row["docid"]) + ".txt"
    file_path = os.path.join(folder_path, file_name)

    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            text = f.read()
            documents.append(text)
            titles.append(row["title"])
            links.append(row["link"] if "link" in df.columns else "#")
            doc_ids.append(doc_id)
    except FileNotFoundError:
        print(f"Missing file: {file_name}")

# VSM Model

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

def vsm_search(query):
    query_vec = vectorizer.transform([query])
    return cosine_similarity(query_vec, doc_vectors)[0]

# BM25 Model

tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query):
    return bm25.get_scores(query.split())

# Embedding Model

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)

def embedding_search(query):
    query_embedding = model.encode([query])[0]
    return np.dot(doc_embeddings, query_embedding)

# Query Processing

abbreviations = {
    "ai": "artificial intelligence",
    "iot": "internet of things",
    "ds": "data science",
    "ml": "machine learning",
    "nlp": "natural language processing"
}

def process_query(query):
    query = str(TextBlob(query).correct())  # spelling correction
    query = query.lower()

    words = query.split()
    words = [abbreviations.get(w, w) for w in words]

    return " ".join(words)

# Ranking

def get_top_results(scores):
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:10]

# Streamlit UI

st.title("IR Search System")

query = st.text_input("Enter your query")

model_choice = st.selectbox(
    "Select Model",
    ["VSM", "BM25", "Embedding"]
)

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a query")
    else:
        query = process_query(query)

        if model_choice == "VSM":
            scores = vsm_search(query)
        elif model_choice == "BM25":
            scores = bm25_search(query)
        else:
            scores = embedding_search(query)

        results = get_top_results(scores)

        st.write("### Results:")

        for idx, score in results:
            st.markdown(f"### {titles[idx]}")
            st.write(f"DocID: doc{doc_ids[idx]}")
            st.markdown(f"[Open Document]({links[idx]})")
            st.write(documents[idx][:200])
            st.write(f"Score: {round(float(score), 4)}")
            st.write("---")