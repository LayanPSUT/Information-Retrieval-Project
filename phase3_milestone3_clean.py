import os
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


# =========================
# CONFIG
# =========================

APP_DIR = Path(__file__).resolve().parent
QUALITATIVE_LOG = APP_DIR / "phase3_qualitative_cases.csv"

DEFAULT_HF_MODEL = "google/flan-t5-small"

TOP_K = 10


# =========================
# FILE LOADING HELPERS
# =========================


def find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def ensure_dataset_folder():
    """Use dataset folder if present; otherwise extract a dataset zip if present."""
    dataset_dir = APP_DIR / "dataset"
    if dataset_dir.exists():
        return dataset_dir

    zip_path = find_first_existing([
        APP_DIR / "dataset.zip",
        APP_DIR / "dataset(1).zip",
        APP_DIR / "dataset (1).zip",
    ])

    if zip_path is None:
        raise FileNotFoundError(
            "Could not find dataset folder or dataset.zip/dataset(1).zip in the app folder."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Avoid extracting macOS metadata files.
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX/")]
        zf.extractall(APP_DIR, members=members)

    if not dataset_dir.exists():
        raise FileNotFoundError("Dataset zip was extracted, but no dataset folder was found.")
    return dataset_dir


@st.cache_data(show_spinner="Loading documents...")
def load_data():
    metadata_path = find_first_existing([
        APP_DIR / "metadata.xlsx",
        APP_DIR / "metadata(1).xlsx",
        APP_DIR / "metadata (1).xlsx",
    ])
    if metadata_path is None:
        raise FileNotFoundError("Could not find metadata.xlsx or metadata(1).xlsx.")

    dataset_dir = ensure_dataset_folder()
    df = pd.read_excel(metadata_path)

    required_columns = {"docid", "title"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"metadata file is missing required columns: {missing_columns}")

    documents, titles, links, doc_ids, topics = [], [], [], [], []

    for _, row in df.iterrows():
        raw_doc_id = str(row["docid"]).strip()
        file_name = raw_doc_id if raw_doc_id.endswith(".txt") else raw_doc_id + ".txt"
        file_path = dataset_dir / file_name

        try:
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Missing file: {file_name}")
            continue

        documents.append(text)
        titles.append(str(row["title"]))
        links.append(str(row["link"]) if "link" in df.columns else "#")
        doc_ids.append(raw_doc_id.replace(".txt", ""))
        topics.append(str(row["topic"]) if "topic" in df.columns else "Unknown")

    if not documents:
        raise ValueError("No documents were loaded. Check dataset file names and metadata docid values.")

    loaded = pd.DataFrame({
        "doc_id": doc_ids,
        "title": titles,
        "topic": topics,
        "link": links,
        "text": documents,
    })
    return loaded


# =========================
# QUERY PROCESSING
# =========================

ABBREVIATIONS = {
    "ai": "artificial intelligence",
    "iot": "internet of things",
    "ds": "data science",
    "ml": "machine learning",
    "nlp": "natural language processing",
}


def process_query(query):
    """Original Phase 2 preprocessing: spelling correction, lowercasing, abbreviations."""
    query = str(TextBlob(str(query)).correct())
    query = query.lower()
    words = query.split()
    words = [ABBREVIATIONS.get(w, w) for w in words]
    return " ".join(words)


# =========================
# LOCAL LLM THROUGH HUGGING FACE TRANSFORMERS
# =========================


@st.cache_resource(show_spinner="Loading local Hugging Face LLM...")
def load_hf_llm(model_name=DEFAULT_HF_MODEL):
    """
    Loads a small local Hugging Face sequence-to-sequence LLM.
    This avoids the Transformers pipeline task-name issue on some Macs.
    No OpenAI and no Ollama are used.
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing packages. Install them with: "
            "python3 -m pip install transformers torch sentencepiece"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch


def call_local_llm(prompt, model_name=DEFAULT_HF_MODEL):
    """Calls a local Hugging Face LLM. No OpenAI and no Ollama."""
    tokenizer, model, torch = load_hf_llm(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            num_beams=2,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def clean_llm_text(text):
    text = str(text).strip()
    text = re.sub(r"^[\s\-:]+", "", text)
    text = re.sub(r"[\n\r]+", " ", text)
    return text.strip(' "')


def llm_rewrite_query(query, model_name):
    """Strategy 1: Query Rewriting."""
    prompt = (
        "Rewrite this search query to be clearer and more specific. "
        "Do not answer it. Return only the rewritten query.\n"
        f"Query: {query}"
    )
    rewritten = clean_llm_text(call_local_llm(prompt, model_name=model_name))
    if not rewritten or len(rewritten.split()) > 25:
        return query
    return rewritten


def llm_expand_query(query, model_name, max_terms=8):
    """Strategy 2: Query Expansion."""
    prompt = (
        "Generate related search terms and synonyms for this query. "
        "Return only a comma-separated list, no explanation.\n"
        f"Query: {query}"
    )
    raw = clean_llm_text(call_local_llm(prompt, model_name=model_name))

    # Accept comma-separated, semicolon-separated, or line-separated terms.
    pieces = re.split(r"[,;\n]", raw)
    clean_terms = []
    seen = set()
    for term in pieces:
        term = re.sub(r"^[0-9]+[.)]\s*", "", term.strip().lower())
        term = re.sub(r"[^a-zA-Z0-9\s\-]", "", term).strip()
        if term and term not in seen and term != query.lower():
            clean_terms.append(term)
            seen.add(term)
        if len(clean_terms) >= max_terms:
            break

    return clean_terms


def build_llm_augmented_query(original_query, model_name):
    """
    Applies the two required LLM strategies:
    1. Query Rewriting
    2. Query Expansion
    """
    rewritten_query = llm_rewrite_query(original_query, model_name)
    expanded_terms = llm_expand_query(rewritten_query, model_name)
    augmented_query = " ".join([rewritten_query] + expanded_terms)

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "expanded_terms": expanded_terms,
        "augmented_query": augmented_query,
    }


# =========================
# RETRIEVAL MODELS
# =========================


@st.cache_resource(show_spinner="Building retrieval models...")
def build_retrieval_indexes(documents):
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedding_model.encode(documents, normalize_embeddings=True, show_progress_bar=False)

    return vectorizer, doc_vectors, bm25, embedding_model, doc_embeddings


def vsm_search(query, vectorizer, doc_vectors):
    query_vec = vectorizer.transform([query])
    return cosine_similarity(query_vec, doc_vectors)[0]


def bm25_search(query, bm25):
    return bm25.get_scores(query.lower().split())


def embedding_search(query, embedding_model, doc_embeddings):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.dot(doc_embeddings, query_embedding)


def run_search(model_choice, query, vectorizer, doc_vectors, bm25, embedding_model, doc_embeddings):
    if model_choice == "VSM":
        return vsm_search(query, vectorizer, doc_vectors)
    if model_choice == "BM25":
        return bm25_search(query, bm25)
    if model_choice == "Embedding":
        return embedding_search(query, embedding_model, doc_embeddings)
    raise ValueError(f"Unknown model choice: {model_choice}")


def get_top_results(scores, top_k=TOP_K):
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def results_to_dataframe(results, data):
    rows = []
    for rank, (idx, score) in enumerate(results, start=1):
        row = data.iloc[idx]
        rows.append({
            "rank": rank,
            "doc_id": row["doc_id"],
            "title": row["title"],
            "topic": row["topic"],
            "score": round(float(score), 4),
            "link": row["link"],
            "snippet": row["text"][:250].replace("\n", " "),
        })
    return pd.DataFrame(rows)


def precision_at_10(results_df, expected_topic):
    if not expected_topic or expected_topic == "No expected topic selected":
        return None
    return float((results_df["topic"].astype(str) == str(expected_topic)).mean())


# =========================
# QUALITATIVE ANALYSIS LOGGING
# =========================


def append_qualitative_case(case_data):
    case_df = pd.DataFrame([case_data])
    if QUALITATIVE_LOG.exists():
        existing = pd.read_csv(QUALITATIVE_LOG)
        combined = pd.concat([existing, case_df], ignore_index=True)
    else:
        combined = case_df
    combined.to_csv(QUALITATIVE_LOG, index=False)


def load_qualitative_log():
    if QUALITATIVE_LOG.exists():
        return pd.read_csv(QUALITATIVE_LOG)
    return pd.DataFrame(columns=[
        "timestamp", "query", "model", "case_type", "baseline_top_docs",
        "llm_top_docs", "rewritten_query", "expanded_terms", "notes"
    ])


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Milestone 3 Retrieval System", layout="wide")
st.title("Milestone 3: LLM-Augmented Retrieval System")
st.caption("LLM-Augmented Retrieval using Query Rewriting and Query Expansion.")

try:
    data = load_data()
    documents = data["text"].tolist()
    vectorizer, doc_vectors, bm25, embedding_model, doc_embeddings = build_retrieval_indexes(documents)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("LLM Settings")
    hf_model = st.text_input("Hugging Face LLM", DEFAULT_HF_MODEL)
    st.info("Uses a local Hugging Face Transformers LLM. First run may download the model.")

    st.header("Retrieval Settings")
    model_choice = st.selectbox("Retrieval model", ["VSM", "BM25", "Embedding"], index=1)

    topics = sorted(data["topic"].dropna().astype(str).unique().tolist())
    expected_topic = st.selectbox(
        "Expected topic for quick Precision@10 comparison",
        ["No expected topic selected"] + topics,
    )

search_tab, compare_tab, cases_tab = st.tabs([
    "Search with LLM",
    "Baseline vs LLM Comparison",
    "Qualitative Helped/Hurt Cases",
])

with search_tab:
    st.subheader("LLM-Augmented Search")
    query = st.text_input("Enter your query", key="single_query")

    if st.button("Search using Query Rewriting + Query Expansion", key="single_search"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                original_processed = process_query(query)
                llm_info = build_llm_augmented_query(original_processed, hf_model)
                final_query = process_query(llm_info["augmented_query"])

                scores = run_search(
                    model_choice, final_query,
                    vectorizer, doc_vectors, bm25, embedding_model, doc_embeddings,
                )
                results = get_top_results(scores)
                results_df = results_to_dataframe(results, data)

                st.write("### LLM Query Augmentation")
                st.write(f"**Original processed query:** {original_processed}")
                st.write(f"**Rewritten query:** {llm_info['rewritten_query']}")
                st.write(f"**Expanded terms:** {', '.join(llm_info['expanded_terms'])}")
                st.write(f"**Final retrieval query:** {final_query}")

                p10 = precision_at_10(results_df, expected_topic)
                if p10 is not None:
                    st.metric("Precision@10 using selected expected topic", round(p10, 3))

                st.write("### Top-10 Results")
                for _, row in results_df.iterrows():
                    st.markdown(f"#### {int(row['rank'])}. {row['title']}")
                    st.write(f"DocID: {row['doc_id']} | Topic: {row['topic']} | Score: {row['score']}")
                    st.markdown(f"[Open Document]({row['link']})")
                    st.write(row["snippet"])
                    st.write("---")

            except Exception as e:
                st.error(str(e))

with compare_tab:
    st.subheader("Compare Classical Baseline vs LLM-Augmented Retrieval")
    compare_query = st.text_input("Enter query to compare", key="compare_query")

    if st.button("Compare Top-10 Results", key="compare_search"):
        if not compare_query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                baseline_query = process_query(compare_query)
                baseline_scores = run_search(
                    model_choice, baseline_query,
                    vectorizer, doc_vectors, bm25, embedding_model, doc_embeddings,
                )
                baseline_df = results_to_dataframe(get_top_results(baseline_scores), data)

                llm_info = build_llm_augmented_query(baseline_query, hf_model)
                llm_query = process_query(llm_info["augmented_query"])
                llm_scores = run_search(
                    model_choice, llm_query,
                    vectorizer, doc_vectors, bm25, embedding_model, doc_embeddings,
                )
                llm_df = results_to_dataframe(get_top_results(llm_scores), data)

                st.session_state["last_compare"] = {
                    "query": compare_query,
                    "model": model_choice,
                    "baseline_query": baseline_query,
                    "llm_info": llm_info,
                    "baseline_df": baseline_df,
                    "llm_df": llm_df,
                }

                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Classical Baseline")
                    st.write(f"**Query:** {baseline_query}")
                    p10_base = precision_at_10(baseline_df, expected_topic)
                    if p10_base is not None:
                        st.metric("Baseline Precision@10", round(p10_base, 3))
                    st.dataframe(baseline_df[["rank", "doc_id", "title", "topic", "score"]], use_container_width=True)

                with col2:
                    st.write("### LLM-Augmented")
                    st.write(f"**Rewritten:** {llm_info['rewritten_query']}")
                    st.write(f"**Expanded terms:** {', '.join(llm_info['expanded_terms'])}")
                    p10_llm = precision_at_10(llm_df, expected_topic)
                    if p10_llm is not None:
                        st.metric("LLM Precision@10", round(p10_llm, 3))
                    st.dataframe(llm_df[["rank", "doc_id", "title", "topic", "score"]], use_container_width=True)

                if p10_base is not None and p10_llm is not None:
                    diff = p10_llm - p10_base
                    if diff > 0:
                        st.success(f"LLM helped for this query: Precision@10 improved by {round(diff, 3)}.")
                    elif diff < 0:
                        st.error(f"LLM hurt for this query: Precision@10 decreased by {round(abs(diff), 3)}.")
                    else:
                        st.info("No Precision@10 change for this query.")

            except Exception as e:
                st.error(str(e))

    if "last_compare" in st.session_state:
        st.write("### Save this as a qualitative report case")
        case_type = st.selectbox("Case type", ["LLM helped", "LLM hurt", "No clear change"])
        notes = st.text_area(
            "Notes for report",
            placeholder="Explain why the LLM helped or hurt. Example: expansion added useful synonyms, or rewrite changed the intent.",
        )
        if st.button("Save qualitative case"):
            last = st.session_state["last_compare"]
            append_qualitative_case({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": last["query"],
                "model": last["model"],
                "case_type": case_type,
                "baseline_top_docs": "; ".join(last["baseline_df"]["doc_id"].head(10).astype(str).tolist()),
                "llm_top_docs": "; ".join(last["llm_df"]["doc_id"].head(10).astype(str).tolist()),
                "rewritten_query": last["llm_info"]["rewritten_query"],
                "expanded_terms": "; ".join(last["llm_info"]["expanded_terms"]),
                "notes": notes,
            })
            st.success(f"Saved to {QUALITATIVE_LOG.name}")

with cases_tab:
    st.subheader("Saved Qualitative Cases")
    cases_df = load_qualitative_log()
    st.dataframe(cases_df, use_container_width=True)

    helped_count = int((cases_df["case_type"] == "LLM helped").sum()) if not cases_df.empty else 0
    hurt_count = int((cases_df["case_type"] == "LLM hurt").sum()) if not cases_df.empty else 0

    col1, col2 = st.columns(2)
    col1.metric("Saved helped cases", helped_count)
    col2.metric("Saved hurt cases", hurt_count)

    if helped_count < 2 or hurt_count < 2:
        st.warning(
            "For the PDF report, save at least 2 cases where the LLM helped "
            "and at least 2 cases where it hurt performance."
        )
    else:
        st.success("You have enough helped/hurt cases for the qualitative analysis requirement.")

    st.download_button(
        "Download qualitative cases CSV",
        data=cases_df.to_csv(index=False),
        file_name="phase3_qualitative_cases.csv",
        mime="text/csv",
    )
