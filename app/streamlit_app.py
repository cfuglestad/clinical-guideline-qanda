"""Streamlit demo interface for Clinical Guideline Q&A."""

import os
from pathlib import Path

import streamlit as st

from src.generation.agent import QAAgent
from src.ingestion.chunker import chunk_pages
from src.ingestion.loader import load_directory
from src.retrieval.store import GuidelineStore

# Inject Streamlit secrets into environment for langchain-openai
if "OPENAI_API_KEY" not in os.environ:
    try:
        key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = key
    except (KeyError, FileNotFoundError):
        st.error(
            "Missing `OPENAI_API_KEY`. Add it in Streamlit Cloud: "
            "App settings → Secrets → `OPENAI_API_KEY = \"sk-...\"`"
        )
        st.stop()

PERSIST_DIR = "chroma_db"
DATA_DIRS = ["data/sample", "data/raw"]

st.set_page_config(
    page_title="Clinical Guideline Q&A",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Clinical Guideline Q&A")
st.markdown(
    "Ask a clinical question and get an answer grounded in published practice "
    "guidelines, with transparent citations and retrieval confidence."
)


@st.cache_resource
def load_agent() -> QAAgent:
    """Initialize the store, auto-ingest if empty, and return the agent."""
    store = GuidelineStore(persist_dir=PERSIST_DIR)

    if store.count == 0:
        pages = []
        for data_dir in DATA_DIRS:
            d = Path(data_dir)
            if d.exists():
                pages.extend(load_directory(d))

        if pages:
            chunks = chunk_pages(pages)
            store.add_chunks(chunks)
        else:
            st.warning("No guideline files found in data/sample/ or data/raw/.")

    return QAAgent(store=store)


agent = load_agent()

question = st.text_input(
    "Your clinical question:",
    placeholder="e.g., What are the USPSTF screening recommendations for colorectal cancer?",
)

if question:
    with st.spinner("Retrieving evidence and generating answer..."):
        result = agent.ask(question)

    # Confidence indicator
    confidence = result["confidence"]
    if result["abstained"]:
        st.error(f"Insufficient evidence (confidence: {confidence:.0%})")
    else:
        st.success(f"Retrieval confidence: {confidence:.0%}")

    # Answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Citations
    if result["citations"]:
        st.subheader("Sources")
        for citation in result["citations"]:
            st.caption(citation)

    # Retrieved passages (expandable)
    with st.expander("Retrieved passages", expanded=False):
        for i, chunk in enumerate(result["retrieved_chunks"]):
            score = chunk.get("score", 0.0)
            st.markdown(
                f"**[{i + 1}]** {chunk['heading']} "
                f"({chunk['source_file']}, p.{chunk['page_numbers']}) "
                f"— similarity: {score:.3f}"
            )
            st.text(chunk["text"][:500])
            st.divider()
