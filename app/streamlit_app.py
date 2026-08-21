"""Streamlit demo interface for Clinical Guideline Q&A."""

import streamlit as st

from src.generation.agent import QAAgent
from src.retrieval.store import GuidelineStore

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
    """Initialize the vector store and QA agent."""
    store = GuidelineStore()
    if store.count == 0:
        st.warning(
            "No guidelines loaded. Run `python -m src.ingestion.ingest` first."
        )
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
