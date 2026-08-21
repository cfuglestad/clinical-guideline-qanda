# Clinical Guideline Q&A

Retrieval-augmented generation over publicly available clinical practice guidelines. Ask a clinical question, get an answer grounded in guideline text with transparent citations and retrieval confidence.

## What it does

1. Ingests clinical practice guidelines (PDF) with section-aware chunking
2. Embeds chunks into a local vector store (ChromaDB + sentence-transformers)
3. Retrieves relevant passages for a user's clinical question
4. Generates an answer with explicit citations back to source chunks
5. Reports retrieval confidence; abstains when evidence is insufficient

## Why this exists

Most RAG demos split documents into arbitrary 500-token windows and call it done. This project treats chunking, retrieval evaluation, and citation grounding as first-class concerns:

- **Section-aware chunking** respects document structure (headings, recommendation boxes) rather than splitting mid-sentence
- **Retrieval evaluation** with labeled question-passage pairs measures precision@k and recall@k
- **Citation grounding** maps every claim in the generated answer to a retrieved chunk
- **Abstention** when retrieval scores are low, rather than hallucinating

## Data sources

All freely available, no PHI:
- USPSTF recommendation statements
- CDC clinical prevention guidelines
- AHA/ACC cardiovascular guidelines

## Tech stack

- **LangGraph** for the retrieval-generation agent workflow
- **ChromaDB** for local vector persistence
- **sentence-transformers** for local embedding (all-MiniLM-L6-v2)
- **OpenAI** (or compatible) for answer generation
- **pypdf** for PDF text extraction
- **Streamlit** for the demo interface
- **Python 3.11+**, typed throughout

## Quick start

```bash
git clone https://github.com/cfuglestad/clinical-guideline-qanda.git
cd clinical-guideline-qanda
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

Ingest guidelines and start the app:
```bash
python -m src.ingestion.ingest
streamlit run app/streamlit_app.py
```

## Project structure

```
src/
  ingestion/     # PDF parsing, section-aware chunking, embedding
  retrieval/     # Vector store queries, reranking
  generation/    # LangGraph agent, prompt templates, citation
  evaluation/    # Labeled Q&A pairs, retrieval metrics
app/             # Streamlit demo interface
data/
  raw/           # Source PDFs (gitignored)
  processed/     # Chunked + embedded (gitignored)
guidelines/      # Download scripts or small reference files
evaluation/      # Gold-standard Q&A pairs + results
tests/           # Unit and integration tests
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

## License

MIT
