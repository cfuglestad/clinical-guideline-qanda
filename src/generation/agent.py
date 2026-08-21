"""LangGraph-based RAG agent for clinical guideline Q&A.

The agent follows a retrieve-then-generate pattern:
1. Retrieve relevant chunks from the vector store
2. Assess retrieval confidence (abstain if too low)
3. Generate an answer with explicit citations
"""

from dataclasses import dataclass
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.retrieval.store import GuidelineStore

DEFAULT_MODEL = "gemini-1.5-flash"
ABSTENTION_THRESHOLD = 0.4  # Minimum top-result similarity to proceed


class AgentState(TypedDict):
    """State passed between graph nodes."""

    question: str
    retrieved_chunks: list[dict[str, object]]
    answer: str
    citations: list[str]
    confidence: float
    abstained: bool


@dataclass
class QAAgent:
    """Clinical guideline Q&A agent with retrieval and citation."""

    store: GuidelineStore
    model_name: str = DEFAULT_MODEL
    n_results: int = 5
    abstention_threshold: float = ABSTENTION_THRESHOLD

    def __post_init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0)
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:  # type: ignore[type-arg]
        """Construct the retrieval-generation graph."""
        graph = StateGraph(AgentState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("assess", self._assess_confidence)
        graph.add_node("generate", self._generate)
        graph.add_node("abstain", self._abstain)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges(
            "assess",
            self._should_generate,
            {"generate": "generate", "abstain": "abstain"},
        )
        graph.add_edge("generate", END)
        graph.add_edge("abstain", END)

        return graph

    def _retrieve(self, state: AgentState) -> dict[str, object]:
        """Retrieve relevant chunks from the guideline store."""
        chunks = self.store.query(state["question"], n_results=self.n_results)
        return {"retrieved_chunks": chunks}

    def _assess_confidence(self, state: AgentState) -> dict[str, object]:
        """Compute retrieval confidence from top similarity scores."""
        chunks = state["retrieved_chunks"]
        if not chunks:
            return {"confidence": 0.0}
        top_score = float(chunks[0].get("score", 0.0))  # type: ignore[arg-type]
        return {"confidence": top_score}

    def _should_generate(self, state: AgentState) -> str:
        """Route to generation or abstention based on confidence."""
        if state["confidence"] >= self.abstention_threshold:
            return "generate"
        return "abstain"

    def _generate(self, state: AgentState) -> dict[str, object]:
        """Generate an answer with citations from retrieved chunks."""
        chunks = state["retrieved_chunks"]
        context = "\n\n".join(
            f"[{i + 1}] ({c['source_file']}, p.{c['page_numbers']}, "
            f"section: {c['heading']})\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        prompt = (
            "You are a clinical guideline assistant. Answer the question using ONLY "
            "the provided guideline excerpts. Cite each claim using [N] notation "
            "matching the source numbers. If the excerpts don't contain enough "
            "information to fully answer, say so explicitly.\n\n"
            f"GUIDELINE EXCERPTS:\n{context}\n\n"
            f"QUESTION: {state['question']}\n\n"
            "ANSWER (with [N] citations):"
        )

        response = self._llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        citations = [
            f"[{i + 1}] {c['source_file']}, p.{c['page_numbers']}, section: {c['heading']}"
            for i, c in enumerate(chunks)
        ]

        return {"answer": answer, "citations": citations, "abstained": False}

    def _abstain(self, state: AgentState) -> dict[str, object]:
        """Return an explicit abstention when confidence is too low."""
        return {
            "answer": (
                "I don't have enough evidence in the loaded guidelines to answer "
                "this question reliably. The most relevant passages I found scored "
                f"below my confidence threshold ({self.abstention_threshold:.0%})."
            ),
            "citations": [],
            "abstained": True,
        }

    def ask(self, question: str) -> AgentState:
        """Run the full retrieve-generate pipeline.

        Args:
            question: A clinical question to answer.

        Returns:
            Final agent state with answer, citations, and confidence.
        """
        initial_state: AgentState = {
            "question": question,
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "abstained": False,
        }
        compiled = self._graph.compile()
        result = compiled.invoke(initial_state)
        return result  # type: ignore[return-value]
