"""Retrieval evaluation metrics for clinical guideline Q&A.

Measures how well the retrieval component surfaces the right chunks
for a given question, using labeled Q&A pairs as ground truth.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QAPair:
    """A labeled question-answer pair with relevant section annotations."""

    id: str
    question: str
    expected_answer: str
    relevant_sections: tuple[str, ...]
    difficulty: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Evaluation result for a single question."""

    question_id: str
    question: str
    relevant_sections: tuple[str, ...]
    retrieved_headings: tuple[str, ...]
    retrieved_scores: tuple[float, ...]
    hits_at_k: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate metrics across all Q&A pairs."""

    total_questions: int
    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    results_by_difficulty: dict[str, dict[str, float]]
    per_question: tuple[RetrievalResult, ...]


def load_qa_pairs(path: Path) -> list[QAPair]:
    """Load labeled Q&A pairs from a JSON file."""
    data = json.loads(path.read_text())
    return [
        QAPair(
            id=p["id"],
            question=p["question"],
            expected_answer=p["expected_answer"],
            relevant_sections=tuple(p["relevant_sections"]),
            difficulty=p["difficulty"],
        )
        for p in data["pairs"]
    ]


def _normalize_heading(heading: str) -> str:
    """Normalize heading text for fuzzy matching."""
    return heading.strip().lower()


def _heading_matches(
    retrieved: str, relevant: tuple[str, ...]
) -> bool:
    """Check if a retrieved heading matches any relevant section.

    Uses substring matching to handle partial heading matches
    (e.g., '4. Confirmatory Blood Pressure Measurement' matches
    'Confirmatory Blood Pressure Measurement').
    """
    retrieved_norm = _normalize_heading(retrieved)
    for section in relevant:
        section_norm = _normalize_heading(section)
        if section_norm in retrieved_norm or retrieved_norm in section_norm:
            return True
    return False


def evaluate_retrieval(
    qa_pairs: list[QAPair],
    retrieval_fn: Callable[[str], list[dict[str, object]]],
    k: int = 5,
) -> EvaluationReport:
    """Evaluate retrieval quality against labeled Q&A pairs.

    Args:
        qa_pairs: Ground-truth question-answer pairs with relevant sections.
        retrieval_fn: Function that takes a question string and returns
            a list of dicts with 'heading' and 'score' keys.
        k: Number of results to evaluate (precision@k, recall@k).

    Returns:
        EvaluationReport with per-question and aggregate metrics.
    """
    results: list[RetrievalResult] = []

    for pair in qa_pairs:
        retrieved = retrieval_fn(pair.question)
        retrieved_headings = tuple(
            str(r.get("heading", "")) for r in retrieved[:k]
        )
        retrieved_scores = tuple(
            float(r.get("score", 0.0) or 0.0)  # type: ignore[arg-type]
            for r in retrieved[:k]
        )

        # Count hits: retrieved chunks whose heading matches a relevant section
        hits = [
            i for i, h in enumerate(retrieved_headings)
            if _heading_matches(h, pair.relevant_sections)
        ]
        hits_at_k = len(hits)

        precision = hits_at_k / k if k > 0 else 0.0
        recall = (
            hits_at_k / len(pair.relevant_sections)
            if pair.relevant_sections
            else 0.0
        )
        # Reciprocal rank: 1/position of first relevant result
        rr = 1.0 / (hits[0] + 1) if hits else 0.0

        results.append(
            RetrievalResult(
                question_id=pair.id,
                question=pair.question,
                relevant_sections=pair.relevant_sections,
                retrieved_headings=retrieved_headings,
                retrieved_scores=retrieved_scores,
                hits_at_k=hits_at_k,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=rr,
            )
        )

    # Aggregate metrics
    n = len(results)
    mean_p = sum(r.precision_at_k for r in results) / n if n else 0.0
    mean_r = sum(r.recall_at_k for r in results) / n if n else 0.0
    mean_rr = sum(r.reciprocal_rank for r in results) / n if n else 0.0

    # Metrics by difficulty
    by_difficulty: dict[str, dict[str, float]] = {}
    for difficulty in ("easy", "medium", "hard"):
        subset = [r for r, p in zip(results, qa_pairs, strict=False)
                  if p.difficulty == difficulty]
        if subset:
            by_difficulty[difficulty] = {
                "count": float(len(subset)),
                "mean_precision_at_k": (
                    sum(r.precision_at_k for r in subset) / len(subset)
                ),
                "mean_recall_at_k": (
                    sum(r.recall_at_k for r in subset) / len(subset)
                ),
                "mean_reciprocal_rank": (
                    sum(r.reciprocal_rank for r in subset) / len(subset)
                ),
            }

    return EvaluationReport(
        total_questions=n,
        k=k,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_reciprocal_rank=mean_rr,
        results_by_difficulty=by_difficulty,
        per_question=tuple(results),
    )


def format_report(report: EvaluationReport) -> str:
    """Format an evaluation report as a readable string."""
    lines = [
        f"Retrieval Evaluation Report (k={report.k})",
        f"{'='*50}",
        f"Total questions: {report.total_questions}",
        f"Mean Precision@{report.k}: {report.mean_precision_at_k:.3f}",
        f"Mean Recall@{report.k}: {report.mean_recall_at_k:.3f}",
        f"Mean Reciprocal Rank: {report.mean_reciprocal_rank:.3f}",
        "",
        "By Difficulty:",
    ]
    for diff, metrics in report.results_by_difficulty.items():
        lines.append(
            f"  {diff}: P@k={metrics['mean_precision_at_k']:.3f} "
            f"R@k={metrics['mean_recall_at_k']:.3f} "
            f"MRR={metrics['mean_reciprocal_rank']:.3f} "
            f"(n={int(metrics['count'])})"
        )
    lines.append("")
    lines.append("Per-Question Results:")
    for r in report.per_question:
        status = "HIT" if r.hits_at_k > 0 else "MISS"
        lines.append(
            f"  [{status}] {r.question_id}: P@k={r.precision_at_k:.2f} "
            f"R@k={r.recall_at_k:.2f} RR={r.reciprocal_rank:.2f}"
        )
        lines.append(f"         Q: {r.question[:60]}...")
    return "\n".join(lines)
