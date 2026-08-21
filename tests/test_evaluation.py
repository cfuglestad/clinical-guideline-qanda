"""Tests for the retrieval evaluation framework."""

from pathlib import Path

from src.evaluation.metrics import (
    QAPair,
    evaluate_retrieval,
    load_qa_pairs,
)

DATASET_PATH = Path("evaluation/hypertension_qa_pairs.json")


def _perfect_retrieval(question: str) -> list[dict[str, object]]:
    """Mock retrieval that always returns the right sections."""
    return [
        {"heading": "SUMMARY OF RECOMMENDATION", "score": 0.95},
        {"heading": "RATIONALE", "score": 0.85},
        {"heading": "EVIDENCE REVIEW", "score": 0.75},
        {"heading": "DISCUSSION", "score": 0.65},
        {"heading": "5. Treatment Approaches", "score": 0.55},
    ]


def _empty_retrieval(question: str) -> list[dict[str, object]]:
    """Mock retrieval that returns irrelevant results."""
    return [
        {"heading": "Unrelated Section A", "score": 0.3},
        {"heading": "Unrelated Section B", "score": 0.2},
    ]


def test_load_qa_pairs_from_json() -> None:
    pairs = load_qa_pairs(DATASET_PATH)
    assert len(pairs) == 12
    assert pairs[0].id == "htn-001"
    assert pairs[0].difficulty == "easy"
    assert "SUMMARY OF RECOMMENDATION" in pairs[0].relevant_sections


def test_perfect_retrieval_has_full_recall() -> None:
    pairs = [
        QAPair(
            id="test-1",
            question="What is the recommendation?",
            expected_answer="Grade A.",
            relevant_sections=("SUMMARY OF RECOMMENDATION",),
            difficulty="easy",
        ),
    ]
    report = evaluate_retrieval(pairs, _perfect_retrieval, k=5)
    assert report.mean_recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0


def test_empty_retrieval_has_zero_recall() -> None:
    pairs = [
        QAPair(
            id="test-2",
            question="What is the screening interval?",
            expected_answer="Annual.",
            relevant_sections=("3. Screening Intervals",),
            difficulty="medium",
        ),
    ]
    report = evaluate_retrieval(pairs, _empty_retrieval, k=5)
    assert report.mean_recall_at_k == 0.0
    assert report.mean_reciprocal_rank == 0.0


def test_report_groups_by_difficulty() -> None:
    pairs = [
        QAPair(
            id="easy-1",
            question="Test easy",
            expected_answer="Answer",
            relevant_sections=("SUMMARY OF RECOMMENDATION",),
            difficulty="easy",
        ),
        QAPair(
            id="hard-1",
            question="Test hard",
            expected_answer="Answer",
            relevant_sections=("EVIDENCE REVIEW",),
            difficulty="hard",
        ),
    ]
    report = evaluate_retrieval(pairs, _perfect_retrieval, k=5)
    assert "easy" in report.results_by_difficulty
    assert "hard" in report.results_by_difficulty
    assert report.results_by_difficulty["easy"]["count"] == 1.0
