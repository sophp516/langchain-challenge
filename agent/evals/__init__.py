"""Evaluation framework for citation accuracy."""
from .report_quality import CitationAccuracyEvaluator, CitationAccuracyResult
from .runner import evaluate_citations

__all__ = ["CitationAccuracyEvaluator", "CitationAccuracyResult", "evaluate_citations"]
