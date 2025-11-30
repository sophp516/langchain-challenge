"""Evaluates citation accuracy in research reports."""
import re
from pydantic import BaseModel, Field
from typing import Dict, List


class CitationAccuracyResult(BaseModel):
    """Results from citation accuracy evaluation."""
    total_citations: int
    valid_citations: int
    invalid_citations: int
    missing_sources: int
    accuracy_score: float
    issues: List[str]


class CitationAccuracyEvaluator:
    """Evaluates citation accuracy in reports."""

    def extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text."""
        return [int(c) for c in re.findall(r'\[(\d+)\]', text) if c.isdigit()]

    def extract_references(self, report_text: str) -> Dict[int, str]:
        """Extract references section mapping numbers to sources."""
        ref_match = re.search(r'##\s*References?\s*\n\n(.*)', report_text, re.IGNORECASE | re.DOTALL)
        if not ref_match:
            return {}
        
        references = {}
        for match in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\n\[|\Z)', ref_match.group(1), re.MULTILINE | re.DOTALL):
            references[int(match.group(1))] = match.group(2).strip()
        return references

    def evaluate(self, report_text: str) -> CitationAccuracyResult:
        """Evaluate citation accuracy."""
        references = self.extract_references(report_text)
        text_without_refs = re.sub(r'##\s*References?\s*\n\n.*', '', report_text, flags=re.IGNORECASE | re.DOTALL)
        citations = self.extract_citations(text_without_refs)
        
        valid = [c for c in citations if c in references]
        invalid = [c for c in citations if c not in references]
        cited_nums = set(citations)
        missing = [r for r in references.keys() if r not in cited_nums]
        
        issues = []
        for c in invalid:
            issues.append(f"Citation [{c}] references non-existent source")
        for m in missing:
            issues.append(f"Source [{m}] in references but never cited")
        if not citations:
            issues.append("No citations found in report")
        
        accuracy = len(valid) / len(citations) if citations else 0.0
        
        return CitationAccuracyResult(
            total_citations=len(citations),
            valid_citations=len(valid),
            invalid_citations=len(invalid),
            missing_sources=len(missing),
            accuracy_score=accuracy,
            issues=issues
        )
