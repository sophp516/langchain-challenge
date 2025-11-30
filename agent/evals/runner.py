"""Evaluation runner for citation accuracy testing."""
import asyncio
import json
from typing import Optional
from datetime import datetime
from .report_quality import CitationAccuracyEvaluator


async def evaluate_citations(
    report_text: str,
    output_file: Optional[str] = None
) -> dict:
    """
    Evaluate citation accuracy in a report.
    
    Args:
        report_text: Generated report text
        output_file: Optional path to save JSON results
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = CitationAccuracyEvaluator()
    result = evaluator.evaluate(report_text)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "citation_accuracy": {
            "score": result.accuracy_score,
            "total_citations": result.total_citations,
            "valid_citations": result.valid_citations,
            "invalid_citations": result.invalid_citations,
            "missing_sources": result.missing_sources,
            "issues": result.issues
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("CITATION ACCURACY EVALUATION")
    print("="*60)
    print(f"Accuracy Score: {result.accuracy_score:.2%}")
    print(f"Total Citations: {result.total_citations}")
    print(f"Valid: {result.valid_citations} | Invalid: {result.invalid_citations} | Missing Sources: {result.missing_sources}")
    if result.issues:
        print(f"\n⚠️  Issues Found ({len(result.issues)}):")
        for issue in result.issues[:5]:
            print(f"  - {issue}")
    print("="*60)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    
    return results


# Example usage
if __name__ == "__main__":
    async def test():
        report = """
# Test Report

This is a test [1][2]. More content [3].

## References
[1] Source 1
[2] Source 2
[3] Source 3
"""
        await evaluate_citations(report)
    
    asyncio.run(test())
