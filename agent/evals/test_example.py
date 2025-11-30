"""Example test cases for citation accuracy evaluation."""
import asyncio
from runner import evaluate_citations


async def test_perfect_citations():
    """Test report with perfect citations."""
    report = """
# AI in Healthcare

AI is transforming healthcare [1][2]. Machine learning improves diagnostics [3].

## References
[1] Smith, J. (2023). AI in Medicine. Journal of Healthcare.
[2] Doe, A. (2024). Healthcare Technology. Medical Review.
[3] Lee, B. (2023). ML Diagnostics. AI Journal.
"""
    print("\n✅ Test: Perfect Citations")
    await evaluate_citations(report)


async def test_invalid_citations():
    """Test report with invalid citations."""
    report = """
# Climate Change

Climate change is real [1][5]. The evidence is clear [99].

## References
[1] IPCC Report 2023
[2] NASA Climate Data
"""
    print("\n❌ Test: Invalid Citations")
    await evaluate_citations(report)


async def test_missing_citations():
    """Test report with sources not cited."""
    report = """
# Quantum Computing

Quantum computing is advancing [1].

## References
[1] IBM Quantum Research
[2] Google Quantum AI
[3] Microsoft Quantum Lab
"""
    print("\n⚠️  Test: Missing Citations")
    await evaluate_citations(report)


if __name__ == "__main__":
    asyncio.run(test_perfect_citations())
    asyncio.run(test_invalid_citations())
    asyncio.run(test_missing_citations())

