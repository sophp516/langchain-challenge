"""
Verification utilities for source quality filtering and fact-checking
"""
from typing import Dict, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from utils.model import llm
import asyncio
from urllib.parse import urlparse


# Trusted domains for credibility scoring
TRUSTED_DOMAINS = {
    'edu': 0.9,      # Educational institutions
    'gov': 0.95,     # Government sources
    'org': 0.7,      # Organizations (varies)
}

REPUTABLE_SOURCES = {
    'wikipedia.org': 0.75,
    'nature.com': 0.95,
    'science.org': 0.95,
    'arxiv.org': 0.85,
    'nejm.org': 0.95,
    'ieee.org': 0.9,
    'acm.org': 0.9,
    'springer.com': 0.85,
    'sciencedirect.com': 0.85,
    'ncbi.nlm.nih.gov': 0.95,
    'who.int': 0.9,
    'reuters.com': 0.85,
    'apnews.com': 0.85,
    'bbc.com': 0.8,
}

LOW_QUALITY_INDICATORS = [
    'blog', 'forum', 'reddit', 'quora', 'yahoo answers',
    'pinterest', 'tumblr', 'medium.com'
]


def calculate_source_credibility(url: str, title: str, content: str) -> float:
    """
    Calculate credibility score for a source (0.0 to 1.0)
    Based on domain, content quality indicators, and structure
    """
    if not url:
        return 0.5  # Default for unknown sources

    score = 0.5  # Start with neutral score

    # Parse domain
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace('www.', '')
    except:
        return score

    # Check against reputable sources
    for reputable_domain, domain_score in REPUTABLE_SOURCES.items():
        if reputable_domain in domain:
            score = max(score, domain_score)
            return score

    # Check TLD (top-level domain)
    tld = domain.split('.')[-1]
    if tld in TRUSTED_DOMAINS:
        score = max(score, TRUSTED_DOMAINS[tld])

    # Penalize low-quality indicators
    domain_lower = domain.lower()
    for indicator in LOW_QUALITY_INDICATORS:
        if indicator in domain_lower:
            score *= 0.6
            break

    # Check content quality indicators
    if content:
        content_length = len(content)
        if content_length < 200:
            score *= 0.8  # Too short
        elif content_length > 1000:
            score = min(1.0, score * 1.1)  # Substantial content

        # Check for citations/references (simple heuristic)
        if '[' in content or 'http' in content or 'doi:' in content.lower():
            score = min(1.0, score * 1.05)

    return min(1.0, max(0.1, score))  # Clamp between 0.1 and 1.0


async def filter_quality_sources(
    search_results: List[Dict],
    min_credibility: float = 0.4
) -> List[Tuple[Dict, float]]:
    """
    Filter search results by source quality
    Returns list of (result, credibility_score) tuples
    """
    filtered_results = []

    for result in search_results:
        url = result.get('url', '')
        title = result.get('title', '')
        content = result.get('content', '')

        credibility = calculate_source_credibility(url, title, content)

        if credibility >= min_credibility:
            filtered_results.append((result, credibility))

    # Sort by credibility (highest first)
    filtered_results.sort(key=lambda x: x[1], reverse=True)

    return filtered_results


async def extract_claims(text: str) -> List[str]:
    """
    Extract factual claims from text using LLM
    """
    prompt = f"""
    Extract the key factual claims from the following text.
    A factual claim is a statement that can be verified or falsified.

    Text:
    {text[:2000]}  # Limit to avoid token limits

    Return only a list of claims, one per line, without numbering or bullets.
    Focus on the most important claims (maximum 10).
    """

    messages = [
        SystemMessage(content="You are a fact-checking assistant that extracts verifiable claims from text."),
        HumanMessage(content=prompt)
    ]

    response = await llm.ainvoke(messages)
    claims_text = response.content if hasattr(response, 'content') else str(response)

    # Parse claims (one per line)
    claims = [claim.strip() for claim in claims_text.strip().split('\n') if claim.strip()]
    return claims


async def cross_validate_claim(
    claim: str,
    research_results: Dict[str, str]
) -> Dict:
    """
    Cross-validate a claim across multiple sources
    Returns validation result with support level and contradictions
    """
    if not research_results:
        return {
            'claim': claim,
            'support_level': 'unknown',
            'supporting_sources': [],
            'contradicting_sources': [],
            'confidence': 0.0
        }

    # Build context from all sources
    sources_context = ""
    source_list = []
    for i, (source, content) in enumerate(list(research_results.items())[:5]):  # Limit to 5 sources
        sources_context += f"Source {i+1} ({source}):\n{content[:500]}\n\n"
        source_list.append(source)

    validation_prompt = f"""
    Evaluate whether the following claim is supported by the provided sources.

    Claim: {claim}

    Sources:
    {sources_context}

    Determine:
    1. Is the claim SUPPORTED, CONTRADICTED, or UNVERIFIED by these sources?
    2. Which sources support it? (provide source numbers)
    3. Which sources contradict it? (provide source numbers)
    4. What is your confidence level? (0-100)

    Respond in this exact format:
    STATUS: [SUPPORTED/CONTRADICTED/UNVERIFIED]
    SUPPORTING: [comma-separated source numbers or "none"]
    CONTRADICTING: [comma-separated source numbers or "none"]
    CONFIDENCE: [0-100]
    REASONING: [brief explanation]
    """

    messages = [
        SystemMessage(content="You are a fact-checking assistant that validates claims against sources."),
        HumanMessage(content=validation_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Parse response
    status = 'unknown'
    supporting = []
    contradicting = []
    confidence = 0.0

    for line in response_text.split('\n'):
        line = line.strip()
        if line.startswith('STATUS:'):
            status = line.split(':', 1)[1].strip().lower()
        elif line.startswith('SUPPORTING:'):
            supporting_text = line.split(':', 1)[1].strip()
            if supporting_text.lower() != 'none':
                # Parse source numbers
                try:
                    indices = [int(s.strip())-1 for s in supporting_text.split(',') if s.strip().isdigit()]
                    supporting = [source_list[i] for i in indices if i < len(source_list)]
                except:
                    pass
        elif line.startswith('CONTRADICTING:'):
            contradicting_text = line.split(':', 1)[1].strip()
            if contradicting_text.lower() != 'none':
                try:
                    indices = [int(s.strip())-1 for s in contradicting_text.split(',') if s.strip().isdigit()]
                    contradicting = [source_list[i] for i in indices if i < len(source_list)]
                except:
                    pass
        elif line.startswith('CONFIDENCE:'):
            try:
                confidence = float(line.split(':', 1)[1].strip()) / 100.0
            except:
                confidence = 0.5

    return {
        'claim': claim,
        'support_level': status,
        'supporting_sources': supporting,
        'contradicting_sources': contradicting,
        'confidence': confidence
    }


async def detect_contradictions(
    research_results: Dict[str, str]
) -> List[Dict]:
    """
    Detect contradictions between different sources
    """
    if len(research_results) < 2:
        return []

    # Build comparison prompt with source pairs
    sources_list = list(research_results.items())[:5]  # Limit to 5 sources for performance

    comparison_text = ""
    for i, (source, content) in enumerate(sources_list):
        comparison_text += f"Source {i+1} ({source}):\n{content[:400]}\n\n"

    contradiction_prompt = f"""
    Analyze these sources and identify any contradictions or conflicting information.

    {comparison_text}

    List any contradictions you find in this format:
    CONTRADICTION: [brief description]
    SOURCE_A: [source number]
    SOURCE_B: [source number]
    SEVERITY: [LOW/MEDIUM/HIGH]

    If no contradictions, respond with "NO_CONTRADICTIONS"
    """

    messages = [
        SystemMessage(content="You are a fact-checking assistant that identifies contradictions between sources."),
        HumanMessage(content=contradiction_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    if "NO_CONTRADICTIONS" in response_text.upper():
        return []

    # Parse contradictions (simplified)
    contradictions = []
    lines = response_text.split('\n')
    current_contradiction = {}

    for line in lines:
        line = line.strip()
        if line.startswith('CONTRADICTION:'):
            if current_contradiction:
                contradictions.append(current_contradiction)
            current_contradiction = {'description': line.split(':', 1)[1].strip()}
        elif line.startswith('SOURCE_A:'):
            current_contradiction['source_a'] = line.split(':', 1)[1].strip()
        elif line.startswith('SOURCE_B:'):
            current_contradiction['source_b'] = line.split(':', 1)[1].strip()
        elif line.startswith('SEVERITY:'):
            current_contradiction['severity'] = line.split(':', 1)[1].strip()

    if current_contradiction:
        contradictions.append(current_contradiction)

    return contradictions


async def calculate_research_confidence(
    research_results: Dict[str, str],
    source_credibilities: Dict[str, float]
) -> float:
    """
    Calculate overall confidence in research results
    Based on source credibility, agreement, and coverage
    """
    if not research_results:
        return 0.0

    # Average credibility of sources
    credibility_scores = [source_credibilities.get(source, 0.5) for source in research_results.keys()]
    avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5

    # Source count factor (more sources = higher confidence, with diminishing returns)
    source_count = len(research_results)
    count_factor = min(1.0, 0.5 + (source_count * 0.1))  # 0.5 to 1.0

    # Combined confidence
    confidence = avg_credibility * count_factor

    return min(1.0, max(0.0, confidence))