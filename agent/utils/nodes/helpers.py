from typing import Dict, List, Tuple
from urllib.parse import urlparse
import re

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


def format_source_as_markdown_link(source: str) -> str:
    """
    Format a source as a proper markdown link.
    Handles:
    - Raw URLs: "https://example.com" -> "[example.com](https://example.com)"
    - Title (URL) format: "Title (https://example.com)" -> "[Title](https://example.com)"
    - Already formatted or other: return as-is
    """
    # Check if it's "Title (URL)" format
    title_url_match = re.match(r'^(.+?)\s*\((https?://[^\)]+)\)$', source)
    if title_url_match:
        title = title_url_match.group(1).strip()
        url = title_url_match.group(2)
        return f"[{title}]({url})"

    # Check if it's a raw URL
    if re.match(r'^https?://', source):
        # Extract domain for display text
        domain_match = re.match(r'https?://(?:www\.)?([^/]+)', source)
        display = domain_match.group(1) if domain_match else source
        return f"[{display}]({source})"

    # Return as-is if already formatted or unknown format
    return source


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
