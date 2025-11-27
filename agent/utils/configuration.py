from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient
import os



class AgentConfig(BaseModel):
    """Configuration for the deep research agent - can be passed from UI"""

    # Search API settings
    search_api: Literal["tavily", "serper"] = Field(
        default="serper",
        description="Which search API to use for research"
    )

    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results per query"
    )

    # Research depth settings
    max_research_depth: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum depth for multi-layer research (1-5)"
    )

    max_clarification_rounds: int = Field(
        default=0,  # TESTING: Disabled clarification to test core flow
        ge=0,
        le=10,
        description="Maximum number of clarification questions (0-10)"
    )

    max_revision_rounds: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of report revision attempts"
    )

    # User feedback settings
    enable_user_feedback: bool = Field(
        default=True,
        description="Whether to collect user feedback on the report"
    )

    # Cross-reference verification
    enable_cross_verification: bool = Field(
        default=True,
        description="Whether to verify cross-references between sources"
    )

    min_credibility_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum credibility score for sources (0.0-1.0)"
    )

    # MCP fetch settings (for full content extraction)
    enable_mcp_fetch: bool = Field(
        default=True,
        description="Enable MCP fetch for full article content extraction"
    )

    max_mcp_fetches: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Maximum number of URLs to fetch full content for per section (0-10)"
    )


def get_config_from_configurable(configurable: dict) -> AgentConfig:
    """
    Extract agent configuration from LangGraph configurable dict.
    Filters out system keys like thread_id, checkpoint_ns, etc.
    """
    # Filter out non-config keys
    config_keys = AgentConfig.model_fields.keys()
    config_dict = {k: v for k, v in configurable.items() if k in config_keys}
    return AgentConfig(**config_dict)



try:
    class SerperClient:
        """Wrapper for Serper.dev API using LangChain's GoogleSerperAPIWrapper"""

        def __init__(self, api_key: str):
            self.wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)

        def search(self, query: str, max_results: int = 10, search_depth: str = "basic") -> dict:
            """
            Perform a search using Serper API via LangChain wrapper
            Returns results in a format compatible with Tavily
            """
            try:
                # Get results from LangChain wrapper
                raw_results = self.wrapper.results(query)

                # Convert to Tavily-like format
                results = []
                for item in raw_results.get("organic", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "content": item.get("snippet", ""),
                        "score": 0.8
                    })

                return {"results": results}

            except Exception as e:
                print(f"Serper API error: {e}")
                return {"results": []}

    SERPER_AVAILABLE = True
except ImportError:
    SerperClient = None  # Define as None if import fails
    print("WARNING: langchain-community not installed, Serper search unavailable")
    SERPER_AVAILABLE = False


# Initialize Web Search Clients
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=tavily_api_key)

serper_api_key = os.getenv("SERPER_API_KEY")
if SERPER_AVAILABLE and serper_api_key:
    serper_client = SerperClient(api_key=serper_api_key)
else:
    serper_client = None
    if not serper_api_key:
        print("WARNING: SERPER_API_KEY not set, Serper search will not be available")


import asyncio
import os
from typing import Optional, Dict, List
import re
from firecrawl import FirecrawlApp

try:
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    print("WARNING: firecrawl-py not installed. Run: pip install firecrawl-py")


class FirecrawlFetchClient:
    """Client for Firecrawl API to extract full article content"""

    # TODO: Fix timeout issue for better success rate [low priority]
    def __init__(self):
        self.max_content_length = 10000  # Max chars per article (prevent token overflow)
        self.timeout_ms = 10000  # 10 second timeout (aggressive)
        self.skip_domains = [
            'arxiv.org/pdf',  # PDFs timeout issue
            'academia.edu',   # Often slow
            'sciencedirect.com',  # Paywall/slow
        ]

        # Initialize Firecrawl client
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            print("WARNING: FIRECRAWL_API_KEY not set. Full content extraction will be disabled.")
            self.client = None
        elif not FIRECRAWL_AVAILABLE:
            print("WARNING: firecrawl-py not installed. Full content extraction will be disabled.")
            self.client = None
        else:
            self.client = FirecrawlApp(api_key=api_key)

    async def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch full content from a single URL using Firecrawl API.

        Args:
            url: The URL to fetch

        Returns:
            Markdown content of the page, or None if fetch fails
        """
        if not self.client:
            return None

        # Skip problematic domains
        for skip_domain in self.skip_domains:
            if skip_domain in url:
                print(f"Skipping known slow domain: {url[:60]}...")
                return None

        try:
            # Run Firecrawl scrape in thread pool with timeout wrapper
            loop = asyncio.get_event_loop()

            # Wrap in asyncio timeout to prevent hanging
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.scrape(
                        url,
                        formats=['markdown'],
                        timeout=self.timeout_ms,  # 10 second timeout (milliseconds)
                        only_main_content=True,  # Skip headers/footers/nav for faster extraction
                        mobile=False  # Desktop mode is faster
                    )
                ),
                timeout=15.0  # Overall 15 second timeout at Python level
            )

            # Extract markdown content
            # Response is a Document object with .markdown attribute
            if response and hasattr(response, 'markdown'):
                markdown_content = response.markdown

                # Clean and truncate
                cleaned = self._clean_markdown(markdown_content)
                return cleaned[:self.max_content_length]

            return None

        except asyncio.TimeoutError:
            print(f"Firecrawl timeout (15s exceeded): {url[:60]}...")
            return None
        except Exception as e:
            error_msg = str(e)
            # Suppress verbose timeout errors
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                print(f"Firecrawl timeout: {url[:60]}...")
            else:
                print(f"Firecrawl error for {url[:60]}...: {error_msg[:100]}")
            return None

    async def fetch_multiple(self, urls: List[str], max_concurrent: int = 2) -> Dict[str, str]:
        """
        Fetch full content from multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch
            max_concurrent: Maximum concurrent fetches (default 2 to avoid rate limits)

        Returns:
            Dict mapping URL to markdown content (only successful fetches)
        """
        # Limit concurrent requests to avoid rate limits and timeouts
        # Lower concurrency = more reliable fetches
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url: str) -> tuple[str, Optional[str]]:
            async with semaphore:
                print(f"      Firecrawl fetching: {url[:60]}...")
                content = await self.fetch_url(url)
                if content:
                    print(f"        ✓ Success: {len(content)} chars")
                return (url, content)

        # Fetch all URLs
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed fetches and exceptions
        successful = {}
        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                url, content = result
                successful[url] = content

        print(f"      Firecrawl: {len(successful)}/{len(urls)} successful")
        return successful

    def _clean_markdown(self, markdown: str) -> str:
        """
        Clean and normalize markdown content from MCP fetch.

        - Remove excessive whitespace
        - Remove navigation elements
        - Clean up code blocks
        """
        # Remove multiple consecutive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', markdown)

        # Remove common navigation/footer patterns
        nav_patterns = [
            r'Skip to (?:main )?content',
            r'Table of Contents',
            r'Share on (?:Twitter|Facebook|LinkedIn)',
            r'Subscribe to (?:our )?newsletter',
            r'Follow us on'
        ]
        for pattern in nav_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Trim whitespace
        cleaned = cleaned.strip()

        return cleaned


# Global instance (lazy initialization)
_firecrawl_client: Optional[FirecrawlFetchClient] = None


def get_firecrawl_client() -> FirecrawlFetchClient:
    """Get or create the global Firecrawl fetch client instance"""
    global _firecrawl_client
    if _firecrawl_client is None:
        _firecrawl_client = FirecrawlFetchClient()
    return _firecrawl_client


async def fetch_full_content(urls: List[str], max_urls: int = 5) -> Dict[str, str]:
    """
    Convenience function to fetch full content from URLs.

    Args:
        urls: List of URLs to fetch
        max_urls: Maximum number of URLs to fetch (default 5)

    Returns:
        Dict mapping URL to full markdown content
    """
    client = get_firecrawl_client()
    limited_urls = urls[:max_urls]
    return await client.fetch_multiple(limited_urls)
