"""
Web search tool for searching the internet.
"""

import logging
from typing import Any, Dict, List, Optional
import re
import html

import requests
from pydantic import BaseModel, Field

from ..base import Tool


class WebSearchParameters(BaseModel):
    """Parameters for the web search tool."""

    query: str = Field(description="Search query")
    max_results: int = Field(
        default=5, description="Maximum number of results to return"
    )
    safe_search: bool = Field(default=True, description="Enable safe search filtering")


class WebSearchTool(Tool):
    """
    Web search tool for searching the internet.

    Implements:
      1. DuckDuckGo Instant Answer API (JSON) for direct answers / definitions.
      2. DuckDuckGo HTML endpoint parsing for general web results (no mock data).
    """

    name = "web_search"
    description = "Search the internet for information"
    version = "1.0.0"

    Parameters = WebSearchParameters

    def __init__(self):
        """Initialize the web search tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        # DuckDuckGo doesn't require API keys
        self.ddg_api_url = "https://api.duckduckgo.com/"
        self.ddg_html_url = "https://html.duckduckgo.com/html/"

    def execute(
        self, query: str, max_results: int = 5, safe_search: bool = True
    ) -> Dict[str, Any]:
        """
        Search the internet for information.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            safe_search: Enable safe search filtering

        Returns:
            Dictionary containing search results
        """
        try:
            # 1. Attempt Instant Answer
            instant_answer = self._get_duckduckgo_instant_answer(query)
            if instant_answer:
                return {
                    "success": True,
                    "result": {
                        "query": query,
                        "results": [instant_answer],
                        "total_results": 1,
                        "safe_search": safe_search,
                        "source": "DuckDuckGo Instant Answer API",
                    },
                    "error": None,
                }

            # 2. Fallback to HTML search parsing
            search_results = self._get_duckduckgo_search_results(
                query=query, max_results=max_results, safe_search=safe_search
            )

            return {
                "success": True,
                "result": {
                    "query": query,
                    "results": search_results,
                    "total_results": len(search_results),
                    "safe_search": safe_search,
                    "source": "DuckDuckGo Web Search",
                },
                "error": None,
            }

        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return {"success": False, "result": None, "error": str(e)}

    def _get_duckduckgo_instant_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """Get instant answer from DuckDuckGo API."""
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            response = requests.get(self.ddg_api_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Check for instant answer
            if data.get("AbstractText"):
                return {
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                    "source": data.get("AbstractSource", "DuckDuckGo"),
                    "rank": 1,
                    "type": "instant_answer",
                }

            # Check for definition
            if data.get("Definition"):
                return {
                    "title": f"Definition: {query}",
                    "url": data.get("DefinitionURL", ""),
                    "snippet": data.get("Definition", ""),
                    "source": data.get("DefinitionSource", "DuckDuckGo"),
                    "rank": 1,
                    "type": "definition",
                }

            return None

        except Exception as e:
            self.logger.debug(f"DuckDuckGo instant answer failed: {e}")
            return None

    def _get_duckduckgo_search_results(
        self, query: str, max_results: int, safe_search: bool
    ) -> List[Dict[str, Any]]:
        """
        Perform a DuckDuckGo HTML search and parse organic results.

        NOTE:
          - This uses the public HTML interface (no API key).
          - Parsing is heuristic and may require maintenance if markup changes.
        """
        results: List[Dict[str, Any]] = []
        try:
            # DuckDuckGo HTML form prefers POST with 'q'
            # Safe search: 'kp' parameter (values: -2 = off, -1 = moderate, 1 = strict)
            # We'll map: True -> 1 (strict), False -> -2 (off)
            kp_value = "1" if safe_search else "-2"

            resp = requests.post(
                self.ddg_html_url,
                data={"q": query, "kp": kp_value},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36"
                },
                timeout=15,
            )
            resp.raise_for_status()
            html_text = resp.text

            # Pattern for result title anchors
            # Example anchor: <a rel="nofollow" class="result__a" href="https://example.com">Title</a>
            anchor_pattern = re.compile(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )

            # Snippet pattern (following sibling with class result__snippet)
            snippet_pattern = re.compile(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="[^"]+"[^>]*>.*?</a>.*?(?:<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>)',
                re.IGNORECASE | re.DOTALL,
            )

            # Extract anchors
            for match in anchor_pattern.finditer(html_text):
                url = match.group(1)
                raw_title = match.group(2)
                title = html.unescape(re.sub(r"<[^>]+>", " ", raw_title)).strip()
                # Attempt to find a snippet starting from end of this anchor
                start_pos = match.end()
                snippet_match = snippet_pattern.search(
                    html_text,
                    pos=match.start(),
                    endpos=min(len(html_text), match.end() + 4000),
                )
                snippet = ""
                if snippet_match:
                    snippet_raw = snippet_match.group(1) or ""
                    snippet = html.unescape(
                        re.sub(r"<[^>]+>", " ", snippet_raw)
                    ).strip()
                if not snippet:
                    # Fallback: take nearby text window
                    window = html_text[start_pos : start_pos + 600]
                    window_clean = html.unescape(re.sub(r"<[^>]+>", " ", window))
                    snippet = " ".join(window_clean.split())[:240]

                results.append(
                    {
                        "title": title or url,
                        "url": url,
                        "snippet": snippet,
                        "source": url.split("/")[2] if "://" in url else url,
                        "type": "web_result",
                    }
                )
                if len(results) >= max_results:
                    break

        except Exception as e:
            self.logger.debug(f"DuckDuckGo search parse failed: {e}")

        # Rank & metadata
        for idx, r in enumerate(results, start=1):
            r["rank"] = idx
            r["cached"] = False
        return results

    def get_search_engines(self) -> List[str]:
        """
        Get list of supported search engines (only DuckDuckGo currently implemented).
        """
        return ["duckduckgo"]

    def get_help(self) -> Dict[str, Any]:
        """
        Get help information for the web search tool.

        Returns:
            Dictionary containing help information
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": "Text to search for",
                "max_results": "Maximum number of results (1-20)",
                "safe_search": "Enable safe search filtering",
            },
            "examples": [
                "Search for Python programming tutorials",
                "Find news about artificial intelligence",
                "Look up weather forecast information",
            ],
            "supported_engines": self.get_search_engines(),
            "features": [
                "DuckDuckGo instant answers",
                "DuckDuckGo HTML organic results parsing",
                "Safe search parameter mapping",
                "Snippet extraction (heuristic)",
            ],
            "note": "Results obtained via DuckDuckGo public endpoints (no mock data).",
        }
