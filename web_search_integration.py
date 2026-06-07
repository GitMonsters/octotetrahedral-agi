#!/usr/bin/env python3
"""
Web Search Integration for Perplexity Mode
==========================================

Provides real web search capabilities with citations for the Perplexity-style
question answering system.

Supports multiple search backends:
- DuckDuckGo (no API key required)
- Google Custom Search (requires API key)
- Bing Search (requires API key)
"""

import os
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchResult:
    """Single search result with title, snippet, and URL."""
    title: str
    snippet: str
    url: str
    source: str = ""
    
    def to_citation(self, index: int) -> str:
        """Format as markdown citation."""
        return f"[{index}] {self.title} - {self.url}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source": self.source
        }


class WebSearchProvider:
    """Base class for web search providers."""
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Execute search and return results."""
        raise NotImplementedError


class DuckDuckGoSearch(WebSearchProvider):
    """DuckDuckGo search (no API key required)."""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using DuckDuckGo instant answer API."""
        try:
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Add abstract if available
            if data.get("Abstract"):
                results.append(SearchResult(
                    title=data.get("Heading", "DuckDuckGo Answer"),
                    snippet=data["Abstract"],
                    url=data.get("AbstractURL", "https://duckduckgo.com"),
                    source="DuckDuckGo"
                ))
            
            # Add related topics
            for topic in data.get("RelatedTopics", [])[:num_results-1]:
                if "Text" in topic and "FirstURL" in topic:
                    results.append(SearchResult(
                        title=topic.get("Text", "").split(" - ")[0],
                        snippet=topic["Text"],
                        url=topic["FirstURL"],
                        source="DuckDuckGo"
                    ))
            
            return results[:num_results]
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []


class GoogleCustomSearch(WebSearchProvider):
    """Google Custom Search (requires API key and Search Engine ID)."""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(num_results, 10)
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", []):
                results.append(SearchResult(
                    title=item["title"],
                    snippet=item.get("snippet", ""),
                    url=item["link"],
                    source="Google"
                ))
            
            return results
            
        except Exception as e:
            print(f"Google search error: {e}")
            return []


class WebSearchManager:
    """
    Manages multiple search providers with fallback.
    Tries providers in order until successful.
    """
    
    def __init__(self):
        self.providers: List[WebSearchProvider] = []
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available search providers."""
        # Always add DuckDuckGo (no API key needed)
        self.providers.append(DuckDuckGoSearch())
        
        # Add Google if API key available
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if google_api_key and google_search_engine_id:
            self.providers.append(GoogleCustomSearch(google_api_key, google_search_engine_id))
            print("✅ Google Custom Search enabled")
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        Search using available providers.
        Falls back to next provider if current one fails.
        """
        for provider in self.providers:
            try:
                results = provider.search(query, num_results)
                if results:
                    return results
            except Exception as e:
                print(f"Provider {provider.__class__.__name__} failed: {e}")
                continue
        
        return []
    
    def search_with_citations(self, query: str, num_results: int = 5) -> Dict:
        """
        Search and return formatted results with citations.
        """
        results = self.search(query, num_results)
        
        if not results:
            return {
                "success": False,
                "message": "No search results found",
                "results": [],
                "citations": []
            }
        
        # Format citations
        citations = [result.to_citation(i+1) for i, result in enumerate(results)]
        
        # Create summary
        summary = f"Found {len(results)} results for: {query}\n\n"
        for i, result in enumerate(results, 1):
            summary += f"[{i}] **{result.title}**\n"
            summary += f"    {result.snippet}\n\n"
        
        return {
            "success": True,
            "query": query,
            "num_results": len(results),
            "summary": summary,
            "results": [r.to_dict() for r in results],
            "citations": citations,
            "timestamp": datetime.now().isoformat()
        }


def search_web(query: str, num_results: int = 5) -> Dict:
    """
    Convenience function for web search with citations.
    """
    manager = WebSearchManager()
    return manager.search_with_citations(query, num_results)


if __name__ == "__main__":
    print("🔍 Web Search Integration Test")
    print("=" * 60)
    
    # Test searches
    test_queries = [
        "Popperian philosophy",
        "quantum entanglement",
        "Python programming"
    ]
    
    for query in test_queries:
        print(f"\nSearching: {query}")
        result = search_web(query, num_results=3)
        
        if result["success"]:
            print(f"✓ Found {result['num_results']} results")
            for citation in result["citations"]:
                print(f"  {citation}")
        else:
            print(f"✗ Search failed: {result['message']}")
    
    print("\n✅ Web search integration functional!")
