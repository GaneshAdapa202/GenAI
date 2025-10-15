import os
import requests
from typing import List, Dict, Optional

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

def tavily_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Simple wrapper for Tavily search. Returns a list of result dicts with keys:
    'title', 'url', 'snippet', 'raw_content' (when available)
    Make sure TAVILY_API_KEY is set in your environment (.env).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set in environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_results": max_results
    }
    resp = requests.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # tavily returns structured results — keep the fields we need
    items = []
    for r in data.get("results", [])[:max_results]:
        items.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("snippet") or r.get("answer") or "",
            "raw_content": r.get("raw_content") or ""
        })
    return items

def basic_extract_job_from_result(result: Dict, default_location: Optional[str] = None) -> Dict:
    """
    extractor that builds a job dict from a Tavily search result.
    """
    title = result.get("title") or ""
    snippet = result.get("snippet") or ""
    url = result.get("url")
    # naive company/location guesses from title/snippet:
    company = None
    location = default_location
    # Try simple heuristics: if " at " in title, split "Title at Company"
    if " at " in title:
        parts = title.split(" at ")
        title = parts[0].strip()
        company = parts[1].strip()
    # fallback: try to find " - Company" style
    if not company and " - " in title:
        parts = title.split(" - ")
        if len(parts) >= 2:
            company = parts[-1].strip()
            title = " - ".join(parts[:-1]).strip()

    return {
        "title": title,
        "company": company,
        "location": location,
        "description": snippet,
        "url": url,
        "date_posted": None  # Tavily may return dates elsewhere — expand later
    }
