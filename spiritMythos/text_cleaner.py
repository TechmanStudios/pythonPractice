# Functions for cleaning and filtering scraped text
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# List of keywords or patterns to ignore (ads, social, etc.)
IGNORE_SELECTORS = [
    'nav', 'footer', 'script', 'style', 'aside', 'form', 'header',
    '.social', '.ad', '.ads', '.banner', '.cookie', '.share',
]
IGNORE_KEYWORDS = [
    'facebook', 'twitter', 'instagram', 'pinterest', 'linkedin',
    'advertisement', 'sponsored', 'cookie', 'privacy', 'terms',
]

def clean_text(soup: BeautifulSoup) -> str:
    # Remove unwanted elements
    for selector in IGNORE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    # Extract main text
    text = soup.get_text(separator=' ', strip=True)
    # Remove lines with ignore keywords
    lines = [line for line in text.splitlines() if not any(kw in line.lower() for kw in IGNORE_KEYWORDS)]
    return '\n'.join(lines).strip()

def is_relevant_link(url: str, base_url: str) -> bool:
    # Only follow links within the spiritmythos.org domain
    parsed = urlparse(url)
    return parsed.netloc == '' or base_url in url
