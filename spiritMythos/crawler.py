# crawler.py
"""
Crawl spiritmythos.org and collect all unique, relevant internal URLs.
Output: spiritmythos_urls.txt
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from text_cleaner import is_relevant_link
import time

BASE_URL = "http://www.spiritmythos.org/"
VISITED = set()
ALL_LINKS = set()
START_URLS = [BASE_URL]


def crawl(url):
    if url in VISITED:
        return
    VISITED.add(url)
    print(f"Crawling: {url}")
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a", href=True):
            next_url = urljoin(url, link["href"])
            parsed = urlparse(next_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if is_relevant_link(next_url, BASE_URL):
                if next_url not in VISITED:
                    ALL_LINKS.add(next_url)
                    print(f"    Found internal link: {next_url}")
                    crawl(next_url)
        time.sleep(1)  # Be polite to the server
    except Exception as e:
        print(f"Error crawling {url}: {e}")


def main():
    for url in START_URLS:
        crawl(url)
    with open("spiritmythos_urls.txt", "w", encoding="utf-8") as f:
        for link in sorted(ALL_LINKS):
            f.write(link + "\n")
    print("\nCrawling complete.")
    print("Internal links saved in: spiritmythos_urls.txt")

if __name__ == "__main__":
    main()
