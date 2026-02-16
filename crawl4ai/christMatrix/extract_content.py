import os
import requests
from bs4 import BeautifulSoup

URLS_MD = os.path.join(os.path.dirname(__file__), "site_urls.md")
OUTPUT_MD = os.path.join(os.path.dirname(__file__), "site_content.md")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

def get_urls_from_md(urls_md):
    urls = []
    with open(urls_md, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                url = line[2:].strip()
                urls.append(url)
    return urls

def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")
    # Try to extract main content area (WordPress: 'entry-content', 'post-content', or <main>)
    main = soup.find(class_="entry-content") or soup.find(class_="post-content") or soup.find("main")
    if not main:
        # Fallback: get the largest <article> or <div>
        articles = soup.find_all("article")
        if articles:
            main = max(articles, key=lambda a: len(a.get_text()))
        else:
            divs = soup.find_all("div")
            if divs:
                main = max(divs, key=lambda d: len(d.get_text()))
    return main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

def fetch_and_save_markdown(urls, output_md):
    with open(output_md, "w", encoding="utf-8") as fout:
        for i, url in enumerate(urls, 1):
            try:
                resp = requests.get(url, timeout=15, headers=HEADERS)
                resp.raise_for_status()
                text = extract_main_content(resp.text)
                fout.write(f"\n\n---\n\n# Source: {url}\n\n")
                fout.write(text)
                fout.write("\n")
                print(f"[{i}/{len(urls)}] Extracted: {url}")
            except Exception as e:
                print(f"Failed to extract {url}: {e}")
    print(f"All content saved to {output_md}")

if __name__ == "__main__":
    urls = get_urls_from_md(URLS_MD)
    fetch_and_save_markdown(urls, OUTPUT_MD)
