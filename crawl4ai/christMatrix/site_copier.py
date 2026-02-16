import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://christmatrix.com/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "site_copy")

# File extensions to save
ASSET_EXTENSIONS = [".css", ".js"]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_file(url, rel_path):
    out_path = os.path.join(OUTPUT_DIR, rel_path)
    ensure_dir(os.path.dirname(out_path))
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved: {rel_path}")
    except Exception as e:
        print(f"Failed to save {url}: {e}")


def get_asset_links(soup, base_url):
    assets = set()
    # CSS
    for link in soup.find_all("link", rel="stylesheet"):
        href = link.get("href")
        if href:
            assets.add(urljoin(base_url, href))
    # JS
    for script in soup.find_all("script"):
        src = script.get("src")
        if src:
            assets.add(urljoin(base_url, src))
    return assets


def get_html_links(soup, base_url):
    links = set()
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and urlparse(href).netloc in ("", urlparse(base_url).netloc):
            full_url = urljoin(base_url, href)
            if full_url.startswith(base_url):
                links.add(full_url)
    return links


def rel_path_from_url(url):
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    if not path or path.endswith("/"):
        path += "index.html"
    if parsed.query:
        path += "_" + parsed.query.replace("/", "_")
    return path


def crawl_and_save(base_url, max_pages=300):
    visited = set()
    to_visit = [base_url]
    count = 0
    while to_visit and count < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
            html = resp.text
            rel_path = rel_path_from_url(url)
            save_file(url, rel_path)
            soup = BeautifulSoup(html, "html.parser")
            # Find and save assets
            for asset_url in get_asset_links(soup, url):
                ext = os.path.splitext(asset_url)[1].lower()
                if ext in ASSET_EXTENSIONS:
                    asset_rel = rel_path_from_url(asset_url)
                    save_file(asset_url, asset_rel)
            # Queue up more HTML pages
            for link in get_html_links(soup, url):
                if link not in visited and link not in to_visit:
                    to_visit.append(link)
            visited.add(url)
            count += 1
        except Exception as e:
            print(f"Failed to process {url}: {e}")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    crawl_and_save(BASE_URL)
    print(f"Site copy complete. Files saved in {OUTPUT_DIR}")
