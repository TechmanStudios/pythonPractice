import os
import requests
from bs4 import BeautifulSoup
import gzip

DOWNLOADS_DIR = "downloads"
BASE_URL = "https://sacred-texts.com/download.htm"


def fetch_ebook_links():
    """Fetch the download page and extract ebook titles and .txt.gz links."""
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    ebooks = []
    # Find all numbered list items with .txt.gz links
    for li in soup.find_all("li"):  # All list items
        a = li.find("a", href=True)
        if a and a['href'].endswith(".txt.gz"):
            title = li.get_text(strip=True)
            url = a['href']
            if not url.startswith("http"):
                url = f"https://sacred-texts.com/{url.lstrip('/')}"
            ebooks.append({"title": title, "url": url})
    return ebooks


def download_and_extract(ebook):
    """Download and extract a .txt.gz ebook."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    filename = os.path.basename(ebook["url"])
    gz_path = os.path.join(DOWNLOADS_DIR, filename)
    txt_path = gz_path[:-3]  # Remove .gz
    # Download
    print(f"Downloading: {ebook['title']} -> {filename}")
    resp = requests.get(ebook["url"], stream=True)
    resp.raise_for_status()
    with open(gz_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    # Extract
    with gzip.open(gz_path, "rb") as gz_in, open(txt_path, "wb") as txt_out:
        txt_out.write(gz_in.read())
    print(f"Saved: {txt_path}")


def main():
    ebooks = fetch_ebook_links()
    print(f"Found {len(ebooks)} ebooks.")
    for ebook in ebooks:
        try:
            download_and_extract(ebook)
        except Exception as e:
            print(f"Failed to download {ebook['title']}: {e}")


if __name__ == "__main__":
    main()
