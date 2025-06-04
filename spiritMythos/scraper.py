# Main script for scraping and archiving text from spiritmythos.org
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from text_cleaner import clean_text, is_relevant_link
from pdf_extractor import extract_pdf_text
from whisper_transcriber import transcribe_audio
import os
import time

BASE_URL = "http://www.spiritmythos.org/"
VISITED = set()
ALL_LINKS = set()

# Prompt for output file path at the start
output_path = input("Enter output file path (leave blank for 'spiritmythos_content.txt'): ").strip()
if not output_path:
    output_path = "spiritmythos_content.txt"
OUTPUT_FILE = output_path

# Optionally, load URLs from a file or start with BASE_URL
START_URLS = [BASE_URL]


def crawl(url):
    if url in VISITED:
        return
    VISITED.add(url)
    print(f"Crawling: {url}")
    try:
        ext = os.path.splitext(url)[1].lower()
        soup = None
        if ext == ".pdf":
            print(f"  Downloading and extracting PDF: {url}")
            text = extract_pdf_text(url)
        elif ext in [".html", ".htm", ".php", ""]:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = clean_text(soup)
        elif ext in [".mp3", ".mp4"]:
            print(f"  Downloading and transcribing audio/video: {url}")
            text = transcribe_audio(url)
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            print(f"  Skipping image file: {url}")
            text = ""
        else:
            print(f"  Skipping unsupported file type: {url}")
            text = ""
        if text:
            print(f"  Saving text from: {url}")
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"URL: {url}\n{text}\n\n")
        # Find and crawl relevant links only if soup exists
        if soup is not None:
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


if __name__ == "__main__":
    # Run the crawler first to generate the URL list
    try:
        from crawler import main as run_crawler
        print("Running crawler to generate URL list...")
        run_crawler()
    except Exception as e:
        print(f"Error running crawler: {e}")

    # Prompt for input URL list and output file
    url_list_path = input("Enter input URL list path (leave blank for 'spiritmythos_urls.txt'): ").strip()
    if not url_list_path:
        url_list_path = "spiritmythos_urls.txt"
    output_path = input("Enter output file path (leave blank for 'spiritmythos_content.txt'): ").strip()
    if not output_path:
        output_path = "spiritmythos_content.txt"
    OUTPUT_FILE = output_path

    # Read URLs from file
    with open(url_list_path, "r", encoding="utf-8") as f:
        url_list = [line.strip() for line in f if line.strip()]

    for url in url_list:
        try:
            parsed = urlparse(url)
            path = parsed.path
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                print(f"  Downloading and extracting PDF: {url}")
                text = extract_pdf_text(url)
            elif ext in [".html", ".htm", ".php", ""]:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                text = clean_text(soup)
            elif ext in [".mp3", ".mp4"]:
                print(f"  Downloading and transcribing audio/video: {url}")
                text = transcribe_audio(url)
            elif ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
                print(f"  Skipping image file: {url}")
                text = ""
            else:
                print(f"  Skipping unsupported file type: {url}")
                text = ""
            if text:
                print(f"  Saving text from: {url}")
                with open(OUTPUT_FILE, "a", encoding="utf-8") as outf:
                    outf.write(f"URL: {url}\n{text}\n\n")
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    print("\nScraping complete.")
    print(f"Text archived in: {OUTPUT_FILE}")
