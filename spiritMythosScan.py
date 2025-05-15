import requests
from bs4 import BeautifulSoup
import time
import os

# Prompt for custom save path
print("Please provide a directory path to save the output file.")
print("Leave blank to use the current directory.")
save_path = input("Save path: ").strip() or os.getcwd()
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created directory: {save_path}")
urls_file = os.path.join(save_path, 'spiritmythos_urls.txt')
content_file = os.path.join(save_path, 'spiritmythos_content.txt')
print(f"Using URLs from: {urls_file}")
print(f"Content will be saved to: {content_file}")

# --- Adaptive Text Extraction ---
print("\n=== Starting adaptive text extraction ===")

def find_content_area(soup):
    print("Detecting content area...")
    candidates = soup.find_all(['div', 'section', 'article'])
    if not candidates:
        print("No div/section/article found; using <body> as fallback.")
        return soup.body
    
    best_candidate = None
    max_text_length = 0
    for candidate in candidates:
        text = candidate.get_text(strip=True)
        text_length = len(text)
        link_count = len(candidate.find_all('a'))
        if text_length > 100 and (link_count / (text_length / 1000)) < 0.5:
            if text_length > max_text_length:
                max_text_length = text_length
                best_candidate = candidate
                print(f"  Candidate: {candidate.name} (text length: {text_length})")
    
    if best_candidate:
        print(f"Selected content area: {best_candidate.name}")
        return best_candidate
    print("No suitable content area; using <body> as fallback.")
    return soup.body

def extract_adaptive_text(url):
    print(f"Extracting text from {url}...")
    try:
        response = requests.get(url, headers={'User-Agent': 'MyCrawler 1.0'})
        response.raise_for_status()

        # Skip non-HTML content
        if 'text/html' not in response.headers.get('Content-Type', '').lower():
            print("Non-HTML content (e.g., KMZ, PDF); skipping.")
            return None, None, None

        soup = BeautifulSoup(response.text, 'html.parser')
        if not soup.body:
            print("No valid HTML body found; skipping.")
            return None, None, None

        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else "No Title"
        print(f"Title: {title}")

        headers = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        print(f"Found {len(headers)} headers: {headers if headers else 'None'}")

        content_area = find_content_area(soup)
        if not content_area:
            print("Content area is None; skipping.")
            return title, headers, []

        body_text = []
        for elem in content_area.find_all(True, recursive=True):
            text = elem.get_text(strip=True)
            if text and len(text) > 50:
                if elem.name == 'a' or (elem.find('a') and len(text) < 100):
                    continue
                body_text.append(text)
        seen = set()
        body_text = [t for t in body_text if not (t in seen or seen.add(t))]
        print(f"Extracted {len(body_text)} body text blocks.")

        return title, headers, body_text
    except requests.RequestException as e:
        print(f"Error: {e}")
        return None, None, None

# Load URLs from existing file
if not os.path.exists(urls_file):
    print(f"Error: {urls_file} not found. Please provide the correct file.")
    exit(1)

with open(urls_file, 'r') as f:
    urls = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(urls)} URLs from {urls_file}")

# Extract and save text
with open(content_file, 'w', encoding='utf-8') as f:
    for i, url in enumerate(urls):
        print(f"\n=== Processing {i+1}/{len(urls)}: {url} ===")
        title, headers, body_text = extract_adaptive_text(url)
        
        if title and (headers or body_text):
            f.write(f"--- Content from {url} ---\n")
            f.write(f"Title: {title}\n\n")
            if headers:
                f.write("Headers:\n")
                for h in headers:
                    f.write(f"- {h}\n")
                f.write("\n")
            if body_text:
                f.write("Body Text:\n")
                for text in body_text:
                    f.write(f"{text}\n")
                f.write("\n")
        else:
            print("No relevant content extracted.")
        
        print("Pausing for 2 seconds...")
        time.sleep(2)

print("\n=== Text extraction complete ===")
print(f"Content saved to {content_file}")