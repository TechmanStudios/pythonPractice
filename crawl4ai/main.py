import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

START_URL = "https://nesialibraryproject.wordpress.com/"
DEPTH_LIMIT = 6
OUTPUT_FILE = "site_urls.txt"
CONTENT_URLS_FILE = "site_content_urls.txt"
OUTPUT_MD = "nesia_content.md"


def is_internal(url, base_netloc):
    parsed = urlparse(url)
    return (not parsed.netloc or parsed.netloc == base_netloc) and parsed.scheme in ("http", "https", "")


def crawl_urls(start_url, depth_limit=6):
    visited = set()
    to_visit = [(start_url, 0)]
    found_urls = []
    base_netloc = urlparse(start_url).netloc

    while to_visit:
        url, depth = to_visit.pop(0)
        if url in visited or depth > depth_limit:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            found_urls.append(url)
            if depth < depth_limit:
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a['href'])
                    if is_internal(link, base_netloc) and link not in visited:
                        to_visit.append((link, depth + 1))
            time.sleep(0.5)  # Be polite
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
    return found_urls


def filter_content_urls(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin:
        urls = [line.strip() for line in fin if line.strip()]
    # Keep only URLs with '/#content' or '/akashic-definitions/.../#content'
    filtered = [u for u in urls if ("#content" in u and not "?share=facebook" in u)]
    with open(output_file, "w", encoding="utf-8") as fout:
        for url in filtered:
            fout.write(url + "\n")
    print(f"Filtered {len(filtered)} content URLs written to {output_file}")


def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")
    # Try to extract the main content area (WordPress: 'entry-content' or 'post-content')
    main = soup.find(class_="entry-content") or soup.find(class_="post-content")
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


def fetch_and_save_markdown(urls_file, output_md):
    with open(urls_file, "r", encoding="utf-8") as fin:
        urls = [line.strip() for line in fin if line.strip()]
    with open(output_md, "w", encoding="utf-8") as fout:
        for i, url in enumerate(urls, 1):
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                text = extract_main_content(resp.text)
                fout.write(f"\n\n---\n\n# Source: {url}\n\n")
                fout.write(text)
                fout.write("\n")
                print(f"[{i}/{len(urls)}] Extracted: {url}")
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to extract {url}: {e}")
    print(f"All content saved to {output_md}")


def clean_markdown(input_md, output_md):
    import re
    with open(input_md, "r", encoding="utf-8") as fin:
        content = fin.read()
    # Split into sections by ---
    sections = content.split('\n---\n')
    cleaned = []
    for section in sections:
        if section.strip() == "":
            continue
        # Find the source url
        match = re.search(r'# Source: (.+)', section)
        if match:
            url = match.group(1)
            if "/?" in url:
                continue  # skip this section
        # Remove social/share blocks at the end of the section
        lines = section.strip().splitlines()
        # Remove trailing lines matching social/share patterns
        social_patterns = [
            r'^Facebook$', r'^Like$', r'^Loading\.\.\.$', r'^\(Opens in new window\)$',
            r'^Share this:', r'^Click to share on', r'^X$', r'^Email$', r'^Twitter$', r'^Pinterest$', r'^Reddit$', r'^LinkedIn$', r'^Tumblr$', r'^Pocket$', r'^Print$', r'^Telegram$', r'^WhatsApp$', r'^Skype$', r'^Messenger$', r'^SMS$', r'^More$', r'^\s*$'
        ]
        # Remove blocks of these lines from the end
        while lines and any(re.match(pat, lines[-1].strip()) for pat in social_patterns):
            lines.pop()
        # Remove any remaining 'Share this:' and following social/share lines anywhere in the section
        new_lines = []
        skip = False
        for line in lines:
            if re.match(r'^Share this:', line.strip()):
                skip = True
                continue
            if skip and any(re.match(pat, line.strip()) for pat in social_patterns):
                continue
            if skip and not any(re.match(pat, line.strip()) for pat in social_patterns):
                skip = False
            if not skip:
                new_lines.append(line)
        cleaned.append('\n'.join(new_lines).strip())
    with open(output_md, "w", encoding="utf-8") as fout:
        fout.write('\n\n---\n\n'.join(cleaned))
    print(f"Cleaned markdown written to {output_md}")


def main():
    print(f"Crawling {START_URL} up to depth {DEPTH_LIMIT}...")
    urls = crawl_urls(START_URL, DEPTH_LIMIT)
    print(f"Found {len(urls)} unique URLs. Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")
    print("Done.")
    filter_content_urls("site_urls.txt", "site_content_urls.txt")
    fetch_and_save_markdown(CONTENT_URLS_FILE, OUTPUT_MD)
    clean_markdown("nesia_content.md", "nesia_content_cleaned.md")


if __name__ == "__main__":
    main()
