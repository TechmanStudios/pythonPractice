import os
from urllib.parse import urljoin

SITE_COPY_DIR = os.path.join(os.path.dirname(__file__), "site_copy")
OUTPUT_MD = os.path.join(os.path.dirname(__file__), "site_urls.md")


def scan_site_copy_for_urls(site_dir):
    urls = set()
    for root, dirs, files in os.walk(site_dir):
        for file in files:
            if file.endswith(".html"):
                rel_path = os.path.relpath(os.path.join(root, file), site_dir)
                # Convert Windows backslashes to URL slashes
                url_path = rel_path.replace(os.sep, "/")
                # Remove trailing /index.html for clean URLs
                if url_path.endswith("/index.html"):
                    url_path = url_path[:-10]
                elif url_path == "index.html":
                    url_path = ""
                url = urljoin("https://christmatrix.com/", url_path)
                urls.add(url)
    return sorted(urls)


def save_urls_to_md(urls, output_md):
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Local Copy: ChristMatrix URLs\n\n")
        for url in urls:
            f.write(f"- {url}\n")
    print(f"Saved {len(urls)} URLs to {output_md}")


if __name__ == "__main__":
    urls = scan_site_copy_for_urls(SITE_COPY_DIR)
    save_urls_to_md(urls, OUTPUT_MD)
