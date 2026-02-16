import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Postlight Parser API Key (replace with your actual key)
POSTLIGHT_API_KEY = "YOUR_API_KEY"
POSTLIGHT_API_URL = "https://mercury.postlight.com/parser"

# Save directory for Markdown files
SAVE_DIR = "C:\\pythonPractice\\extractedArticles"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_html_with_selenium(url):
    """Fetches fully rendered HTML using Selenium for JavaScript-heavy pages."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")

    # Automatically download and use the correct ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Load page
    driver.get(url)
    time.sleep(3)  # Wait for JavaScript to load

    # Get page source
    html = driver.page_source
    driver.quit()
    return html

def extract_content_with_bs4(html):
    """Extracts article content manually using BeautifulSoup with Markdown formatting and no duplicates."""
    soup = BeautifulSoup(html, "html.parser")

    # Find the main article content
    article_div = soup.find("div", id="academyArticleMainBody")

    if not article_div:
        return None  # If there's no content, return None

    formatted_text = []
    seen_paragraphs = set()  # To track unique paragraphs
    current_paragraph = []  # To store and join paragraph chunks properly

    # Extract headings and paragraphs in order
    for element in article_div.find_all(["div", "span"]):
        element_id = element.get("id", "")
        text = element.get_text(strip=True)

        if not text:
            continue  # Skip empty text elements

        # Detect `h1` headers
        if element_id.startswith("title_") and "_" not in element_id:
            # Save the last paragraph before a new heading starts
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph).strip()
                if paragraph_text not in seen_paragraphs:
                    formatted_text.append(paragraph_text + "\n")
                    seen_paragraphs.add(paragraph_text)
                current_paragraph = []
            formatted_text.append(f"\n# {text}\n")

        # Detect `h2` headers
        elif element_id.startswith("title_") and element_id.count("_") == 1:
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph).strip()
                if paragraph_text not in seen_paragraphs:
                    formatted_text.append(paragraph_text + "\n")
                    seen_paragraphs.add(paragraph_text)
                current_paragraph = []
            formatted_text.append(f"\n## {text}\n")

        # Detect `h3` headers
        elif element_id.startswith("title_") and element_id.count("_") == 2:
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph).strip()
                if paragraph_text not in seen_paragraphs:
                    formatted_text.append(paragraph_text + "\n")
                    seen_paragraphs.add(paragraph_text)
                current_paragraph = []
            formatted_text.append(f"\n### {text}\n")

        # Detect normal paragraph text (inside <span> tags)
        elif element.name == "span":
            current_paragraph.append(text)

    # Add the last paragraph if there is one
    if current_paragraph:
        paragraph_text = " ".join(current_paragraph).strip()
        if paragraph_text not in seen_paragraphs:
            formatted_text.append(paragraph_text + "\n")
            seen_paragraphs.add(paragraph_text)

    return "\n".join(formatted_text)  # Join the extracted text with proper Markdown formatting

def extract_tiktok_article(url):
    """Extracts a TikTok Creator Academy article and saves as Markdown."""
    print(f"🔍 Fetching page: {url}")

    # Fetch full HTML
    html = fetch_html_with_selenium(url)

    # Extract formatted content
    content = extract_content_with_bs4(html)

    if not content:
        print(f"❌ No content extracted for {url}")
        return

    # Extract title (first h1 heading)
    soup = BeautifulSoup(html, "html.parser")
    title_div = soup.find("div", id="title_0")  # First h1 heading
    title = title_div.get_text(strip=True) if title_div else "Untitled Article"

    # Clean filename
    safe_title = title.replace(" ", "_").replace("/", "").replace("\\", "")

    # Save to Markdown file
    filename = os.path.join(SAVE_DIR, f"{safe_title}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")  # Add main title as first heading
        f.write(content)

    print(f"✅ Article saved as Markdown: {filename}")

# List of TikTok Creator Academy article URLs
urls = [
        "https://www.tiktok.com/creator-academy/en/article/TikTok-101",
		"https://www.tiktok.com/creator-academy/en/article/get-started-create-an-account",
		"https://www.tiktok.com/creator-academy/en/article/8-tips-for-becoming-a-successful-TikTok-creator",
		"https://www.tiktok.com/creator-academy/en/article/Finding-your-vibe-and-community"
]

# Run the extractor for each URL
for url in urls:
    extract_tiktok_article(url)
