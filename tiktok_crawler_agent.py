import os
import time
from browser_use import Browser
from dotenv import load_dotenv  # Load .env file

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Save directory
SAVE_DIR = "C:\\pythonPractice\\extractedArticles"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://www.tiktok.com/creator-academy/en"

def extract_article_links():
    """Uses Browser-Use AI Agent to navigate and extract article URLs."""

    browser = Browser()  # Initialize browser object
    browser.start()  # Start the AI browser

    try:
        browser.navigate(BASE_URL)  # Use 'navigate()' instead of 'open()'
        time.sleep(5)  # Wait for page to load

        # Scroll multiple times to load articles
        for _ in range(6):  
            browser.scroll("down")
            browser.wait(2)  # Wait for new content to load

        # Extract all article links
        links = browser.get_links()
        article_links = set()

        for link in links:
            href = link.get("href")
            if href and "/article/" in href:
                full_url = f"https://www.tiktok.com{href}" if not href.startswith("http") else href
                article_links.add(full_url)

                # Open article in a new tab, extract its content, and close it
                browser.navigate(full_url)  # Use 'navigate()' instead of 'open()'
                browser.wait(3)
                browser.close()

    except Exception as e:
        print(f"❌ Error during crawling: {e}")

    finally:
        browser.stop()  # Ensure browser stops even if an error occurs

    return list(article_links)

def save_article_links(links):
    """Saves extracted article links to a text file."""
    if not links:
        print("❌ No articles found!")
        return

    filename = os.path.join(SAVE_DIR, "tiktok_article_urls.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")

    print(f"✅ Saved {len(links)} article URLs to {filename}")

def crawl_tiktok_articles():
    """Runs the AI browser to scrape TikTok Creator Academy articles."""
    print(f"🔍 Using AI Agent to crawl: {BASE_URL}")
    print(f"🔑 OpenAI API Key Loaded: {OPENAI_API_KEY}")

    article_links = extract_article_links()

    if not article_links:
        print("❌ No articles found!")
        return

    save_article_links(article_links)

# Run the AI browsing agent
crawl_tiktok_articles()
