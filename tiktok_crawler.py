import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Save directory
SAVE_DIR = "C:\\pythonPractice\\extractedArticles"
os.makedirs(SAVE_DIR, exist_ok=True)

# Base URL for TikTok Creator Academy
BASE_URL = "https://www.tiktok.com/creator-academy/en"

def fetch_dynamic_html(url):
    """Uses Selenium to scroll through the page and load all articles."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    driver.get(url)
    wait = WebDriverWait(driver, 15)

    try:
        # Wait for articles to load
        wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'box-border')]")))

        # Scroll to load all articles
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        
        while scroll_attempts < 10:  # Scroll up to 10 times
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)  # Allow time for articles to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # Stop if no new content loads
            last_height = new_height
            scroll_attempts += 1

        # Save HTML for debugging
        html = driver.page_source
        with open(os.path.join(SAVE_DIR, "debug_tiktok.html"), "w", encoding="utf-8") as f:
            f.write(html)

    except Exception as e:
        print(f"⚠️ Error loading page: {e}")
        html = None

    driver.quit()
    return html

def extract_article_links(html):
    """Extracts article URLs from the dynamically loaded page."""
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = set()

    # Find all article divs
    for div in soup.find_all("div", class_="box-border"):
        a_tag = div.find("a", href=True)
        if a_tag and "/article/" in a_tag["href"]:
            full_url = "https://www.tiktok.com" + a_tag["href"]
            links.add(full_url)

    return list(links)

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
    """Crawls TikTok Creator Academy and extracts all article URLs."""
    print(f"🔍 Crawling: {BASE_URL}")

    # Fetch homepage HTML dynamically
    html = fetch_dynamic_html(BASE_URL)

    # Extract article links
    article_links = extract_article_links(html)

    if not article_links:
        print("❌ No articles found!")
        return

    # Save the article URLs
    save_article_links(article_links)

# Run the crawler
crawl_tiktok_articles()
