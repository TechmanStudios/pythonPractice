import os
import time
import re  # for sanitizing file names
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

def sanitize_filename(filename):
    """
    Sanitize the filename by:
      - Replacing non-breaking spaces with regular spaces.
      - Removing characters that are not allowed on Windows.
      - Optionally replacing spaces with underscores.
    """
    # Replace non-breaking space with a regular space
    filename = filename.replace(u'\xa0', ' ')
    # Remove invalid characters: \ / : * ? " < > |
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces with underscores for consistency
    filename = filename.replace(" ", "_")
    return filename

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
    article_div = soup.find("div", id="academyArticleMainBody")
    if not article_div:
        return None

    formatted_text = []
    seen_paragraphs = set()
    current_paragraph = []

    for element in article_div.find_all(["div", "span"]):
        element_id = element.get("id", "")
        text = element.get_text(strip=True)
        if not text:
            continue

        # h1 headers
        if element_id.startswith("title_") and "_" not in element_id:
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph).strip()
                if paragraph_text not in seen_paragraphs:
                    formatted_text.append(paragraph_text + "\n")
                    seen_paragraphs.add(paragraph_text)
                current_paragraph = []
            formatted_text.append(f"\n# {text}\n")
        # h2 headers
        elif element_id.startswith("title_") and element_id.count("_") == 1:
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph).strip()
                if paragraph_text not in seen_paragraphs:
                    formatted_text.append(paragraph_text + "\n")
                    seen_paragraphs.add(paragraph_text)
                current_paragraph = []
            formatted_text.append(f"\n## {text}\n")
        # h3 headers
        elif element_id.startswith("title_") and element_id.count("_") == 2:
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph).strip()
                if paragraph_text not in seen_paragraphs:
                    formatted_text.append(paragraph_text + "\n")
                    seen_paragraphs.add(paragraph_text)
                current_paragraph = []
            formatted_text.append(f"\n### {text}\n")
        # Normal paragraph text
        elif element.name == "span":
            current_paragraph.append(text)

    if current_paragraph:
        paragraph_text = " ".join(current_paragraph).strip()
        if paragraph_text not in seen_paragraphs:
            formatted_text.append(paragraph_text + "\n")
            seen_paragraphs.add(paragraph_text)

    return "\n".join(formatted_text)

def extract_tiktok_article(url):
    """Extracts a TikTok Creator Academy article and returns its Markdown content."""
    print(f"🔍 Fetching page: {url}")
    html = fetch_html_with_selenium(url)
    content = extract_content_with_bs4(html)
    if not content:
        print(f"❌ No content extracted for {url}")
        return None

    soup = BeautifulSoup(html, "html.parser")
    title_div = soup.find("div", id="title_0")
    title = title_div.get_text(strip=True) if title_div else "Untitled Article"

    # Sanitize title to create a safe file name
    safe_title = sanitize_filename(title)
    markdown_content = f"# {title}\n\n" + content

    # Save individual article as a Markdown file
    filename = os.path.join(SAVE_DIR, f"{safe_title}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"✅ Article saved as Markdown: {filename}")
    return markdown_content

# List of TikTok Creator Academy article URLs
urls = [
    "https://www.tiktok.com/creator-academy/en/article/TikTok-101",
    "https://www.tiktok.com/creator-academy/en/article/get-started-create-an-account",
    "https://www.tiktok.com/creator-academy/en/article/8-tips-for-becoming-a-successful-TikTok-creator",
    "https://www.tiktok.com/creator-academy/en/article/Finding-your-vibe-and-community",
    "https://www.tiktok.com/creator-academy/en/article/how-to-personalize-profile",
    "https://www.tiktok.com/creator-academy/en/article/Account-verification-101",
    "https://www.tiktok.com/creator-academy/en/article/TikTok-Business-vs-Personal",
    "https://www.tiktok.com/creator-academy/en/article/account-check"
    "https://www.tiktok.com/creator-academy/en/article/safety-tips",
    "https://www.tiktok.com/creator-academy/en/article/community-guidelines-overview",
	"https://www.tiktok.com/creator-academy/en/article/creator-code-of-conduct",
	"https://www.tiktok.com/creator-academy/en/article/Originality-Policy",
    "https://www.tiktok.com/creator-academy/en/article/guidelines-recommendation-system-intro",
	"https://www.tiktok.com/creator-academy/en/article/guidelines-moderation-status-and-appeals",
	"https://www.tiktok.com/creator-academy/en/article/video-not-recommended",   
    "https://www.tiktok.com/creator-academy/en/article/ai-generated-content-label",
	"https://www.tiktok.com/creator-academy/en/article/supporting-your-mental-health",
	"https://www.tiktok.com/creator-academy/en/article/security-FAQs",
	"https://www.tiktok.com/creator-academy/en/article/Political-ads",
    "https://www.tiktok.com/creator-academy/en/article/tool-one-minute-feature-overview",
	"https://www.tiktok.com/creator-academy/en/article/tool-full-screen-intro",
	"https://www.tiktok.com/creator-academy/en/article/tool-web-creation-intro",
	"https://www.tiktok.com/creator-academy/en/article/tool-playlists-intro",
	"https://www.tiktok.com/creator-academy/en/article/effect-house-overview",
	"https://www.tiktok.com/creator-academy/en/article/tiktok-studio",
	"https://www.tiktok.com/creator-academy/en/article/unleashing-your-creativity-with-tiktok-studio",
	"https://www.tiktok.com/creator-academy/en/article/symphony",
	"https://www.tiktok.com/creator-academy/en/article/finding-creator-search-insights",
    "https://www.tiktok.com/creator-academy/en/article/tool-analytics-intro",
	"https://www.tiktok.com/creator-academy/en/article/search",
	"https://www.tiktok.com/creator-academy/en/article/Creator-Search-Insights",
	"https://www.tiktok.com/creator-academy/en/article/Why-Search-Analytics-Matter",
	"https://www.tiktok.com/creator-academy/en/article/Getting-discovered-with-Search",
    "https://www.tiktok.com/creator-academy/en/article/editing-like-a-pro",
	"https://www.tiktok.com/creator-academy/en/article/the-importance-of-sound",
	"https://www.tiktok.com/creator-academy/en/article/Streamline-your-ticket-sales-with-Eventbrite",
    "https://www.tiktok.com/creator-academy/en/article/an-introduction-to-promote",
	"https://www.tiktok.com/creator-academy/en/article/Setting-up-your-Promote-campaign",
	"https://www.tiktok.com/creator-academy/en/article/Top-promote-tips-and-tricks",
    "https://www.tiktok.com/creator-academy/en/article/donation-feature",
	"https://www.tiktok.com/creator-academy/en/article/flip-story",
    "https://www.tiktok.com/creator-academy/en/article/Creating-high-quality-videos-on-TikTok",
	"https://www.tiktok.com/creator-academy/en/article/8-tips-for-becoming-a-successful-TikTok-creator",
	"https://www.tiktok.com/creator-academy/en/article/captivating-content-with-great-storytelling",
	"https://www.tiktok.com/creator-academy/en/article/Showcasing-your-expertise-with-specialized-content",
	"https://www.tiktok.com/creator-academy/en/article/elements-of-tiktok-video",
	"https://www.tiktok.com/creator-academy/en/article/Understanding-and-engaging-with-your-audience",
	"https://www.tiktok.com/creator-academy/en/article/Filming-High-Quality-Videos",
	"https://www.tiktok.com/creator-academy/en/article/longer-video-content-strategy",
	"https://www.tiktok.com/creator-academy/en/article/Filming-101",
	"https://www.tiktok.com/creator-academy/en/article/Filming-102",
	"https://www.tiktok.com/creator-academy/en/article/building-a-thriving-community",
	"https://www.tiktok.com/creator-academy/en/article/getting-creative-with-color",
	"https://www.tiktok.com/creator-academy/en/article/capcut-special-effects",
	"https://www.tiktok.com/creator-academy/en/article/audio-101",
	"https://www.tiktok.com/creator-academy/en/article/lighting-101",
    "https://www.tiktok.com/creator-academy/en/article/getting-started-on-CapCut",
	"https://www.tiktok.com/creator-academy/en/article/Creating-engaging-content-with-these-CapCut-tricks",
    "https://www.tiktok.com/creator-academy/en/article/introducing-fan-content",
	"https://www.tiktok.com/creator-academy/en/article/Master-TikTok-most-popular-fan-content",
	"https://www.tiktok.com/creator-academy/en/article/Elevate-your-Commentary-content-with-these-4-tips",
	"https://www.tiktok.com/creator-academy/en/article/Become-an-Edits-expert-with-these-3-tips",
	"https://www.tiktok.com/creator-academy/en/article/Create-unforgettable-reaction-content",
	"https://www.tiktok.com/creator-academy/en/article/entertainment-news-content",
    "https://www.tiktok.com/creator-academy/en/article/developing-your-content-strategy",
	"https://www.tiktok.com/creator-academy/en/article/making-your-videos-more-accessible",
    "https://www.tiktok.com/creator-academy/en/article/get-started-as-an-artist-on-TT",
	"https://www.tiktok.com/creator-academy/en/article/TT-billboard-music-charts",
    "https://www.tiktok.com/creator-academy/en/article/Gaming-on-TikTok-an-overview",
	"https://www.tiktok.com/creator-academy/en/article/Gaming-content-strategy",
	"https://www.tiktok.com/creator-academy/en/article/Gaming-personas",
	"https://www.tiktok.com/creator-academy/en/article/Mental-Health",
	"https://www.tiktok.com/creator-academy/en/article/what-is-STEM-feed",
	"https://www.tiktok.com/creator-academy/en/article/creating-STEM-content",
    "https://www.tiktok.com/creator-academy/en/article/monetization-offerings-overview",
	"https://www.tiktok.com/creator-academy/en/article/Creator-Monetization-Center",
    "https://www.tiktok.com/creator-academy/en/article/creator-rewards-program",
	"https://www.tiktok.com/creator-academy/en/article/effect-creator-rewards",  
    "https://www.tiktok.com/creator-academy/en/article/Subscription",
    "https://www.tiktok.com/creator-academy/en/article/TTCM-introduction",		
	"https://www.tiktok.com/creator-academy/en/article/create-branded-content",
    "https://www.tiktok.com/creator-academy/en/article/Going-LIVE",
	"https://www.tiktok.com/creator-academy/en/article/Unlocking-LIVE-monetization",
    "https://www.tiktok.com/creator-academy/en/article/monetization-shop-overview",
    "https://www.tiktok.com/creator-academy/en/article/creator-rewards-program",
    "https://www.tiktok.com/creator-academy/en/article/eligibility",
	"https://www.tiktok.com/creator-academy/en/article/apply",
    "https://www.tiktok.com/creator-academy/en/article/RPM-understanding-the-four-key-factors",
	"https://www.tiktok.com/creator-academy/en/article/Creator-Rewards-Program-New-RPM-Dashboard",
	"https://www.tiktok.com/creator-academy/en/article/Originality-Policy",
	"https://www.tiktok.com/creator-academy/en/article/longer-video-content-strategy",
	"https://www.tiktok.com/creator-academy/en/article/Creator-Search-Insights",
	"https://www.tiktok.com/creator-academy/en/article/Understanding-and-engaging-with-your-audience",
    "https://www.tiktok.com/creator-academy/en/article/monetization-creativity-program-video-eligible",
	"https://www.tiktok.com/creator-academy/en/article/monetization-creativity-program-qualified-view",
	"https://www.tiktok.com/creator-academy/en/article/monetization-creativity-program-affect",
	"https://www.tiktok.com/creator-academy/en/article/creator-rewards-program-faq-2",
    "https://www.tiktok.com/creator-academy/en/article/Subscription",
	"https://www.tiktok.com/creator-academy/en/article/Setting-up-your-Subscription",
    "https://www.tiktok.com/creator-academy/en/article/Subscription-success-checklist",
	"https://www.tiktok.com/creator-academy/en/article/Subscription-success-sub-only-videos",
	"https://www.tiktok.com/creator-academy/en/article/Subscription-success-adding-links-to-video",   
    "https://www.tiktok.com/creator-academy/en/article/Why-introduce-Subscription",
	"https://www.tiktok.com/creator-academy/en/article/Subscription-vs-Series",
	"https://www.tiktok.com/creator-academy/en/article/About-earnings",
	"https://www.tiktok.com/creator-academy/en/article/Any-rules-for-subscriber-only-content"
]

compiled_articles = []

for url in urls:
    article_md = extract_tiktok_article(url)
    if article_md:
        compiled_articles.append(article_md)

print(f"📝 Total articles collected: {len(compiled_articles)}")

if compiled_articles:
    final_compiled_file = os.path.join(SAVE_DIR, "compiled_articles.md")
    final_content = "\n\n---\n\n".join(compiled_articles)
    with open(final_compiled_file, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"🚀 All articles compiled into: {final_compiled_file}")
else:
    print("⚠️ No articles were compiled.")
