# SpiritMythos Web Scraper

This project scrapes and archives relevant text from http://www.spiritmythos.org/.

## Features
- Recursively crawls the site, following only internal links
- Filters out ads, social links, and irrelevant content
- Saves cleaned text to spiritmythos_content.txt

## Usage
1. Install dependencies:
   pip install -r requirements.txt
2. Run the scraper:
   python scraper.py

## Notes
- Customize filtering by modifying the `clean_text` and `is_relevant_link` helper functions referenced in `scraper.py` and `crawler.py`.
- Be respectful: the script includes a delay between requests.
