# Sacred Texts Ebook Scraper

This project is a Python-based scraper that downloads .txt.gz ebooks from the [Sacred Texts download page](https://sacred-texts.com/download.htm).

## Features
- Scrapes the numbered ebook list from the downloads page
- Downloads each ebook as a .txt.gz file
- (Optional) Decompresses and saves the text files
- Saves metadata for each ebook

## Requirements
- Python 3.8+
- `requests`, `beautifulsoup4`

## Usage
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the scraper:
   ```powershell
   python scraper.py
   ```

Downloaded ebooks will be saved in the `downloads/` directory.
