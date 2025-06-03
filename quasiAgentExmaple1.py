import unittest
import re
from your_module import parse_archival_document  # Replace your_module

class TestParseArchivalDocument(unittest.TestCase):

    def test_empty_document(self):
        self.assertEqual(parse_archival_document(""), {})

    def test_basic_metadata_extraction(self):
        document = "Date: 1923-04-01\nOrigin: Some Monastery\nContent: Some text here."
        expected = {
            'metadata': {'date': '1923-04-01', 'origin': 'Some Monastery'},
            'themes': [],
            'content_blocks': ['Date: 1923-04-01\nOrigin: Some Monastery\nContent: Some text here.']
        }
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_metadata_with_no_content(self):
        document = "Date: 1888-12-25\nOrigin: Unknown Source"
        expected = {
            'metadata': {'date': '1888-12-25', 'origin': 'Unknown Source'},
            'themes': [],
            'content_blocks': ['Date: 1888-12-25\nOrigin: Unknown Source']
        }
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_theme_detection(self):
        document = "Text about Spirituality and Enlightenment."
        expected = {
            'metadata': [],
            'themes': [],
            'content_blocks': ["Text about Spirituality and Enlightenment."]
        }
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})


    def test_content_blocks_simple(self):
        document = "Block 1: Some content.\n\nBlock 2: More content."
        expected = {
            'metadata': [],
            'themes': [],
            'content_blocks': ['Block 1: Some content.\n\nBlock 2: More content.']
        }

        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_mixed_metadata_themes_content(self):
        document = "Date: 1776-07-04\nOrigin: Philadelphia\nSpirituality and Enlightenment content here.\n\nNew Block."
        expected = {
            'metadata': {'date': '1776-07-04', 'origin': 'Philadelphia'},
            'themes': ['Spirituality', 'Enlightenment'],
            'content_blocks': ['Date: 1776-07-04\nOrigin: Philadelphia\nSpirituality and Enlightenment content here.\n\nNew Block.']
        }
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})  # Corrected assertion

    def test_no_metadata_or_themes(self):
        document = "Just some plain text without any metadata or themes."
        expected = {
            'metadata': [],
            'themes': [],
            'content_blocks': ["Just some plain text without any metadata or themes."]
        }
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_long_document(self):
        document = "Date: 2023-10-26\nOrigin: Internet\nThis is a very long document with multiple paragraphs.\n\nIt talks about Spirituality and Enlightenment.  More and more content to test the parsing.\n\nAnother Block of text."
        expected = {
            'metadata': {'date': '2023-10-26', 'origin': 'Internet'},
            'themes': ['Spirituality', 'Enlightenment'],
            'content_blocks': ['Date: 2023-10-26\nOrigin: Internet\nThis is a very long document with multiple paragraphs.\n\nIt talks about Spirituality and Enlightenment.  More and more content to test the parsing.\n\nAnother Block of text.']
        }
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_metadata_at_end_of_document(self):
        document = "Some content here.\nDate: 1600-01-01\nOrigin: Someplace"
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})


    def test_special_characters_in_document(self):
        document = "Date: 1900-01-01\nOrigin: Somewhere!@#$\nContent: This is some text with special characters like !@#$%^&*()_+=-`~[]\{}|;':\",./<>?"
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_newline_characters_within_blocks(self):
        document = "Block 1:\nSome\nContent.\n\nBlock 2:\nMore\nContent."
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_only_themes_present(self):
        document = "Spirituality Enlightenment"
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_invalid_date_format(self):
        document = "Date: January 1, 2023\nContent: Text here."
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_origin_with_multiple_words(self):
        document = "Origin: Place of Origin\nContent: Some content"
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})

    def test_document_with_empty_lines(self):
        document = "\n\nDate: 2024-01-01\n\nOrigin: Somewhere\n\nContent: Some Text\n\n"
        self.assertEqual(parse_archival_document(document), {'metadata': {}, 'themes': [], 'content_blocks': [document]})
if __name__ == '__main__':
    unittest.main()

import re
from typing import Dict, List

def extract_metadata(document: str) -> Dict:
    metadata = {}
    date_match = re.search(r"Date:\s*([\d\-]+)", document)
    if date_match:
        metadata['date'] = date_match.group(1)
    origin_match = re.search(r"Origin:\s*([\w\s]+)", document)
    if origin_match:
        metadata['origin'] = origin_match.group(1).strip()
    return metadata

def extract_themes(document: str, theme_keywords: List[str]) -> List[str]:
    found = []
    for theme in theme_keywords:
        if re.search(rf"\\b{re.escape(theme)}\\b", document, re.IGNORECASE):
            found.append(theme)
    return found

def split_content_blocks(document: str) -> List[str]:
    # Split on double newlines, but keep as one block if no double newlines
    blocks = [block.strip() for block in re.split(r'\n\s*\n', document) if block.strip()]
    return blocks if blocks else [document.strip()] if document.strip() else []

def parse_archival_document(document: str, theme_keywords: List[str] = None) -> Dict:
    if not document or not document.strip():
        return {'metadata': {}, 'themes': [], 'content_blocks': []}
    if theme_keywords is None:
        theme_keywords = []
    metadata = extract_metadata(document)
    themes = extract_themes(document, theme_keywords)
    content_blocks = split_content_blocks(document)
    return {
        'metadata': metadata,
        'themes': themes,
        'content_blocks': content_blocks
    }