import os
import json
from google import genai
from typing import List, Dict
import unittest
import re
from collections import Counter
import spacy
from transformers import pipeline
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Load NLP models once
spacy_nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# The function to be tested (copied from the problem description)
def basic_parse_esoteric_document(document_text: str, spiritual_keywords: list[str]) -> dict:
    """Performs a basic analysis of an esoteric document text.

    This function processes a given document to extract placeholder metadata,
    identify overall thematic keywords, and segment the content into distinct
    blocks (based on paragraphs). Each block is then analyzed for the presence
    and frequency of specified spiritual keywords.

    The parsing is "basic" as it relies on simple string matching for keywords,
    paragraph splitting for content structure, and rudimentary title extraction.
    It does not employ advanced NLP techniques for deeper semantic understanding.

    Args:
        document_text (str): The raw text content of the esoteric archival
            document. This is expected to be a single string, potentially
            containing multiple paragraphs separated by double newlines.
        spiritual_keywords (list[str]): A list of keywords to search for within
            the document. These keywords are considered relevant to the
            spiritual or thematic nature of the content. The matching is
            case-insensitive, but the original casing of keywords is preserved
            in the output.

    Returns:
        dict: A structured dictionary containing the analysis results.
            The dictionary has the following main keys:
            - "metadata" (dict): Contains metadata about the document.
                - "title" (str): The extracted title if found (e.g., "Title: Actual Title"),
                  otherwise "Unknown Title (Placeholder)".
                - "author" (str): Placeholder value "Unknown Author (Placeholder)".
                - "date_extracted" (str): Placeholder value "YYYY-MM-DD (Placeholder)".
                - "source_type" (str): Fixed value "esoteric_archival_document".
                - "processing_level" (str): Fixed value "basic_keyword_extraction".
            - "overall_themes_identified" (list[str]): A sorted list of unique
              spiritual keywords (from the input list) found anywhere in the
              document_text. Case of keywords matches the input `spiritual_keywords`.
            - "content_structure" (dict): Contains the block-level analysis.
                - "total_blocks" (int): The total number of content blocks identified
                  (typically paragraphs).
                - "blocks" (list[dict]): A list where each dictionary represents a
                  content block. Each block dictionary contains:
                    - "block_id" (str): A unique identifier for the block (e.g., "block_1").
                    - "content" (str): The raw text content of the block.
                    - "themes_present" (list[str]): A sorted list of unique spiritual
                      keywords found within this specific block. Case of keywords
                      matches the input `spiritual_keywords`.
                    - "semantic_focus_guess" (str): A string representing the most
                      frequent keyword(s) in the block. If multiple keywords share
                      the highest frequency, they are all listed, sorted alphabetically,
                      and comma-separated. If no keywords are found in the block,
                      this defaults to "General Content".
                    - "character_count" (int): The number of characters in the
                      block's content.

    Example Usage:
        >>> doc_text = ("Title: The Journey Within\\n\\n"
        ...             "The soul seeks enlightenment. Awakening is near.\\n\\n"
        ...             "Meditation brings peace and deeper consciousness. The soul is eternal.")
        >>> keywords = ["Soul", "Enlightenment", "Awakening", "Meditation", "Peace"]
        >>> parsed_doc = basic_parse_esoteric_document(doc_text, keywords)
        >>> print(parsed_doc['metadata']['title'])
        The Journey Within
        >>> print(parsed_doc['overall_themes_identified'])
        ['Awakening', 'Enlightenment', 'Meditation', 'Peace', 'Soul']
        >>> print(parsed_doc['content_structure']['total_blocks'])
        3
        >>> first_block = parsed_doc['content_structure']['blocks'][0]
        >>> print(first_block['content'])
        Title: The Journey Within
        >>> print(first_block['themes_present']) # Title line itself as a block
        []
        >>> second_block = parsed_doc['content_structure']['blocks'][1]
        >>> print(second_block['content'])
        The soul seeks enlightenment. Awakening is near.
        >>> print(second_block['themes_present'])
        ['Awakening', 'Enlightenment', 'Soul']
        >>> print(second_block['semantic_focus_guess'])
        Awakening, Enlightenment, Soul
        >>> third_block = parsed_doc['content_structure']['blocks'][2]
        >>> print(third_block['themes_present'])
        ['Meditation', 'Peace', 'Soul']
        >>> print(third_block['semantic_focus_guess']) # All appear once
        Meditation, Peace, Soul

    Edge Cases:
        - Empty `document_text`:
            - Metadata will contain placeholders (title might be "Unknown Title").
            - `overall_themes_identified` will be an empty list.
            - `content_structure` will have `total_blocks: 0` and `blocks: []`.
        - Empty `spiritual_keywords` list:
            - `overall_themes_identified` will be an empty list.
            - No themes will be identified in blocks; `themes_present` will be empty
              for all blocks.
            - `semantic_focus_guess` for all blocks will be "General Content".
        - No keywords found in the document:
            - `overall_themes_identified` will be an empty list.
            - `themes_present` for all blocks will be empty.
            - `semantic_focus_guess` for all blocks will be "General Content".
        - Document with no double newlines (effectively a single paragraph):
            - If `document_text` is not empty, `total_blocks` will be 1.
            - The entire `document_text` (stripped) will be the content of the
              single block.
        - Document containing only whitespace:
            - `total_blocks` will be 0, `blocks` will be empty.
            - `overall_themes_identified` will be empty.
        - Keywords with special regex characters (e.g., "*", "+", "?"):
            - Handled safely due to `re.escape()`, ensuring they are treated as
              literal characters.
        - Case sensitivity of keywords:
            - Keyword matching is case-insensitive. E.g., if "soul" is in `spiritual_keywords`,
              it will match "Soul", "soul", "SOUL" in the text.
            - The keywords returned in `overall_themes_identified` and
              `themes_present` will retain the original casing as provided in
              the `spiritual_keywords` list.
    """

    # --- 1. Placeholder for Metadata Extraction ---
    metadata = {
        "title": "Unknown Title (Placeholder)",
        "author": "Unknown Author (Placeholder)",
        "date_extracted": "YYYY-MM-DD (Placeholder)", 
        "source_type": "esoteric_archival_document",
        "processing_level": "basic_keyword_extraction"
    }
    title_match = re.search(r"Title:\s*(.+)", document_text, re.IGNORECASE)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    # --- 2. Tag Overall Themes for the Entire Document ---
    document_lower = document_text.lower()
    overall_themes = []
    for keyword in spiritual_keywords:
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', document_lower):
            overall_themes.append(keyword) 
    overall_themes = sorted(list(set(overall_themes)))


    # --- 3. Structure Content into Semantically Organized Blocks ---
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', document_text) if p.strip()]
    
    structured_blocks = []
    block_id_counter = 1

    for para_text in paragraphs:
        para_lower = para_text.lower()
        block_themes = []
        block_keyword_counts = Counter()

        for keyword in spiritual_keywords:
            kw_lower = keyword.lower()
            count = len(re.findall(r'\b' + re.escape(kw_lower) + r'\b', para_lower))
            if count > 0:
                block_themes.append(keyword) 
                block_keyword_counts[keyword] += count 
        
        block_themes = sorted(list(set(block_themes))) 

        semantic_focus_guess = "General Content"
        if block_keyword_counts:
            max_count = 0
            # Check if block_keyword_counts is not empty before accessing most_common(1)[0]
            if block_keyword_counts: 
                 max_count = block_keyword_counts.most_common(1)[0][1]
            
            most_common_keywords = [
                kw for kw, count in block_keyword_counts.items() if count == max_count
            ]
            if most_common_keywords: # ensure list is not empty before join
                 semantic_focus_guess = ", ".join(sorted(most_common_keywords))
            # else: semantic_focus_guess remains "General Content" (e.g. if all counts were 0, though logic above should prevent this)


        structured_blocks.append({
            "block_id": f"block_{block_id_counter}",
            "content": para_text,
            "themes_present": block_themes,
            "semantic_focus_guess": semantic_focus_guess,
            "character_count": len(para_text)
        })
        block_id_counter += 1

    # --- 4. Assemble the Output ---
    output = {
        "metadata": metadata,
        "overall_themes_identified": overall_themes,
        "content_structure": {
            "total_blocks": len(structured_blocks),
            "blocks": structured_blocks
        }
    }

    return output


class TestBasicParseEsotericDocument(unittest.TestCase):

    def assert_metadata_defaults(self, metadata, expected_title="Unknown Title (Placeholder)"):
        self.assertEqual(metadata["title"], expected_title)
        self.assertEqual(metadata["author"], "Unknown Author (Placeholder)")
        self.assertEqual(metadata["date_extracted"], "YYYY-MM-DD (Placeholder)")
        self.assertEqual(metadata["source_type"], "esoteric_archival_document")
        self.assertEqual(metadata["processing_level"], "basic_keyword_extraction")

    def test_basic_functionality_with_title_and_multiple_keywords(self):
        doc_text = ("Title: The Grand Cosmic Plan\n\n"
                    "The universe unfolds. Energy flows freely. Stars align.\n\n"
                    "Seek wisdom, for wisdom illuminates. The path of energy is clear.")
        keywords = ["Universe", "Energy", "Wisdom", "Stars"]
        
        result = basic_parse_esoteric_document(doc_text, keywords)

        # Metadata
        self.assertEqual(result["metadata"]["title"], "The Grand Cosmic Plan")
        self.assert_metadata_defaults(result["metadata"], expected_title="The Grand Cosmic Plan")

        # Overall themes
        self.assertListEqual(result["overall_themes_identified"], ["Energy", "Stars", "Universe", "Wisdom"])

        # Content structure
        self.assertEqual(result["content_structure"]["total_blocks"], 3)
        blocks = result["content_structure"]["blocks"]

        # Block 1 (Title)
        self.assertEqual(blocks[0]["block_id"], "block_1")
        self.assertEqual(blocks[0]["content"], "Title: The Grand Cosmic Plan")
        self.assertListEqual(blocks[0]["themes_present"], [])
        self.assertEqual(blocks[0]["semantic_focus_guess"], "General Content")
        self.assertEqual(blocks[0]["character_count"], len("Title: The Grand Cosmic Plan"))

        # Block 2
        self.assertEqual(blocks[1]["block_id"], "block_2")
        self.assertEqual(blocks[1]["content"], "The universe unfolds. Energy flows freely. Stars align.")
        self.assertListEqual(blocks[1]["themes_present"], ["Energy", "Stars", "Universe"])
        self.assertEqual(blocks[1]["semantic_focus_guess"], "Energy, Stars, Universe") # All once
        self.assertEqual(blocks[1]["character_count"], len("The universe unfolds. Energy flows freely. Stars align."))

        # Block 3
        self.assertEqual(blocks[2]["block_id"], "block_3")
        self.assertEqual(blocks[2]["content"], "Seek wisdom, for wisdom illuminates. The path of energy is clear.")
        self.assertListEqual(blocks[2]["themes_present"], ["Energy", "Wisdom"]) # "wisdom" (2), "energy" (1)
        self.assertEqual(blocks[2]["semantic_focus_guess"], "Wisdom") 
        self.assertEqual(blocks[2]["character_count"], len("Seek wisdom, for wisdom illuminates. The path of energy is clear."))

    def test_docstring_example_behavior(self):
        # This test uses the extended keyword list from the __main__ example in the SUT
        doc_text = ("Title: The Journey Within\n\n"
                    "The soul seeks enlightenment. Awakening is near.\n\n"
                    "Meditation brings peace and deeper consciousness. The soul is eternal.")
        keywords = ["Soul", "Enlightenment", "Awakening", "Meditation", "Peace", "Consciousness"]
        
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assertEqual(result['metadata']['title'], "The Journey Within")
        self.assertListEqual(result['overall_themes_identified'], ['Awakening', 'Consciousness', 'Enlightenment', 'Meditation', 'Peace', 'Soul'])
        self.assertEqual(result['content_structure']['total_blocks'], 3)
        
        blocks = result['content_structure']['blocks']
        
        self.assertEqual(blocks[0]['content'], "Title: The Journey Within")
        self.assertListEqual(blocks[0]['themes_present'], [])
        self.assertEqual(blocks[0]['semantic_focus_guess'], "General Content")

        self.assertEqual(blocks[1]['content'], "The soul seeks enlightenment. Awakening is near.")
        self.assertListEqual(blocks[1]['themes_present'], ['Awakening', 'Enlightenment', 'Soul'])
        self.assertEqual(blocks[1]['semantic_focus_guess'], "Awakening, Enlightenment, Soul")

        self.assertEqual(blocks[2]['content'], "Meditation brings peace and deeper consciousness. The soul is eternal.")
        self.assertListEqual(blocks[2]['themes_present'], ['Consciousness', 'Meditation', 'Peace', 'Soul'])
        self.assertEqual(blocks[2]['semantic_focus_guess'], "Consciousness, Meditation, Peace, Soul")


    def test_empty_document_text(self):
        keywords = ["Test", "Keyword"]
        result = basic_parse_esoteric_document("", keywords)

        self.assert_metadata_defaults(result["metadata"])
        self.assertListEqual(result["overall_themes_identified"], [])
        self.assertEqual(result["content_structure"]["total_blocks"], 0)
        self.assertListEqual(result["content_structure"]["blocks"], [])

    def test_empty_spiritual_keywords(self):
        doc_text = "Some content here.\n\nMore content."
        result = basic_parse_esoteric_document(doc_text, [])

        self.assert_metadata_defaults(result["metadata"]) 
        self.assertListEqual(result["overall_themes_identified"], [])
        
        self.assertEqual(result["content_structure"]["total_blocks"], 2)
        blocks = result["content_structure"]["blocks"]

        self.assertEqual(blocks[0]["content"], "Some content here.")
        self.assertListEqual(blocks[0]["themes_present"], [])
        self.assertEqual(blocks[0]["semantic_focus_guess"], "General Content")

        self.assertEqual(blocks[1]["content"], "More content.")
        self.assertListEqual(blocks[1]["themes_present"], [])
        self.assertEqual(blocks[1]["semantic_focus_guess"], "General Content")

    def test_no_keywords_found_in_document(self):
        doc_text = "A document about mundane things."
        keywords = ["Spirit", "Soul", "Enlightenment"]
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assert_metadata_defaults(result["metadata"])
        self.assertListEqual(result["overall_themes_identified"], [])
        
        self.assertEqual(result["content_structure"]["total_blocks"], 1)
        block = result["content_structure"]["blocks"][0]
        self.assertEqual(block["content"], "A document about mundane things.")
        self.assertListEqual(block["themes_present"], [])
        self.assertEqual(block["semantic_focus_guess"], "General Content")

    def test_single_paragraph_document_no_double_newlines(self):
        doc_text = "This is a single block of text. It mentions wisdom."
        keywords = ["Wisdom", "Text"]
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assert_metadata_defaults(result["metadata"])
        self.assertListEqual(result["overall_themes_identified"], ["Text", "Wisdom"])
        
        self.assertEqual(result["content_structure"]["total_blocks"], 1)
        block = result["content_structure"]["blocks"][0]
        self.assertEqual(block["content"], doc_text)
        self.assertListEqual(block["themes_present"], ["Text", "Wisdom"])
        self.assertEqual(block["semantic_focus_guess"], "Text, Wisdom") 
        self.assertEqual(block["character_count"], len(doc_text))

    def test_document_with_only_whitespace(self):
        doc_text = "   \n\t\n   \n  " 
        keywords = ["Test"]
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assert_metadata_defaults(result["metadata"])
        self.assertListEqual(result["overall_themes_identified"], [])
        self.assertEqual(result["content_structure"]["total_blocks"], 0)
        self.assertListEqual(result["content_structure"]["blocks"], [])

    def test_document_with_leading_trailing_empty_lines_and_varied_spacing(self):
        doc_text = "\n\n  Title: Test Doc  \n\n\n   Actual content. \n \n \n More content. \n\n"
        keywords = ["Content", "Actual"]
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assertEqual(result["metadata"]["title"], "Test Doc") # Stripped
        self.assertListEqual(result["overall_themes_identified"], ["Actual", "Content"]) # Sorted
        self.assertEqual(result["content_structure"]["total_blocks"], 3)
        
        blocks = result["content_structure"]["blocks"]
        self.assertEqual(blocks[0]["content"], "Title: Test Doc") # Stripped
        self.assertListEqual(blocks[0]["themes_present"], [])

        self.assertEqual(blocks[1]["content"], "Actual content.") # Stripped
        self.assertListEqual(blocks[1]["themes_present"], ["Actual", "Content"])

        self.assertEqual(blocks[2]["content"], "More content.") # Stripped
        self.assertListEqual(blocks[2]["themes_present"], ["Content"])


    def test_keywords_with_special_regex_characters(self):
        keywords = ["concept*", "idea+", "truth? (v)", "$value", "dot."]
        doc_text = ("Discussing concept* and idea+. Is truth? (v) a goal? What about $value or dot. ending?")
        
        result = basic_parse_esoteric_document(doc_text, keywords)
        expected_themes = sorted(["concept*", "idea+", "truth? (v)", "$value", "dot."])
        
        self.assertListEqual(result["overall_themes_identified"], expected_themes)
        
        block = result["content_structure"]["blocks"][0]
        self.assertListEqual(block["themes_present"], expected_themes)
        self.assertEqual(block["semantic_focus_guess"], ", ".join(expected_themes))

    def test_keyword_case_insensitivity_and_output_casing_preservation(self):
        doc_text = "The Soul is SOUL. The LIGHT is light."
        keywords = ["Soul", "LIGHT"] 
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assertListEqual(result["overall_themes_identified"], ["LIGHT", "Soul"]) 

        block = result["content_structure"]["blocks"][0]
        self.assertListEqual(block["themes_present"], ["LIGHT", "Soul"]) 
        self.assertEqual(block["semantic_focus_guess"], "LIGHT, Soul")

    def test_keywords_with_mixed_casing_in_input_list_are_distinct(self):
        doc_text = "The soul is a term."
        keywords = ["Soul", "soul", "Term", "term"] # Note: "Term" and "term" are distinct in input
        result = basic_parse_esoteric_document(doc_text, keywords)
        
        # Expecting all distinct input keywords that match to be present, sorted.
        expected_overall = sorted(["Soul", "soul", "Term", "term"])
        self.assertListEqual(result["overall_themes_identified"], expected_overall)

        block = result["content_structure"]["blocks"][0]
        self.assertListEqual(block["themes_present"], expected_overall)
        # Each form of keyword ("Soul", "soul", etc.) gets its own count if present in keyword list
        self.assertEqual(block["semantic_focus_guess"], ", ".join(expected_overall))

    def test_word_boundaries_for_keywords_prevent_partial_matches(self):
        doc_text = "Art is part of the heart. This is an artistic chart."
        keywords = ["Art", "Artist"] 
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assertListEqual(result["overall_themes_identified"], ["Art"])
        
        block = result["content_structure"]["blocks"][0]
        self.assertListEqual(block["themes_present"], ["Art"])
        self.assertEqual(block["semantic_focus_guess"], "Art")

    def test_title_extraction_variations(self):
        test_cases = [
            ("Title: My Book\n\nContent.", "My Book"),
            ("title: another book\n\nContent.", "another book"),
            ("TITLE:   UPPERCASE AND SPACED   \n\nContent.", "UPPERCASE AND SPACED"),
            ("Just content here.\n\nNo title specified.", "Unknown Title (Placeholder)"),
            ("Preface.\n\nTitle: Real Title In Middle\n\nChapter 1.", "Real Title In Middle"),
            ("Title: \n\nContent.", ""), # Title is present but empty after colon
        ]
        for i, (doc_text, expected_title) in enumerate(test_cases):
            with self.subTest(case_index=i, doc_text=doc_text[:20]):
                result = basic_parse_esoteric_document(doc_text, [])
                self.assertEqual(result["metadata"]["title"], expected_title)

    def test_semantic_focus_guess_logic(self):
        kw = ["Alpha", "Beta", "Gamma", "Delta"]
        # Case 1: One dominant
        doc1 = "Alpha Alpha Alpha Beta. Gamma Gamma." # A:3, B:1, G:2
        res1 = basic_parse_esoteric_document(doc1, kw)
        self.assertEqual(res1["content_structure"]["blocks"][0]["semantic_focus_guess"], "Alpha")

        # Case 2: Multiple dominant, check sort
        doc2 = "Beta Beta Alpha Alpha Gamma." # A:2, B:2, G:1
        res2 = basic_parse_esoteric_document(doc2, kw)
        self.assertEqual(res2["content_structure"]["blocks"][0]["semantic_focus_guess"], "Alpha, Beta")

        # Case 3: All same frequency
        doc3 = "Alpha Beta Gamma Delta." # All 1
        res3 = basic_parse_esoteric_document(doc3, kw)
        self.assertEqual(res3["content_structure"]["blocks"][0]["semantic_focus_guess"], "Alpha, Beta, Delta, Gamma") # Sorted

        # Case 4: No keywords found in block
        doc4 = "Plain text."
        res4 = basic_parse_esoteric_document(doc4, kw)
        self.assertEqual(res4["content_structure"]["blocks"][0]["semantic_focus_guess"], "General Content")

    def test_character_count_accuracy(self):
        doc_text = "Short.\n\nThis is a longer line of text.\n\n  Extra spaces around this line.  "
        result = basic_parse_esoteric_document(doc_text, [])
        
        blocks = result["content_structure"]["blocks"]
        self.assertEqual(blocks[0]["content"], "Short.")
        self.assertEqual(blocks[0]["character_count"], len("Short."))
        
        self.assertEqual(blocks[1]["content"], "This is a longer line of text.")
        self.assertEqual(blocks[1]["character_count"], len("This is a longer line of text."))

        self.assertEqual(blocks[2]["content"], "Extra spaces around this line.") # Stripped
        self.assertEqual(blocks[2]["character_count"], len("Extra spaces around this line."))

    def test_unicode_keywords_and_text(self):
        doc_text = "L'âme est lumière. Âme, âme!" # Soul is light. Soul, soul!
        keywords = ["Âme", "Lumière", "Esprit"] # Soul, Light, Spirit. Esprit is not in text.
        
        result = basic_parse_esoteric_document(doc_text, keywords)

        self.assertListEqual(result["overall_themes_identified"], ["Âme", "Lumière"]) # Sorted
        
        block = result["content_structure"]["blocks"][0]
        self.assertListEqual(block["themes_present"], ["Âme", "Lumière"]) 
        # Âme (from L'âme, Âme, âme) count = 3. Lumière count = 1.
        self.assertEqual(block["semantic_focus_guess"], "Âme")
        self.assertEqual(block["character_count"], len("L'âme est lumière. Âme, âme!"))

    def test_keyword_substring_word_boundary_interaction(self):
        doc_text = "The caterpillar eats catnip. A cat sits."
        keywords = ["cat", "catnip"]
        result = basic_parse_esoteric_document(doc_text, keywords)
        
        self.assertListEqual(result["overall_themes_identified"], ["cat", "catnip"])
        
        block = result["content_structure"]["blocks"][0]
        self.assertListEqual(block["themes_present"], ["cat", "catnip"]) # Both found
        # "cat" count = 1 (from "cat sits"), "catnip" count = 1
        self.assertEqual(block["semantic_focus_guess"], "cat, catnip")

    def test_block_ids_are_sequential_and_correct(self):
        doc_text = "Block one content.\n\nBlock two here.\n\nAnd block three."
        result = basic_parse_esoteric_document(doc_text, [])
        
        blocks = result["content_structure"]["blocks"]
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["block_id"], "block_1")
        self.assertEqual(blocks[1]["block_id"], "block_2")
        self.assertEqual(blocks[2]["block_id"], "block_3")

# --- Advanced NLP Tools ---
def nlp_ner(text: str) -> list:
    doc = spacy_nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def nlp_pos(text: str) -> list:
    doc = spacy_nlp(text)
    return [(token.text, token.pos_) for token in doc]

def nlp_summarize(text: str) -> str:
    summary = summarizer(text, max_length=60, min_length=10, do_sample=False)
    return summary[0]['summary_text']

def nlp_sentiment(text: str) -> dict:
    result = sentiment_analyzer(text)[0]
    return {"label": result['label'], "score": result['score']}

def nlp_textblob_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def nlp_token_count(text: str) -> int:
    return len(spacy_nlp(text))

def nlp_word_count(text: str) -> int:
    return len(text.split())

def nlp_unique_words(text: str) -> int:
    return len(set(text.split()))

def nlp_tfidf_keywords(text: str, top_n: int = 5) -> list:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    indices = np.argsort(X.toarray()[0])[::-1][:top_n]
    features = np.array(vectorizer.get_feature_names_out())
    return features[indices].tolist()

def nlp_topic_modeling(text: str, n_topics: int = 2) -> list:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)
    topics = []
    for topic in lda.components_:
        top_words = np.array(vectorizer.get_feature_names_out())[topic.argsort()[-5:][::-1]]
        topics.append(top_words.tolist())
    return topics

def nlp_semantic_similarity(text1: str, text2: str) -> float:
    doc1 = spacy_nlp(text1)
    doc2 = spacy_nlp(text2)
    return doc1.similarity(doc2)

def nlp_language_detection(text: str) -> str:
    blob = TextBlob(text)
    return blob.detect_language() if hasattr(blob, 'detect_language') else 'unknown'

def nlp_readability(text: str) -> float:
    # Flesch Reading Ease (simple version)
    blob = TextBlob(text)
    words = blob.words
    sentences = blob.sentences
    syllables = sum([len(word) for word in words])
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))

# --- AI Agent with Extended NLP Tools ---
def generate_response(messages: List[Dict]) -> str:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running the script.")
    client = genai.Client(api_key=api_key)
    history = []
    for m in messages:
        role = 'user' if m['role'] in ['system', 'user'] else 'model'
        history.append({'role': role, 'parts': [m['content']]})
    chat = client.chats.create(model='gemini-2.5-pro-preview-05-06', history=history[:-1])
    response = chat.send_message(history[-1]['parts'][0])
    return response.text

def extract_code_block(response: str) -> str:
    if '```' not in response:
        return response
    code_block = response.split('```')[1].strip()
    if code_block.startswith('python'):
        code_block = code_block[6:]
    return code_block.strip()

def extract_markdown_block(text: str, block_type: str) -> str:
    import re
    pattern = rf"```{block_type}([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text

def parse_action(response: str) -> Dict:
    try:
        response = extract_markdown_block(response, "action")
        response_json = json.loads(response)
        if "tool_name" in response_json and "args" in response_json:
            return response_json
        else:
            return {"tool_name": "error", "args": {"message": "You must respond with a JSON tool invocation."}}
    except json.JSONDecodeError:
        return {"tool_name": "error", "args": {"message": "Invalid JSON response. You must respond with a JSON tool invocation."}}

def list_files() -> list:
    return [f for f in os.listdir('.') if os.path.isfile(f)]

def read_file(file_name: str) -> str:
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def terminate(message: str):
    print(message)
    return message

def main():
    agent_rules = [
        {
            "role": "system",
            "content": """
You are an AI agent that can perform advanced NLP and semantic analysis using the following tools.

Available tools:
- nlp_ner(text: str) -> list: Named Entity Recognition.
- nlp_pos(text: str) -> list: Part-of-speech tagging.
- nlp_summarize(text: str) -> str: Summarize the text.
- nlp_sentiment(text: str) -> dict: Sentiment analysis (transformers).
- nlp_textblob_sentiment(text: str) -> dict: Sentiment analysis (TextBlob).
- nlp_token_count(text: str) -> int: Token count.
- nlp_word_count(text: str) -> int: Word count.
- nlp_unique_words(text: str) -> int: Unique word count.
- nlp_tfidf_keywords(text: str, top_n: int) -> list: Top TF-IDF keywords.
- nlp_topic_modeling(text: str, n_topics: int) -> list: Topic modeling.
- nlp_semantic_similarity(text1: str, text2: str) -> float: Semantic similarity.
- nlp_language_detection(text: str) -> str: Language detection.
- nlp_readability(text: str) -> float: Flesch reading ease.
- terminate(message: str): End the agent loop and print a summary.

Every response MUST have an action.
Respond in this format:
```action
{
    \"tool_name\": \"insert tool_name\",
    \"args\": {...fill in any required arguments here...}
}
```
"""
        }
    ]
    memory = []
    user_input = input("What would you like the agent to do? ")
    memory.append({"role": "user", "content": user_input})
    max_iterations = 10
    iterations = 0
    while iterations < max_iterations:
        prompt = agent_rules + memory
        print("Agent thinking...")
        response = generate_response(prompt)
        print(f"Agent response: {response}")
        action = parse_action(response)
        result = None
        if action["tool_name"] == "list_files":
            result = {"result": list_files()}
        elif action["tool_name"] == "read_file":
            result = {"result": read_file(action["args"].get("file_name", ""))}
        elif action["tool_name"] == "error":
            result = {"error": action["args"].get("message", "Unknown error")}
        elif action["tool_name"] == "terminate":
            print(action["args"].get("message", "Terminating agent."))
            break
        elif action["tool_name"] == "nlp_ner":
            result = {"result": nlp_ner(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_pos":
            result = {"result": nlp_pos(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_summarize":
            result = {"result": nlp_summarize(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_sentiment":
            result = {"result": nlp_sentiment(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_textblob_sentiment":
            result = {"result": nlp_textblob_sentiment(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_token_count":
            result = {"result": nlp_token_count(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_word_count":
            result = {"result": nlp_word_count(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_unique_words":
            result = {"result": nlp_unique_words(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_tfidf_keywords":
            result = {"result": nlp_tfidf_keywords(action["args"].get("text", ""), action["args"].get("top_n", 5))}
        elif action["tool_name"] == "nlp_topic_modeling":
            result = {"result": nlp_topic_modeling(action["args"].get("text", ""), action["args"].get("n_topics", 2))}
        elif action["tool_name"] == "nlp_semantic_similarity":
            result = {"result": nlp_semantic_similarity(action["args"].get("text1", ""), action["args"].get("text2", ""))}
        elif action["tool_name"] == "nlp_language_detection":
            result = {"result": nlp_language_detection(action["args"].get("text", ""))}
        elif action["tool_name"] == "nlp_readability":
            result = {"result": nlp_readability(action["args"].get("text", ""))}
        else:
            result = {"error": "Unknown action: " + str(action["tool_name"])}
        print(f"Action result: {result}")
        memory.append({"role": "assistant", "content": response})
        memory.append({"role": "user", "content": json.dumps(result)})
        if action["tool_name"] == "terminate":
            break
        iterations += 1

if __name__ == "__main__":
    main()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)