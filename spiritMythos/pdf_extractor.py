# pdf_extractor.py
"""
Extract text from PDF files using PyPDF2, pdfminer, or OCR (pytesseract).
"""
import os
import tempfile
import requests
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from PIL import Image
import pytesseract


def download_pdf(url):
    response = requests.get(url)
    response.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(response.content)
    return path


def extract_text_pypdf2(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or '' for page in reader.pages)
        return text.strip()
    except Exception:
        return ""


def extract_text_pdfminer(pdf_path):
    try:
        return pdfminer_extract_text(pdf_path).strip()
    except Exception:
        return ""


def extract_text_ocr(pdf_path):
    from pdf2image import convert_from_path
    try:
        images = convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
        return text.strip()
    except Exception:
        return ""


def extract_pdf_text(url):
    pdf_path = download_pdf(url)
    text = extract_text_pypdf2(pdf_path)
    if not text:
        text = extract_text_pdfminer(pdf_path)
    if not text:
        text = extract_text_ocr(pdf_path)
    os.remove(pdf_path)
    return text
