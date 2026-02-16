# whisper_transcriber.py
"""
Download and transcribe audio/video files (mp3/mp4) using OpenAI Whisper.
"""
import os
import tempfile
import requests
import whisper

def download_file(url, suffix):
    response = requests.get(url)
    response.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(response.content)
    return path

def transcribe_audio(url):
    ext = os.path.splitext(url)[1].lower()
    if ext not in [".mp3", ".mp4"]:
        raise ValueError("Unsupported file type for transcription.")
    file_path = download_file(url, ext)
    model = whisper.load_model("medium")  # You can use "small", "medium", or "large" for better accuracy
    result = model.transcribe(file_path)
    os.remove(file_path)
    return result["text"].strip()
