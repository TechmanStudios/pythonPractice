import os
import tempfile
import requests
import whisper


def download_audio(url: str) -> str:
    """Download an audio or video file and return a temporary file path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    ext = os.path.splitext(url)[1] or ".mp3"
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
    return path


def transcribe_audio(url: str) -> str:
    """Download media from ``url`` and return the Whisper transcription."""
    audio_path = download_audio(url)
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
    except Exception:
        text = ""
    finally:
        os.remove(audio_path)
    return text
