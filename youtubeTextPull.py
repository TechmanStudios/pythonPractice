#!/usr/bin/env python3
"""
youtubeTextPull.py

Goal:
- Transcribe one or many YouTube videos.
- Prefer Whisper (for proper nouns like "Thoth") and write ONE FILE PER VIDEO.

Key reliability features (2025 YouTube reality):
- Auto-detects an external JS runtime (deno/node/bun/quickjs) and enables it for yt-dlp.
  (yt-dlp strongly recommends installing one for full YouTube support.)
- Tries multiple "player clients" to avoid 403 blocks.
- Optional: use cookies-from-browser to bypass stricter anti-bot checks.

Dependencies (pip):
  pip install yt-dlp openai-whisper
  (you already installed torch w/ CUDA)

System:
  ffmpeg must be installed and in PATH

Usage:
- Run and paste URLs interactively
- Or: python youtubeTextPull.py URL1 URL2 URL3
- Or: type @urls.txt at the URL prompt (one URL per line)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import whisper

# -------------------- Settings --------------------

# Whisper-first (recommended for domain terms like "Thoth")
PREFER_WHISPER = True

# If you want subtitles as a fallback when audio download fails, keep this True.
ALLOW_SUBTITLES_FALLBACK = True

# Optional: use cookies from a local browser profile to reduce 403 blocks.
# Set to "chrome", "edge", or "firefox" if you want this enabled.
# Example: COOKIES_FROM_BROWSER = "chrome"
COOKIES_FROM_BROWSER: Optional[str] = None

# Default URLs if you just press Enter at the first prompt
DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=jH7N0jmDrIY",
]

# Output directory prompt default (leave as None to default to master file's folder)
DEFAULT_OUT_MASTER = r"G:\GPTs\ThothStream\personaPrimer\TSwisdom1.txt"

# Each video gets its own file in OUTPUT_DIR (default: same folder as master file)
OUTPUT_DIR: Optional[str] = None

DEFAULT_MODEL = "medium"  # you have CUDA now, so this is a good pick

# Which YouTube "client" to try first for downloads.
# web_safari often exposes HLS formats that can avoid some PO-token/403 scenarios. :contentReference[oaicite:3]{index=3}
YOUTUBE_CLIENT_PRIORITY = ["web_safari", "tv", "android"]

# -------------------- URL input --------------------

URL_RE = re.compile(r"https?://\S+")


def prompt_default(label: str, default: str) -> str:
    s = input(f"{label} [{default}]: ").strip()
    return s or default


def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text or "")


def _add_urls_from_line(line: str, urls: List[str]) -> None:
    line = line.strip()
    if not line:
        return

    # Allow @urls.txt
    if line.startswith("@"):
        p = Path(line[1:]).expanduser()
        if not p.exists():
            print(f"Could not find URL file: {p}")
            return
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                urls.extend(extract_urls(ln) or [ln])
        return

    urls.extend(extract_urls(line))


def prompt_urls(default_urls: List[str]) -> List[str]:
    print("Enter YouTube URL (or paste many; blank = default list)")
    first = input(f"[{default_urls[0]}]: ").strip()

    urls: List[str] = []
    if first:
        _add_urls_from_line(first, urls)
        print("Paste more URL(s), one per line. Press Enter on a blank line to begin.\n")
        while True:
            line = input().strip()
            if not line:
                break
            _add_urls_from_line(line, urls)
    else:
        urls = list(default_urls)

    # De-dupe while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# -------------------- yt-dlp helpers --------------------

@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    title: str


def run_capture(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def detect_js_runtime() -> Optional[str]:
    """
    yt-dlp supports external JS runtimes for YouTube extraction (EJS).
    Deno is recommended; node/bun/quickjs are also supported. :contentReference[oaicite:4]{index=4}
    Return the runtime name if found on PATH.
    """
    for runtime in ["deno", "node", "bun", "quickjs"]:
        if shutil.which(runtime):
            return runtime
    return None


def ytdlp_base_args() -> List[str]:
    args = ["yt-dlp", "--no-playlist"]

    # Enable JS runtime if present (yt-dlp warns / deprecates extraction without it). :contentReference[oaicite:5]{index=5}
    jsrt = detect_js_runtime()
    if jsrt:
        args += ["--js-runtimes", jsrt]
    else:
        print(
            "WARNING: No supported JS runtime found (deno/node/bun/quickjs).\n"
            "         yt-dlp may fail or miss formats. Install Deno for best results."
        )

    if COOKIES_FROM_BROWSER:
        args += ["--cookies-from-browser", COOKIES_FROM_BROWSER]

    return args


def get_video_info(url: str) -> VideoInfo:
    """
    Use yt-dlp JSON to get id/title.
    Parse stdout only; stderr can contain warnings that break JSON.
    """
    cmd = ytdlp_base_args() + ["--dump-single-json", url]
    cp = run_capture(cmd)
    if cp.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {url}:\n{(cp.stderr or '').strip()}")

    raw = (cp.stdout or "").strip()
    if not raw:
        raise RuntimeError("yt-dlp returned empty JSON on stdout.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        preview = raw[:400].replace("\n", "\\n")
        raise RuntimeError(f"Could not parse yt-dlp JSON. Preview: {preview}") from e

    vid = data.get("id") or ""
    title = data.get("title") or data.get("fulltitle") or vid or "unknown-title"
    if not vid:
        raise RuntimeError("yt-dlp JSON did not contain a video id.")

    return VideoInfo(video_id=vid, title=title)


# -------------------- Subtitle support (optional fallback) --------------------

_ANY_TAG = re.compile(r"</?[^>]+>")
_TS_TAG = re.compile(r"<\d{1,2}:\d{2}:\d{2}\.\d{3}>")
_C_OPEN = re.compile(r"<c[^>]*>")
_C_CLOSE = re.compile(r"</c>")


def clean_caption_text(text: str) -> str:
    text = re.sub(r"(?im)^\s*Kind:\s*captions.*$", "", text)
    text = _TS_TAG.sub(" ", text)
    text = _C_OPEN.sub("", text)
    text = _C_CLOSE.sub("", text)
    text = _ANY_TAG.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def srt_to_text(srt_path: Path) -> str:
    out_lines: List[str] = []
    prev = ""
    for raw in srt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if "-->" in line:
            continue
        line = _ANY_TAG.sub("", line).strip()
        if line and line != prev:
            out_lines.append(line)
            prev = line
    return clean_caption_text(" ".join(out_lines))


def vtt_to_text(vtt_path: Path) -> str:
    out_lines: List[str] = []
    prev = ""
    for raw in vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "WEBVTT":
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if line.startswith(("NOTE", "STYLE", "REGION")):
            continue

        line = _TS_TAG.sub(" ", line)
        line = _C_OPEN.sub("", line)
        line = _C_CLOSE.sub("", line)
        line = _ANY_TAG.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()

        if line and line != prev:
            out_lines.append(line)
            prev = line

    return clean_caption_text(" ".join(out_lines))


def try_get_subtitles(url: str, tmp: Path, vid: str) -> Optional[str]:
    outtmpl = str(tmp / "%(id)s.%(language)s.%(ext)s")
    cmd = ytdlp_base_args() + [
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-lang",
        "en.*",
        "--sub-format",
        "vtt",
        "--convert-subs",
        "srt",
        "-o",
        outtmpl,
        url,
    ]

    cp = run_capture(cmd)
    if cp.returncode != 0:
        return None

    srts = sorted(tmp.glob(f"{vid}*.srt"))
    vtts = sorted(tmp.glob(f"{vid}*.vtt"))

    if srts:
        best = next((p for p in srts if ".en" in p.name), srts[0])
        return srt_to_text(best)

    if vtts:
        best = next((p for p in vtts if ".en" in p.name), vtts[0])
        return vtt_to_text(best)

    return None


# -------------------- Audio download + Whisper --------------------

def download_audio_mp3(url: str, tmp: Path, vid: str) -> Path:
    """
    Best-effort audio download.
    Tries multiple youtube player clients to avoid 403 failures.
    """
    outtmpl = str(tmp / "%(id)s.%(ext)s")

    # Prefer m4a if possible (often simpler than opus/webm in some cases)
    fmt = "bestaudio[ext=m4a]/bestaudio/best"

    last_err = None

    for client in YOUTUBE_CLIENT_PRIORITY:
        cmd = ytdlp_base_args() + [
            "-x",
            "--audio-format",
            "mp3",
            "-f",
            fmt,
            "--extractor-args",
            f"youtube:player_client={client}",
            "-o",
            outtmpl,
            url,
        ]

        print(f"  - Trying yt-dlp audio with player_client={client} ...")
        cp = run_capture(cmd)
        if cp.returncode == 0:
            audio_path = tmp / f"{vid}.mp3"
            if audio_path.exists():
                return audio_path

            mp3s = list(tmp.glob("*.mp3"))
            if mp3s:
                return mp3s[0]
            last_err = "Audio download succeeded but no mp3 found."
            continue

        combined = (cp.stderr or "") + "\n" + (cp.stdout or "")
        last_err = combined.strip()

        # If 403 happens, keep trying other clients, then possibly cookies.
        if "HTTP Error 403" in combined or "403" in combined:
            continue

    raise RuntimeError(
        "Audio download failed for all strategies.\n\n"
        "Most common fixes:\n"
        "1) Install Deno (JS runtime) for yt-dlp YouTube support.\n"
        "2) Enable COOKIES_FROM_BROWSER = 'chrome' (or 'edge'/'firefox') in the script.\n"
        "3) Update yt-dlp to latest.\n\n"
        f"Last yt-dlp output:\n{last_err}"
    )


def transcribe_with_whisper(model, audio_path: Path) -> str:
    use_fp16 = torch.cuda.is_available()
    result = model.transcribe(
        str(audio_path),
        fp16=use_fp16,
        language="en",
        temperature=0,
    )
    return (result.get("text") or "").strip()


# -------------------- Output: separate file per video --------------------

def safe_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "transcript"
    return s[:max_len]


def write_per_video_file(out_dir: Path, info: VideoInfo, url: str, source: str, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{safe_filename(info.title)}__{info.video_id}.txt"
    path = out_dir / fname

    header = f"Video: {info.title}\nURL: {url}\nSource: {source}\n\n"
    path.write_text(header + (text or "[No transcription text produced]") + "\n", encoding="utf-8")
    return path


# -------------------- Main processing --------------------

def process_one(url: str, out_dir: Path, model) -> None:
    info = get_video_info(url)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        text: Optional[str] = None
        source = ""

        if not PREFER_WHISPER and ALLOW_SUBTITLES_FALLBACK:
            text = try_get_subtitles(url, tmp, info.video_id)
            if text:
                source = "subtitles"

        if not text:
            # Whisper path (preferred)
            audio_path = download_audio_mp3(url, tmp, info.video_id)
            text = transcribe_with_whisper(model, audio_path)
            source = "audio (Whisper)"

        # If Whisper fails (rare), and you allowed subtitle fallback:
        if (not text) and ALLOW_SUBTITLES_FALLBACK:
            text = try_get_subtitles(url, tmp, info.video_id)
            if text:
                source = "subtitles"

        per_video_path = write_per_video_file(out_dir, info, url, source, text)
        print(f"Saved per-video transcript to: {per_video_path}")


def main() -> None:
    urls = prompt_urls(DEFAULT_URLS)
    if not urls:
        print("No URLs provided. Exiting.")
        return

    master_path = Path(prompt_default("Master output .txt path (used for default folder)", DEFAULT_OUT_MASTER))
    model_name = prompt_default("Whisper model", DEFAULT_MODEL)

    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else master_path.parent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nWhisper device: {device} | fp16: {torch.cuda.is_available()} | model: {model_name}")
    print(f"Per-video output directory: {out_dir}")

    jsrt = detect_js_runtime()
    print(f"yt-dlp JS runtime: {jsrt if jsrt else 'NONE (install deno recommended)'}")
    if COOKIES_FROM_BROWSER:
        print(f"yt-dlp cookies-from-browser: {COOKIES_FROM_BROWSER}")

    model = whisper.load_model(model_name, device=device)

    total = len(urls)
    for i, url in enumerate(urls, start=1):
        print(f"\n[{i}/{total}] Processing: {url}")
        try:
            process_one(url, out_dir, model)
        except Exception as e:
            print(f"ERROR processing {url}: {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
