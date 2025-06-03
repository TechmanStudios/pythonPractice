import subprocess
import os
import whisper

def process_video(video_url, output_file):
    """Process a single video URL and append its transcription to output_file."""
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load the Whisper model (using 'base' for speed; consider 'small' or 'medium' for extra accuracy)
    model = whisper.load_model("base")

    try:
        # Extract video ID (assumes standard URL format)
        video_id = video_url.split("=")[-1]
        # Get video title
        title = subprocess.check_output(["yt-dlp", "--get-title", video_url]).decode().strip()
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"Video: {title} (URL: {video_url})\n")
            try:
                # Try to extract subtitles first
                subtitles = subprocess.check_output(
                    ["yt-dlp", "--write-auto-sub", "--sub-format", "txt", "--skip-download", "--output", "-", video_url],
                    stderr=subprocess.DEVNULL
                ).decode()
                if subtitles.strip():
                    f.write("Transcription (from subtitles):\n")
                    f.write(subtitles)
                else:
                    raise subprocess.CalledProcessError(1, "yt-dlp")
            except subprocess.CalledProcessError:
                # No subtitles available; download audio and transcribe it
                audio_file = f"{video_id}.mp3"
                subprocess.run(["yt-dlp", "--extract-audio", "--audio-format", "mp3", "--output", audio_file, video_url])
                result = model.transcribe(audio_file)
                transcription = result["text"]
                f.write("Transcription (from audio):\n")
                f.write(transcription)
                os.remove(audio_file)  # Clean up the downloaded audio file
            f.write("\n\n")
    except Exception as e:
        print(f"Error processing {video_url}: {e}")

def main():
    video_url = input("Enter the YouTube video URL: ").strip()
    output_file = input("Enter the output file path (e.g., /path/to/transcriptions.txt): ")
    process_video(video_url, output_file)
    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":
    main()
