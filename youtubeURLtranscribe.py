import subprocess
import os
import whisper

def get_channel_video_urls(channel_url):
    """Fetch all video URLs from a YouTube channel using yt-dlp."""
    command = ["yt-dlp", "--flat-playlist", "--print", "url", channel_url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching video URLs: {result.stderr}")
        return []
    return result.stdout.splitlines()

def process_videos(video_urls, output_file):
    """Process a list of video URLs and append transcriptions to output_file."""
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:  # If there's a directory part in the path
        os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model once (use 'base' for speed, or 'small'/'medium' for accuracy)
    model = whisper.load_model("base")

    with open(output_file, "a", encoding="utf-8") as f:
        for url in video_urls:
            try:
                # Extract video ID from URL
                video_id = url.split("=")[-1]
                # Get video title
                title = subprocess.check_output(["yt-dlp", "--get-title", url]).decode().strip()
                f.write(f"Video: {title} (URL: {url})\n")

                try:
                    # Try to extract subtitles
                    subtitles = subprocess.check_output(
                        ["yt-dlp", "--write-auto-sub", "--sub-format", "txt", "--skip-download", "--output", "-", url],
                        stderr=subprocess.DEVNULL
                    ).decode()
                    if subtitles.strip():
                        f.write("Transcription (from subtitles):\n")
                        f.write(subtitles)
                    else:
                        raise subprocess.CalledProcessError(1, "yt-dlp")
                except subprocess.CalledProcessError:
                    # No subtitles, download audio and transcribe
                    audio_file = f"{video_id}.mp3"
                    subprocess.run(["yt-dlp", "--extract-audio", "--audio-format", "mp3", "--output", audio_file, url])
                    result = model.transcribe(audio_file)
                    transcription = result["text"]
                    f.write("Transcription (from audio):\n")
                    f.write(transcription)
                    os.remove(audio_file)  # Clean up
                f.write("\n\n")
            except Exception as e:
                print(f"Error processing {url}: {e}")

def main():
    # Option 1: Hardcode a list of URLs here
    hardcoded_urls = [
        # "https://www.youtube.com/watch?v=VIDEO_ID_1",
        # "https://www.youtube.com/watch?v=VIDEO_ID_2",
        # Add more URLs as needed
    ]

    # Option 2: Fetch URLs from a channel
    use_channel = input("Do you want to fetch URLs from a channel? (yes/no): ").lower() == "yes"

    if use_channel:
        channel_url = input("Enter the YouTube channel URL: ")
        video_urls = get_channel_video_urls(channel_url)
        if not video_urls:
            print("No video URLs retrieved. Exiting.")
            return
    elif hardcoded_urls:
        video_urls = hardcoded_urls
        print("Using hardcoded list of URLs.")
    else:
        print("No hardcoded URLs provided and channel fetching declined. Please edit the script or try again.")
        return

    output_file = input("Enter the output file path (e.g., C:/Users/You/Documents/transcriptions.txt): ")
    process_videos(video_urls, output_file)
    print(f"Transcriptions saved to {output_file}")

if __name__ == "__main__":
    main()