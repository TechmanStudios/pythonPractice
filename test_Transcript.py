import os
import whisper

def batch_transcribe_audio(
    input_folder=r"G:\GPTs\QLI\Vimeo",
    output_file=r"G:\GPTs\QLI\vimeoText.txt",
    model_size="medium"
):
    """
    Scans input_folder for any .m4a or .mp3 files, transcribes each with Whisper,
    and appends them (with headings) to one big text file: output_file.
    Also prints progress for each file.
    """

    # Grab all .m4a or .mp3 files
    audio_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".m4a", ".mp3"))
    ]
    total_files = len(audio_files)
    print(f"Found {total_files} audio files in '{input_folder}'.")

    # Load your Whisper model
    print(f"Loading Whisper model '{model_size}'... Please wait.")
    model = whisper.load_model(model_size)  
    print("Model loaded! Starting batch transcription...\n")

    # Open the final transcript file once, in write mode
    with open(output_file, "w", encoding="utf-8") as out:
        # Enumerate so we can display how many we've done out of total
        for i, file_name in enumerate(audio_files, start=1):
            file_path = os.path.join(input_folder, file_name)
            print(f"[{i}/{total_files}] Transcribing '{file_name}'...")

            # Transcribe
            result = model.transcribe(file_path)
            transcript_text = result["text"].strip()

            # Write a heading + the transcript text
            out.write(f"### {file_name}\n\n{transcript_text}\n\n")
            out.write("----\n\n")  # Separator between transcripts

    print(f"\nAll done! Your combined transcript is saved to:\n{output_file}")

if __name__ == "__main__":
    batch_transcribe_audio(
        input_folder=r"G:\GPTs\QLI\Vimeo",  
        output_file=r"G:\GPTs\QLI\vimeoText.txt",
        model_size="medium"
    )
