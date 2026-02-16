import whisper
import whisper.audio  # import the audio submodule once
import subprocess

def main():
    audio_file = r"G:\GPTs\QLI\Vimeo\Energy Cleansing.m4a"
    output_file = r"C:\pythonPractice\textOut\test1.txt"

    print("Loading the Whisper model...")
    model = whisper.load_model("medium")  
    print("Model loaded. Starting transcription...")

    # Debug: Patch whisper.audio.run() to print the command
    old_run = subprocess.run

    def debug_run(cmd, *args, **kwargs):
        print(f"DEBUG: Subprocess command => {cmd}")
        return old_run(cmd, *args, **kwargs)

    # Assign our debug function to whisper.audio's run
    whisper.audio.run = debug_run

    # Now run the transcription
    result = model.transcribe(audio_file)
    print("Transcription complete! Writing output...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"Transcript saved to: {output_file}")

if __name__ == "__main__":
    main()