
import os
from pathlib import Path
import tempfile

from transcribe_anything.api import transcribe

# Path to the sample audio file
input_audio_file = Path("src/transcribe_anything/assets/sample.mp3")

# Define the output directory
output_dir = Path(tempfile.gettempdir()) / "transcribe_anything_test_output"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Transcribing {input_audio_file} using German model...")

# Call the transcribe function with the new model and language
transcribe(
    url_or_file=str(input_audio_file),
    output_dir=str(output_dir),
    model="primeline/whisper-large-v3-turbo-german", # Direct repo_id
    language="de",
    device="insane", # Assuming 'insane' (GPU) for this model
)

print(f"Transcription complete. Output saved to {output_dir}")

# Optionally, print the contents of the generated SRT file
srt_file = output_dir / "out.srt"
if srt_file.exists():
    print("\n--- Generated SRT content ---")
    print(srt_file.read_text(encoding="utf-8"))
else:
    print("SRT file not found.")
