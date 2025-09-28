import gradio as gr
import os
from pathlib import Path
import tempfile
import shutil
# import json # Not needed if parse_whisper_options is used directly

from transcribe_anything.api import transcribe
from transcribe_anything._cmd import WHISPER_MODEL_OPTIONS # Import model options
from transcribe_anything.parse_whisper_options import parse_whisper_options # Corrected import

def gradio_transcribe(
    url_or_file_input: str,
    model_input: str,
    language_input: str,
    device_input: str,
    output_folder_input: str,
    task_input: str,
    hf_token_input: str | None,
    initial_prompt_input: str | None,
    progress=gr.Progress(), # Add progress bar
) -> tuple[str | None, str | None, str | None, str]:
    try:
        progress(0, desc="Starting transcription...")
        # Ensure output_folder_input is treated as a Path
        base_output_dir = Path(output_folder_input)
        
        # Clean up previous run's output to ensure a fresh start for the demo
        if base_output_dir.exists():
            shutil.rmtree(base_output_dir)
        progress(0.1, desc="Creating output directory...")
        base_output_dir.mkdir(parents=True, exist_ok=True)

        progress(0.2, desc="Calling transcription API...")
        # Call the core transcribe function
        final_output_path = transcribe(
            url_or_file=url_or_file_input,
            output_dir=str(base_output_dir),
            model=model_input,
            language=language_input if language_input != "auto" else None,
            device=device_input,
            task=task_input,
            hugging_face_token=hf_token_input,
            initial_prompt=initial_prompt_input,
        )
        progress(0.7, desc="Processing output files...")
        # Construct paths to the expected output files
        srt_path = Path(final_output_path) / "out.srt"
        txt_path = Path(final_output_path) / "out.txt"
        json_path = Path(final_output_path) / "out.json"
        
        # Check if files exist before returning
        progress(0.9, desc="Finalizing...")
        return (
            str(srt_path) if srt_path.exists() else None,
            str(txt_path) if txt_path.exists() else None,
            str(json_path) if json_path.exists() else None,
            f"Transcription successful! Files saved to: {final_output_path}"
        )
    except Exception as e:
        import traceback
        error_message = f"An error occurred: {e}\n\n{traceback.format_exc()}"
        return (None, None, None, error_message)

def get_all_languages():
    """Helper to get all available languages."""
    options = parse_whisper_options()
    return [None, "auto"] + options["language"]

def get_all_tasks():
    """Helper to get all available tasks (transcribe/translate)."""
    options = parse_whisper_options()
    return options["task"]

# Define Gradio Interface
with gr.Blocks(title="Transcribe Anything GUI (with German Whisper Turbo)") as iface_blocks:
    gr.Markdown("# Transcribe Anything GUI (with German Whisper Turbo)")
    gr.Markdown("Transcribe audio/video from URL or local files using Whisper models. The output will be saved in a subfolder within the specified output base folder, named after the input file.")

    with gr.Row():
        url_or_file_input = gr.Textbox(label="URL or Local File Path", placeholder="Enter YouTube URL or local path (e.g., audio.mp3)", interactive=True)
        model_input = gr.Dropdown(
            WHISPER_MODEL_OPTIONS + ["primeline/whisper-large-v3-turbo-german"],
            label="Model",
            value="primeline/whisper-large-v3-turbo-german", # Default to our German model
            interactive=True,
        )
        language_input = gr.Dropdown(
            choices=get_all_languages(),
            label="Language",
            value="de", # Default to German
            interactive=True,
        )
        device_input = gr.Radio(
            ["cuda", "cpu", "insane"], # Expose common device options
            label="Device",
            value="insane", # Default to insane for GPU
            interactive=True,
        )
        output_folder_input = gr.Textbox(label="Output Base Folder (e.g., C:\transcriptions)", value=str(Path(tempfile.gettempdir()) / "transcribe_anything_output"), interactive=True)

    with gr.Accordion("Advanced Options", open=False):
        task_input = gr.Dropdown(
            choices=get_all_tasks(),
            label="Task (Transcribe or Translate)",
            value="transcribe",
            interactive=True,
        )
        hf_token_input = gr.Textbox(label="HuggingFace Token (Optional, for private models)", type="password", interactive=True)
        initial_prompt_input = gr.Textbox(label="Initial Prompt (Optional, provide context for transcription)", lines=3, interactive=True)

    submit_button = gr.Button("Transcribe")

    with gr.Row():
        srt_output = gr.File(label="SRT Output")
        txt_output = gr.File(label="TXT Output")
        json_output = gr.File(label="JSON Output")
    status_output = gr.Textbox(label="Status / Info")

    submit_button.click(
        fn=gradio_transcribe,
        inputs=[
            url_or_file_input,
            model_input,
            language_input,
            device_input,
            output_folder_input,
            task_input,
            hf_token_input,
            initial_prompt_input,
        ],
        outputs=[
            srt_output,
            txt_output,
            json_output,
            status_output,
        ],
    )

if __name__ == "__main__":
    iface_blocks.launch(inbrowser=True)
