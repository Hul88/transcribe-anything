@echo off
echo Creating and activating virtual environment "venv"...
set VENV_DIR=venv
python -m venv %VENV_DIR%
call %VENV_DIR%\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing core dependencies, including Gradio, openai-whisper, and CPU-enabled Torch...
pip install static-ffmpeg yt-dlp appdirs disklru FileLock webvtt-py==0.4.6 uv-iso-env python-dotenv gradio openai-whisper==20240930 numpy==1.26.4 torch==2.2.1

echo Installing transcribe-anything in editable mode...
pip install -e .

echo.
echo Installation complete!
echo.
echo =========================================================================================
echo IMPORTANT: If you have an NVIDIA GPU, you can upgrade to a CUDA-enabled Torch for better performance.
echo To do this, after installation, keep the virtual environment active (as it is now)
echo and run the following command in your terminal:
echo.
echo    pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121
echo.
echo You can then close this window and run the GUI by double-clicking 'run_gui.bat'.
echo =========================================================================================
echo.
pause
