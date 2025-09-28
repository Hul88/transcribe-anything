@echo off
set VENV_DIR=venv_gradio
call %VENV_DIR%\Scripts\activate.bat
python gui.py
pause
