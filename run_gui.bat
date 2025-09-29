@echo off
set VENV_DIR=venv
call %VENV_DIR%\Scripts\activate.bat
set PYTHONPATH=%CD%
python gui.py
pause
