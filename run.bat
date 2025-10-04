@echo off
REM Activate the venv and run interactive Python - adjust module below as needed
call "%~dp0venv\Scripts\activate.bat"
python -c "import decimal_computer; print('OK ->', getattr(decimal_computer,'__file__',decimal_computer))"
pause
