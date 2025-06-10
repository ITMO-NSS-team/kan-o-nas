@echo off

REM Add directories to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;C:\dev\aim\nas4kan;C:\dev\aim\nas4kan\cases

REM Run the Python script
C:\dev\aim\nas4kan\venv\Scripts\python.exe C:/dev/aim/nas4kan/cases\mnist_classification.py
