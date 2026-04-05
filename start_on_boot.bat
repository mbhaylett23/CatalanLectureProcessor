@echo off
REM Auto-start Catalan Lecture Processor on boot
REM Place a shortcut to this file in: shell:startup
REM (Win+R > shell:startup > paste shortcut)

set PYTHONUTF8=1
cd /d "G:\My Drive\PythonCode\TranslationProject"
call C:\Users\mbruy\anaconda3\condabin\conda.bat run -n catalan-lecture python watchdog.py --once

REM Then keep monitoring in the background
start "" /min cmd /c "C:\Users\mbruy\anaconda3\condabin\conda.bat run -n catalan-lecture python watchdog.py"
