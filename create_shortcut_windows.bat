@echo off
:: Creates a desktop shortcut for the Catalan Lecture Processor
:: Run this ONCE to create the shortcut, then use the desktop icon

set SCRIPT_DIR=%~dp0
set SHORTCUT_NAME=Catalan Lecture Processor
set DESKTOP=%USERPROFILE%\Desktop

echo Creating desktop shortcut...

:: Use PowerShell to create a proper .lnk shortcut
powershell -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $shortcut = $ws.CreateShortcut('%DESKTOP%\%SHORTCUT_NAME%.lnk'); ^
   $shortcut.TargetPath = '%SCRIPT_DIR%launch.bat'; ^
   $shortcut.WorkingDirectory = '%SCRIPT_DIR%'; ^
   $shortcut.Description = 'Catalan Lecture Processor - Transcribe, translate, and summarise lectures'; ^
   $shortcut.WindowStyle = 1; ^
   $shortcut.Save()"

echo.
echo Done! You should see "%SHORTCUT_NAME%" on your desktop.
echo Double-click it to start the app.
echo.
pause
