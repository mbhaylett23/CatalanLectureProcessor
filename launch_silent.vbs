' Catalan Lecture Processor - Silent Launcher (no black window)
' This launches the app without showing a command prompt window.
' The app opens directly in your browser.

Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c python setup_and_run.py", 0, False
