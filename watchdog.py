"""Watchdog — monitors the Gradio lecture processor and auto-restarts if it goes down.

Run in a SEPARATE terminal:
    conda run -n catalan-lecture python watchdog.py

Or use --once to just ensure the server is running (for scheduled tasks):
    conda run -n catalan-lecture python watchdog.py --once
"""

import os
import sys
import time
import urllib.request
import subprocess
import logging

URL = "http://127.0.0.1:7860/"
CHECK_INTERVAL = 60   # seconds between checks
FAIL_THRESHOLD = 2    # consecutive failures before restarting
RESTART_COOLDOWN = 120  # seconds to wait after a restart before checking again

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(PROJECT_DIR, "run_desktop.py")
CONDA_ENV = "catalan-lecture"
LOG_FILE = os.path.join(PROJECT_DIR, "watchdog.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("watchdog")


def notify_windows(title, message):
    """Show a Windows toast notification via PowerShell."""
    ps_script = f"""
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
    $template = @"
    <toast>
        <visual><binding template="ToastText02">
            <text id="1">{title}</text>
            <text id="2">{message}</text>
        </binding></visual>
        <audio src="ms-winsoundevent:Notification.Default"/>
    </toast>
"@
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml($template)
    $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Lecture Processor")
    $notifier.Show([Windows.UI.Notifications.ToastNotification]::new($xml))
    """
    try:
        subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def check_server():
    """Return True if server responds, False otherwise."""
    try:
        req = urllib.request.Request(URL, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


def find_conda():
    """Find the conda executable."""
    for path in [
        os.path.expanduser("~/anaconda3/condabin/conda.bat"),
        os.path.expanduser("~/miniconda3/condabin/conda.bat"),
        "conda",
    ]:
        if os.path.isfile(path) or path == "conda":
            return path
    return "conda"


def start_server():
    """Launch run_desktop.py in a new detached process."""
    conda = find_conda()
    log.info("Starting server: %s run -n %s python %s", conda, CONDA_ENV, RUN_SCRIPT)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    # Launch detached so it survives if watchdog is stopped
    if sys.platform == "win32":
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            [conda, "run", "-n", CONDA_ENV, "python", RUN_SCRIPT],
            cwd=PROJECT_DIR,
            env=env,
            creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            [conda, "run", "-n", CONDA_ENV, "python", RUN_SCRIPT],
            cwd=PROJECT_DIR,
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    log.info("Server process launched, waiting %ds for startup...", RESTART_COOLDOWN)
    notify_windows("Lecture Processor", "Server was down — restarting automatically...")


def run_once():
    """Check once and start server if not running. For use in scheduled tasks."""
    if check_server():
        log.info("Server is already running.")
    else:
        log.info("Server is not running. Starting...")
        start_server()
        time.sleep(RESTART_COOLDOWN)
        if check_server():
            log.info("Server started successfully.")
            notify_windows("Lecture Processor", "Server started automatically on boot.")
        else:
            log.error("Server failed to start.")
            notify_windows("Lecture Processor ERROR", "Auto-start failed! Check watchdog.log")


def run_loop():
    """Continuous monitoring loop with auto-restart."""
    log.info("Watchdog started — checking %s every %ds", URL, CHECK_INTERVAL)
    log.info("Auto-restart enabled (after %d consecutive failures)", FAIL_THRESHOLD)
    print("Press Ctrl+C to stop.\n")

    consecutive_fails = 0
    was_down = False

    while True:
        up = check_server()

        if up:
            if was_down:
                log.info("Server is back online!")
                notify_windows("Lecture Processor", "Server is back online!")
                was_down = False
            else:
                log.info("Server responding")
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            log.warning("No response (fail %d/%d)", consecutive_fails, FAIL_THRESHOLD)

            if consecutive_fails >= FAIL_THRESHOLD and not was_down:
                log.error("Server is DOWN — attempting auto-restart...")
                start_server()
                was_down = True
                # Wait for restart before checking again
                time.sleep(RESTART_COOLDOWN)
                continue

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        if "--once" in sys.argv:
            run_once()
        else:
            run_loop()
    except KeyboardInterrupt:
        log.info("Watchdog stopped.")
