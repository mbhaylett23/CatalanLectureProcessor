"""
Auto-setup and launch script for Catalan Lecture Processor.

This script handles everything:
1. Creates a Python virtual environment (if not already created)
2. Installs all dependencies (if not already installed)
3. Checks for ffmpeg
4. Launches the Gradio app in the default browser

Students just double-click launch.bat (Windows) or launch.command (macOS).
"""

import subprocess
import sys
import os
import platform
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(PROJECT_DIR, "requirements.txt")

# Virtual environment lives on the LOCAL disk for performance.
# Cloud-synced drives (Google Drive, OneDrive) are too slow for Python imports.
_LOCAL_VENV_BASE = os.path.join(os.path.expanduser("~"), ".venvs")
VENV_DIR = os.path.join(_LOCAL_VENV_BASE, "CatalanLectureProcessor")
SETUP_VERSION = "2"  # Bump this to force reinstall on next launch
SETUP_MARKER = os.path.join(VENV_DIR, ".setup_complete")

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

if IS_WINDOWS:
    PYTHON_VENV = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP_VENV = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON_VENV = os.path.join(VENV_DIR, "bin", "python")
    PIP_VENV = os.path.join(VENV_DIR, "bin", "pip")


def print_header(msg):
    print()
    print("=" * 60)
    print(f"  {msg}")
    print("=" * 60)
    print()


def print_step(msg):
    print(f"  >> {msg}")


def check_python():
    """Make sure we're running Python 3.10+."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"\n  ERROR: Python 3.10 or higher is required.")
        print(f"  You have Python {major}.{minor}.")
        print(f"  Download the latest from: https://www.python.org/downloads/")
        input("\n  Press Enter to exit...")
        sys.exit(1)
    arch = platform.machine()  # e.g. x86_64, arm64, AMD64
    print_step(f"Python {major}.{minor} detected ({platform.system()} {arch})")


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    if shutil.which("ffmpeg"):
        print_step("ffmpeg found")
        return True

    print()
    print("  WARNING: ffmpeg is not installed.")
    print("  ffmpeg is needed to process audio files.")
    print()
    if IS_WINDOWS:
        print("  To install it, open a new Command Prompt and run:")
        print("    winget install ffmpeg")
        print()
        print("  Or download from: https://ffmpeg.org/download.html")
    elif IS_MAC:
        print("  To install it, open Terminal and run:")
        print("    brew install ffmpeg")
        print()
        print("  (If you don't have brew: https://brew.sh)")
    else:
        print("  Install with your package manager:")
        print("    sudo apt install ffmpeg    (Ubuntu/Debian)")
        print("    sudo dnf install ffmpeg    (Fedora)")
    print()
    print("  The app will still launch, but audio processing will fail")
    print("  until ffmpeg is installed.")
    print()
    return False


def create_venv():
    """Create virtual environment if it doesn't exist."""
    if os.path.exists(PYTHON_VENV):
        print_step("Virtual environment already exists")
        return

    print_step("Creating virtual environment (one-time setup)...")
    os.makedirs(_LOCAL_VENV_BASE, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    print_step(f"Virtual environment created at {VENV_DIR}")


def install_dependencies():
    """Install pip packages if not already done."""
    if os.path.exists(SETUP_MARKER):
        try:
            with open(SETUP_MARKER) as f:
                if f.read().strip() == SETUP_VERSION:
                    print_step("Dependencies already installed")
                    return
                print_step("Setup updated -- reinstalling dependencies...")
        except OSError:
            pass

    print_step("Installing dependencies (this may take a few minutes)...")
    print_step("Downloading: Whisper, translation models, UI framework...")
    print()

    subprocess.check_call(
        [PYTHON_VENV, "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Install PyTorch — platform-specific to avoid downloading huge GPU builds
    if IS_WINDOWS:
        # Windows: use CPU-only index (saves downloading 2GB+ CUDA version)
        print_step("Installing PyTorch (CPU-only)...")
        subprocess.check_call(
            [PYTHON_VENV, "-m", "pip", "install", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
        )
    else:
        # macOS (Intel + Apple Silicon): default PyPI wheel works on both
        print_step("Installing PyTorch...")
        subprocess.check_call(
            [PYTHON_VENV, "-m", "pip", "install", "torch"],
        )

    # Install the rest of the requirements
    print_step("Installing remaining packages...")
    subprocess.check_call(
        [PYTHON_VENV, "-m", "pip", "install", "-r", REQUIREMENTS],
    )

    # Write marker so we don't reinstall every time
    with open(SETUP_MARKER, "w") as f:
        f.write(SETUP_VERSION)

    print()
    print_step("All dependencies installed!")


def launch_app():
    """Launch the Gradio app."""
    print_header("Launching Catalan Lecture Processor")
    print_step("Starting app... your browser will open automatically.")
    print_step("When you're done, close this window to stop the app.")
    print()
    print("  -------------------------------------------------")
    print("  The app runs at: http://127.0.0.1:7860")
    print("  -------------------------------------------------")
    print()

    run_desktop = os.path.join(PROJECT_DIR, "run_desktop.py")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.call([PYTHON_VENV, run_desktop], env=env)


def main():
    print_header("Catalan Lecture Processor - Setup")

    try:
        check_python()
        check_ffmpeg()
        create_venv()
        install_dependencies()
        launch_app()
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: A command failed: {e}")
        print("  Please check the error messages above.")
        input("\n  Press Enter to exit...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  App stopped by user.")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        input("\n  Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
