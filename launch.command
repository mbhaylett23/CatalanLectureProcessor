#!/bin/bash
# Catalan Lecture Processor - macOS Launcher
# Double-click this file to start the app
#
# First time? You may need to:
#   1. Right-click this file > Open (to bypass Gatekeeper)
#   2. Or: System Settings > Privacy & Security > Allow

cd "$(dirname "$0")"

# Check Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  ERROR: Python 3 is not installed."
    echo ""
    echo "  To install it, open Terminal and run:"
    echo "    brew install python@3.12"
    echo ""
    echo "  If you don't have Homebrew, first run:"
    echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    read -p "  Press Enter to exit..."
    exit 1
fi

python3 setup_and_run.py

# Keep window open if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "  Something went wrong. Please screenshot this window"
    echo "  and send it to your instructor."
    echo ""
    read -p "  Press Enter to close..."
fi
