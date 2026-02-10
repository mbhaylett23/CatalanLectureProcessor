#!/bin/bash
# Creates a desktop shortcut (macOS .app bundle) for the Catalan Lecture Processor
# Run this ONCE to create the app, then use the desktop icon

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Catalan Lecture Processor"
DESKTOP="$HOME/Desktop"
APP_PATH="$DESKTOP/$APP_NAME.app"

echo "Creating desktop app..."

# Create .app bundle structure
mkdir -p "$APP_PATH/Contents/MacOS"
mkdir -p "$APP_PATH/Contents/Resources"

# Create the launcher script inside the .app
cat > "$APP_PATH/Contents/MacOS/launcher" << 'LAUNCHER'
#!/bin/bash
SCRIPT_DIR="PLACEHOLDER_DIR"
cd "$SCRIPT_DIR"

# Open Terminal with the setup script
osascript -e "
tell application \"Terminal\"
    activate
    do script \"cd '$SCRIPT_DIR' && python3 setup_and_run.py\"
end tell
"
LAUNCHER

# Replace placeholder with actual path
sed -i '' "s|PLACEHOLDER_DIR|$SCRIPT_DIR|g" "$APP_PATH/Contents/MacOS/launcher"
chmod +x "$APP_PATH/Contents/MacOS/launcher"

# Create Info.plist
cat > "$APP_PATH/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Catalan Lecture Processor</string>
    <key>CFBundleDisplayName</key>
    <string>Catalan Lecture Processor</string>
    <key>CFBundleIdentifier</key>
    <string>com.lecture.processor</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
PLIST

echo ""
echo "Done! You should see '$APP_NAME' on your desktop."
echo "Double-click it to start the app."
echo ""
echo "NOTE: The first time you open it, macOS may ask you to"
echo "allow it in System Settings > Privacy & Security."
echo ""
