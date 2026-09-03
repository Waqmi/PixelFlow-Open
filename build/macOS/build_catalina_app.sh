#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "${SCRIPT_DIR:h:h}"

if [[ "$(uname -m)" != "x86_64" ]]; then
  print -u2 "Catalina build must run on an Intel (x86_64) Mac."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-.venv-catalina/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  print -u2 "Missing .venv-catalina. Create an Intel Python 3.9 environment and install build/macOS/requirements-catalina.txt first."
  exit 1
fi

APP_NAME="${APP_NAME:-PixelFlow-macOS-catalina-x64}"
export MACOSX_DEPLOYMENT_TARGET=10.15
export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache-catalina-x64"

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --target-arch x86_64 \
  --icon "$PWD/resources/app-icon.icns" \
  --add-data "$PWD/resources:resources" \
  --hidden-import onnxruntime \
  --collect-binaries onnxruntime \
  --exclude-module PySide6.QtNetwork \
  --exclude-module PySide6.QtOpenGL \
  --exclude-module PySide6.QtOpenGLWidgets \
  --exclude-module PySide6.QtPdf \
  --exclude-module PySide6.QtPdfWidgets \
  --exclude-module PySide6.QtQml \
  --exclude-module PySide6.QtQmlModels \
  --exclude-module PySide6.QtQmlMeta \
  --exclude-module PySide6.QtQmlWorkerScript \
  --exclude-module PySide6.QtQuick \
  --exclude-module PySide6.QtQuickControls2 \
  --exclude-module PySide6.QtQuickWidgets \
  --exclude-module PySide6.QtMultimedia \
  --exclude-module PySide6.QtMultimediaWidgets \
  main_image_tool.py

/usr/bin/xattr -rc "dist/$APP_NAME.app"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName PixelFlow" -c "Set :CFBundleName PixelFlow" \
  "dist/$APP_NAME.app/Contents/Info.plist"
codesign --force --deep --sign - "dist/$APP_NAME.app"

echo "Catalina app generated: dist/$APP_NAME.app"
