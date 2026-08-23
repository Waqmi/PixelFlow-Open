#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
if [[ "${SCRIPT_DIR:t}" == "macOS" && "${SCRIPT_DIR:h:t}" == "02_构建配置" ]]; then
  cd "${SCRIPT_DIR:h:h}"
else
  cd "${SCRIPT_DIR:h:h}"
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
APP_NAME="${APP_NAME:-PixelFlow-macOS-arm64}"
TARGET_ARCH="${TARGET_ARCH:-arm64}"

export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache-$TARGET_ARCH"
SPEC_OUTPUT_DIR="$PWD/build/generated-specs"
mkdir -p "$SPEC_OUTPUT_DIR"

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --specpath "$SPEC_OUTPUT_DIR" \
  --windowed \
  --name "$APP_NAME" \
  --target-arch "$TARGET_ARCH" \
  --icon "$PWD/resources/app-icon.icns" \
  --add-data "$PWD/resources:resources" \
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
  --exclude-module PySide6.QtSvg \
  --exclude-module PySide6.QtSvgWidgets \
  --exclude-module PySide6.QtMultimedia \
  --exclude-module PySide6.QtMultimediaWidgets \
  main_image_tool.py

if [[ -f prune_mac_bundle.py ]]; then
  "$PYTHON_BIN" prune_mac_bundle.py
fi
/usr/bin/xattr -rc "dist/$APP_NAME.app"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName PixelFlow" -c "Set :CFBundleName PixelFlow" \
  "dist/$APP_NAME.app/Contents/Info.plist"
codesign --force --deep --sign - "dist/$APP_NAME.app"

echo "Mac 应用已生成：dist/$APP_NAME.app"
