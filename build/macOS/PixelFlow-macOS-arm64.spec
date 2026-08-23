# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_image_tool.py'],
    pathex=[],
    binaries=[],
    datas=[('resources', 'resources')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtNetwork', 'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtQml', 'PySide6.QtQmlModels', 'PySide6.QtQmlMeta', 'PySide6.QtQmlWorkerScript', 'PySide6.QtQuick', 'PySide6.QtQuickControls2', 'PySide6.QtQuickWidgets', 'PySide6.QtSvg', 'PySide6.QtSvgWidgets', 'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'numpy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PixelFlow-macOS-arm64',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources/app-icon.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PixelFlow-macOS-arm64',
)
app = BUNDLE(
    coll,
    name='PixelFlow-macOS-arm64.app',
    icon='resources/app-icon.icns',
    bundle_identifier=None,
)
