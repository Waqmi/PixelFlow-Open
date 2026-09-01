# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).parents[1]


EXCLUDES = [
    "PySide6.QtNetwork",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQml",
    "PySide6.QtQmlModels",
    "PySide6.QtQmlMeta",
    "PySide6.QtQmlWorkerScript",
    "PySide6.QtQuick",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
]

LOCAL_HIDDEN_IMPORTS = [
    "image_engine_api",
    "generate_main_images",
    "output_validator",
    "category_engine",
    "category_engine.category_classifier",
    "category_engine.category_template_manager",
    "category_engine.composition_engine",
    "category_engine.source_type_classifier",
    "local_model_assistant",
    "onnxruntime",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]


a = Analysis(
    [str(PROJECT_ROOT / "main_image_tool.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[(str(PROJECT_ROOT / "resources"), "resources")],
    hiddenimports=LOCAL_HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PixelFlow-Windows-x64",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon=[str(PROJECT_ROOT / "resources" / "app-icon.ico")],
)
