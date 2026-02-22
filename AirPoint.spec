# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for AirPoint.
Produces a single folder (not one-file) so profiles/ can live alongside it.
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect ALL mediapipe data (models, .tflite, .binarypb, .txt, etc.)
mediapipe_datas = collect_data_files('mediapipe')
mediapipe_imports = collect_submodules('mediapipe')

PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(PROJECT_DIR, 'airpoint_entry.py')],
    pathex=[PROJECT_DIR],
    binaries=[],
    datas=[
        # App files
        (os.path.join(PROJECT_DIR, 'main.py'), '.'),
        (os.path.join(PROJECT_DIR, 'launcher.py'), '.'),
        (os.path.join(PROJECT_DIR, 'VERSION'), '.'),
    ] + mediapipe_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.face_mesh',
        'cv2',
        'numpy',
        'pyautogui',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.sip',
        'google.protobuf',
        'google.protobuf.descriptor',
    ] + mediapipe_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'scipy', 'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AirPoint',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,       # NO console window
    disable_windowed_traceback=False,
    argv_emulation=True,  # macOS: pass argv properly
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AirPoint',
)

# macOS only: wrap into a .app bundle
import platform
if platform.system() == 'Darwin':
    app = BUNDLE(
        coll,
        name='AirPoint.app',
        icon=None,
        bundle_identifier='org.chetana.airpoint',
        info_plist={
            'CFBundleName': 'AirPoint',
            'CFBundleDisplayName': 'AirPoint',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSCameraUsageDescription': 'AirPoint needs camera access for hand tracking.',
            'NSHighResolutionCapable': True,
        },
    )
