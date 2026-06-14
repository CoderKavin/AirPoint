# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for AirPoint.
Produces a single folder (not one-file) so profiles/ can live alongside it.
"""
import os
import sys
import platform
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect ALL mediapipe data (models, .tflite, .binarypb, .txt, etc.)
mediapipe_datas = collect_data_files('mediapipe')
mediapipe_imports = collect_submodules('mediapipe')

# Collect native .pyd/.dll/.so files that mediapipe and its deps need at runtime
mediapipe_binaries = collect_dynamic_libs('mediapipe')
cv2_binaries = collect_dynamic_libs('cv2')
protobuf_binaries = collect_dynamic_libs('google.protobuf')

PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

# Platform-specific icon. Falls back to no icon if the file isn't present
# (e.g. during local dev before the icon has been generated).
_ICON_MAC = os.path.join(PROJECT_DIR, 'assets', 'icon.icns')
_ICON_WIN = os.path.join(PROJECT_DIR, 'assets', 'icon.ico')
if platform.system() == 'Darwin' and os.path.exists(_ICON_MAC):
    APP_ICON = _ICON_MAC
elif platform.system() == 'Windows' and os.path.exists(_ICON_WIN):
    APP_ICON = _ICON_WIN
else:
    APP_ICON = None

# --- Windows: ship the Microsoft Visual C++ runtime NEXT TO each consumer --
# PyInstaller omits msvcp140.dll/msvcp140_1.dll/concrt140.dll, treating them as
# OS-provided "system" DLLs. OpenCV (cv2) and MediaPipe's opencv_world3410.dll
# are C++ and need them. Crucially, Windows resolves a DLL's imports from the
# IMPORTING DLL's own directory — NOT from _internal root — so the runtime must
# sit in the SAME folder as each consumer. That is exactly how PyQt5 ships its
# own copies (PyQt5/Qt5/bin), which is why PyQt5 loads but cv2/mediapipe did
# not when the runtime was only in _internal root. Place a copy next to each.
vcredist_binaries = []
if platform.system() == 'Windows':
    _sys32 = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
    _required_crt = [
        'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll',
    ]
    _optional_crt = ['msvcp140_1.dll', 'msvcp140_2.dll']
    # _internal root (for python310.dll) + next to the cv2 and mediapipe C++ DLLs.
    _crt_dests = ['.', 'cv2', os.path.join('mediapipe', 'python')]
    for _dll in _required_crt + _optional_crt:
        _src = os.path.join(_sys32, _dll)
        if os.path.exists(_src):
            for _dest in _crt_dests:
                vcredist_binaries.append((_src, _dest))
        elif _dll in _required_crt:
            raise SystemExit(
                f"AirPoint.spec: required VC++ runtime '{_dll}' not found in "
                f"{_sys32}. Install the Visual C++ Redistributable on the build "
                f"machine so the frozen app can bundle it."
            )

a = Analysis(
    [os.path.join(PROJECT_DIR, 'airpoint_entry.py')],
    pathex=[PROJECT_DIR],
    binaries=mediapipe_binaries + cv2_binaries + protobuf_binaries + vcredist_binaries,
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
        'mediapipe.python._framework_bindings',
        'certifi',  # auto-updater points urllib's SSL at certifi's CA bundle
    ] + mediapipe_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'scipy',
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
    icon=APP_ICON,
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
if platform.system() == 'Darwin':
    app = BUNDLE(
        coll,
        name='AirPoint.app',
        icon=APP_ICON,
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
