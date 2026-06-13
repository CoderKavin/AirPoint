"""
AirPoint — single entry point for PyInstaller bundle.
Runs the auto-updater, then launches the main app, all in one process.
"""
import os
import sys

# Windows DPI awareness — must be set before any GUI/pyautogui calls
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    # Force UTF-8 stdout/stderr so emoji debug prints don't choke cp1252 consoles.
    # Only relevant in source-run mode — in the frozen bundle stdout is sent to devnull.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None:
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

# Suppress all warnings/logs before any heavy imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "3"

import warnings
warnings.filterwarnings("ignore")

# APP_DIR = folder where profiles/ and VERSION live.
# On macOS .app bundle: sys.executable is inside AirPoint.app/Contents/MacOS/
# so APP_DIR should be the folder *containing* the .app bundle.
# On Windows/Linux folder build: sys.executable is directly in the dist folder.
if getattr(sys, 'frozen', False):
    exe_dir = os.path.dirname(sys.executable)
    # Detect macOS .app bundle (path contains .app/Contents/MacOS)
    if '.app/Contents/MacOS' in exe_dir.replace('\\', '/'):
        APP_DIR = os.path.abspath(os.path.join(exe_dir, '..', '..', '..'))
    else:
        APP_DIR = exe_dir
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# BUNDLE_DIR = where PyInstaller unpacked bundled data (VERSION, main.py, …).
# In a frozen build this is the _internal folder (sys._MEIPASS), NOT the folder
# that holds the .exe. VERSION ships INSIDE the bundle, so it must be read from
# here. Reading it from APP_DIR (next to the exe) finds nothing, which makes the
# updater believe the installed version is 0.0.0 and prompt to "update" on every
# single launch — the bug behind the endless update loop.
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
else:
    BUNDLE_DIR = APP_DIR

# Ensure profiles dir exists
os.makedirs(os.path.join(APP_DIR, "profiles"), exist_ok=True)


def run_updater():
    """Check for updates and apply one if the user accepts.
    Exits the process via SystemExit(0) if an update is being installed —
    the detached swap script then replaces our files and relaunches us.
    Never blocks launch on failure.
    """
    if "--skip-update" in sys.argv:
        return
    try:
        import launcher
        # Point the launcher module at our resolved APP_DIR
        # (matters for the .app bundle on macOS, where APP_DIR is the folder
        # CONTAINING AirPoint.app, not the bundle internals).
        launcher.APP_DIR = APP_DIR
        # VERSION lives inside the bundle (_internal), not next to the exe.
        launcher.VERSION_FILE = os.path.join(BUNDLE_DIR, "VERSION")
        launcher.CRASH_LOG = os.path.join(APP_DIR, "crash.log")

        if launcher.perform_update_check():
            # Update is staged; the swap script will take over.
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        pass  # Never let update failure prevent the app from launching.


def _vc_runtime_installed():
    """Check if VC++ Runtime marker exists (set after successful install)."""
    return os.path.exists(os.path.join(APP_DIR, ".vc_installed"))


def _write_dll_log(exc):
    """Record the native-load failure to dll_error.log so the real cause
    (which module/DLL failed) can be diagnosed instead of guessed."""
    try:
        import traceback
        log_path = os.path.join(APP_DIR, "dll_error.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("AirPoint failed to load a required native library.\n\n")
            f.write(f"Error: {exc}\n\n")
            f.write("Full traceback:\n")
            f.write(traceback.format_exc())
        return log_path
    except Exception:
        return None


def _show_dll_error(exc, log_path=None):
    """Explain a native-DLL load failure; offer a one-time VC++ install."""
    if sys.platform != "win32":
        return

    VC_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()

        # If we already installed the runtime before, don't offer again —
        # the problem is something else (needs reboot, wrong arch, etc.)
        if _vc_runtime_installed():
            details = f"\n\nA log was saved to:\n{log_path}" if log_path else ""
            messagebox.showerror(
                "AirPoint",
                "AirPoint can't load a required native library.\n\n"
                f"Failed component:\n{exc}\n\n"
                "Visual C++ is already installed, so this is NOT a VC++ "
                "problem. The most likely cause is that Windows Defender "
                "quarantined a file inside AirPoint's own folder "
                "(the 'AirPoint\\_internal' folder), leaving the app "
                "incomplete.\n\n"
                "To fix it:\n"
                "1. Open Windows Security -> Virus & threat protection -> "
                "Protection history, and click Restore on anything listed "
                "for AirPoint.\n"
                "2. Under Manage settings -> Exclusions, add the AirPoint "
                "folder as an exclusion.\n"
                "3. Delete the AirPoint folder, then re-extract the "
                "download into the excluded location and relaunch."
                + details
            )
            root.destroy()
            return

        answer = messagebox.askyesno(
            "AirPoint — One-time Setup",
            "AirPoint needs to install a small system component "
            "(Microsoft Visual C++ Runtime) to run.\n\n"
            "This is a one-time setup and only takes a moment.\n\n"
            "Install now?"
        )

        if answer:
            import urllib.request
            import tempfile
            import subprocess

            installer = os.path.join(tempfile.gettempdir(), "vc_redist.x64.exe")

            # Show a "downloading" message
            info = tk.Toplevel(root)
            info.title("AirPoint")
            info.geometry("340x80")
            info.resizable(False, False)
            tk.Label(info, text="Downloading required component...\nThis will only take a moment.", pady=20).pack()
            info.update()

            urllib.request.urlretrieve(VC_URL, installer)
            info.destroy()

            # Run the installer silently — /install /passive does not need admin
            # and shows a small progress bar
            result = subprocess.run(
                [installer, "/install", "/passive", "/norestart"], check=False
            )

            try:
                os.remove(installer)
            except OSError:
                pass

            # vc_redist exit codes: 0 = success, 1638 = newer already installed,
            # 3010 = success but reboot required. Anything else = failure.
            install_ok = result.returncode in (0, 1638, 3010)

            if install_ok:
                # Mark install complete so we don't offer it again next launch
                try:
                    with open(os.path.join(APP_DIR, ".vc_installed"), "w") as f:
                        f.write("1")
                except OSError:
                    pass

                messagebox.showinfo(
                    "AirPoint",
                    "Setup complete! Please relaunch AirPoint.\n\n"
                    "If AirPoint still doesn't start, try restarting your computer."
                )
            else:
                messagebox.showerror(
                    "AirPoint",
                    f"Setup didn't complete (installer exit code {result.returncode}).\n\n"
                    "Please install the Microsoft Visual C++ Redistributable manually from:\n"
                    "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
                    "Then relaunch AirPoint."
                )
        else:
            messagebox.showinfo(
                "AirPoint",
                "AirPoint can't run without this component.\n"
                "You can relaunch AirPoint to install it later."
            )

        root.destroy()

    except Exception:
        # If anything fails, fall back to a plain message
        print(
            "AirPoint failed to load a required native library.\n"
            f"Error: {exc}\n"
            + (f"Details written to: {log_path}\n" if log_path else "")
            + "If Visual C++ is already installed, check Windows Defender's\n"
            "Protection history for quarantined AirPoint files and restore them.\n"
            "Otherwise install: https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )


def _looks_like_native_load_failure(exc):
    """Heuristic: does this exception look like a missing VC++ runtime / native lib?"""
    msg = str(exc).upper()
    return any(token in msg for token in (
        "_FRAMEWORK_BINDINGS",
        "DLL LOAD FAILED",
        "MSVCP",
        "MSVCR",
        "VCRUNTIME",
        "VCOMP",
        "API-MS-WIN",
    ))


def run_app():
    """Launch the main AirPoint application."""
    try:
        # Load MediaPipe into a still-clean process BEFORE `import main`
        # (which pulls in OpenCV, PyQt5, etc.). In a frozen build MediaPipe's
        # _framework_bindings native module fails with "DLL initialization
        # routine failed" if OpenCV/Qt native DLLs load first. Importing it
        # here first is the documented workaround.
        import mediapipe  # noqa: F401
        import main
    except (ImportError, OSError) as e:
        if _looks_like_native_load_failure(e):
            log_path = _write_dll_log(e)
            _show_dll_error(e, log_path)
            raise SystemExit(1)
        raise
    # Override main's APP_DIR in case it was set differently
    main.APP_DIR = APP_DIR
    main.PROFILES_DIR = os.path.join(APP_DIR, "profiles")
    main.CRASH_LOG = os.path.join(APP_DIR, "crash.log")

    # Suppress console output in frozen mode
    if getattr(sys, 'frozen', False):
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # Re-invoke main's __main__ logic
    sys.excepthook = main.show_crash_dialog

    import argparse
    parser = argparse.ArgumentParser(description="AirPoint")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--no-gaze", action="store_true")
    parser.add_argument("--dwell", action="store_true")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--generate-default", action="store_true")
    args = parser.parse_args()

    if args.generate_default:
        import json
        os.makedirs(main.PROFILES_DIR, exist_ok=True)
        path = os.path.join(main.PROFILES_DIR, "default.json")
        with open(path, "w") as f:
            json.dump(main.DEFAULT_CONFIG, f, indent=2)
        raise SystemExit(0)

    try:
        gaze = not args.no_gaze
        controller = main.HandCenterGestureController(enable_gaze_detection=gaze)
        if args.dwell:
            controller.dwell_click_enabled = True
        if args.profile:
            if not controller.load_profile(args.profile):
                # Profile didn't exist or failed to load. Keep the name so a save
                # creates it, but make sure the user knows defaults are in use.
                try:
                    sys.__stderr__.write(
                        f"AirPoint: profile '{args.profile}' not found — "
                        f"using defaults; will be created on first save.\n"
                    )
                except Exception:
                    pass
                controller.profile_name = args.profile
            # else loaded successfully
        controller.run()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    except Exception:
        main.show_crash_dialog(*sys.exc_info())


if __name__ == "__main__":
    run_updater()
    run_app()
