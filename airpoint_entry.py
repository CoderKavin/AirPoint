"""
AirPoint — single entry point for PyInstaller bundle.
Runs the auto-updater, then launches the main app, all in one process.
"""
import os
import sys

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

# Ensure profiles dir exists
os.makedirs(os.path.join(APP_DIR, "profiles"), exist_ok=True)


def run_updater():
    """Check for updates (best-effort, never blocks launch on failure)."""
    if "--skip-update" in sys.argv:
        return
    try:
        # Import launcher module for update logic
        import launcher
        # Override its APP_DIR to match ours
        launcher.APP_DIR = APP_DIR
        launcher.VERSION_FILE = os.path.join(APP_DIR, "VERSION")
        launcher.CRASH_LOG = os.path.join(APP_DIR, "crash.log")

        result = launcher.check_for_update()
        if result:
            remote_tag, zip_url = result
            try:
                remote_ver = launcher.parse_version(remote_tag)
                local_ver = launcher.parse_version(launcher.get_local_version())
            except (ValueError, IndexError):
                return
            if remote_ver > local_ver:
                import tempfile
                import shutil
                tmp_dir = tempfile.mkdtemp(prefix="airpoint_update_")
                try:
                    if launcher.download_and_extract(zip_url, tmp_dir):
                        launcher.apply_update(tmp_dir)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass  # never let update failure prevent app from launching


def run_app():
    """Launch the main AirPoint application."""
    import main
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
