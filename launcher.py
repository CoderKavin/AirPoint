"""
AirPoint Launcher — auto-update + launch.

Checks GitHub releases for a newer version, downloads and applies the update
(preserving user profiles), then launches main.py.

Usage:
    python launcher.py              # Check for updates, then launch
    python launcher.py --skip-update  # Launch without checking
"""

import os
import sys
import json
import shutil
import zipfile
import tempfile
import subprocess
import traceback
import urllib.request
import urllib.error
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(APP_DIR, "VERSION")
CRASH_LOG = os.path.join(APP_DIR, "crash.log")
REPO = "CoderKavin/chetana-version-build"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"

# Directories that must NEVER be overwritten (user data)
PROTECTED_DIRS = {"profiles", "venv", "__pycache__", ".git", ".claude"}
PROTECTED_FILES = {".gitignore"}


def get_local_version():
    try:
        with open(VERSION_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0"


def parse_version(v):
    """Parse 'v1.2.3' or '1.2.3' into a tuple (1, 2, 3)."""
    v = v.lstrip("vV").strip()
    parts = v.split(".")
    return tuple(int(p) for p in parts)


def check_for_update():
    """Check GitHub releases API. Returns (tag, zip_url) or None."""
    try:
        req = urllib.request.Request(API_URL, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        tag = data.get("tag_name", "")
        zip_url = data.get("zipball_url", "")
        if tag and zip_url:
            return tag, zip_url
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
            OSError, KeyError, ValueError):
        pass
    return None


def download_and_extract(zip_url, dest_dir):
    """Download a zip from URL and extract to dest_dir. Returns True on success."""
    tmp_zip = os.path.join(tempfile.gettempdir(), "airpoint_update.zip")
    try:
        print("  Downloading update...")
        urllib.request.urlretrieve(zip_url, tmp_zip)

        print("  Extracting...")
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(dest_dir)

        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)


def apply_update(extracted_dir):
    """Copy new files from extracted_dir into APP_DIR, preserving protected dirs."""
    # GitHub zipball extracts into a subdirectory like "User-Repo-abc1234/"
    subdirs = [d for d in os.listdir(extracted_dir)
               if os.path.isdir(os.path.join(extracted_dir, d))]
    if len(subdirs) == 1:
        source = os.path.join(extracted_dir, subdirs[0])
    else:
        source = extracted_dir

    updated = 0
    for item in os.listdir(source):
        src_path = os.path.join(source, item)
        dst_path = os.path.join(APP_DIR, item)

        # Skip protected directories and files
        if item in PROTECTED_DIRS or item in PROTECTED_FILES:
            continue
        if item.startswith("."):
            continue

        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            updated += 1
        else:
            shutil.copy2(src_path, dst_path)
            updated += 1

    return updated


def main():
    skip_update = "--skip-update" in sys.argv

    local_ver = get_local_version()
    print(f"AirPoint v{local_ver}")

    if not skip_update:
        result = check_for_update()
        if result:
            remote_tag, zip_url = result
            try:
                remote_ver = parse_version(remote_tag)
                local_ver_t = parse_version(local_ver)
            except (ValueError, IndexError):
                remote_ver = (0,)
                local_ver_t = (0,)

            if remote_ver > local_ver_t:
                print(f"Updating to {remote_tag}...")
                tmp_dir = tempfile.mkdtemp(prefix="airpoint_update_")
                try:
                    if download_and_extract(zip_url, tmp_dir):
                        count = apply_update(tmp_dir)
                        print(f"  Updated {count} files. Now running {remote_tag}.")
                    else:
                        print("  Update failed — launching current version.")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                print("Already up to date.")
        else:
            print("Update check skipped (offline or repo not found).")

    # Launch main.py
    main_py = os.path.join(APP_DIR, "main.py")
    if not os.path.exists(main_py):
        print("Error: main.py not found!")
        sys.exit(1)

    # Forward any extra args (except --skip-update)
    extra_args = [a for a in sys.argv[1:] if a != "--skip-update"]
    os.execv(sys.executable, [sys.executable, main_py] + extra_args)


def _show_launcher_crash(exc_type, exc_value, exc_tb):
    """Write crash log and show a simple Tk error dialog (PyQt5 may not be available yet)."""
    try:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        entry = (
            f"\n{'='*60}\n"
            f"LAUNCHER CRASH  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*60}\n"
            f"{tb_text}\n"
        )
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass
    # Try tkinter (always available in Python) for a minimal dialog
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "AirPoint Launcher Error",
            f"The launcher ran into a problem:\n\n{exc_type.__name__}: {exc_value}\n\n"
            f"Details saved to crash.log"
        )
        root.destroy()
    except Exception:
        traceback.print_exception(exc_type, exc_value, exc_tb)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        _show_launcher_crash(*sys.exc_info())
