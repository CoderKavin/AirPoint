"""
AirPoint auto-updater.

For BUNDLED installs (PyInstaller .app / folder build): downloads the platform's
release asset from GitHub (DMG on macOS, ZIP on Windows), prompts the user,
spawns a detached swap script to replace the install while the running app
exits, then the script relaunches the new version.

For SOURCE runs (developer mode): skips the auto-updater entirely. Devs sync
via git.

Can also be invoked directly (`python launcher.py`) to run main.py without
update checks - kept for compatibility with older autostart entries.
"""

import os
import sys
import ssl
import json
import shutil
import zipfile
import tempfile
import subprocess
import traceback
import urllib.request
import urllib.error
from datetime import datetime


def _ssl_context():
    """An SSL context backed by a CA bundle that actually exists in the frozen
    app. The bundled OpenSSL's compiled-in cert path points at a python.org
    build location that is absent on users' machines, so urllib's default
    HTTPS verification fails (silently) on macOS. certifi ships its own CA
    bundle inside the app, so we point verification at that. Falls back to the
    system default if certifi is unavailable (e.g. source runs)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        try:
            return ssl.create_default_context()
        except Exception:
            return None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(APP_DIR, "VERSION")
CRASH_LOG = os.path.join(APP_DIR, "crash.log")
REPO = "CoderKavin/AirPoint"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"

# Asset filenames produced by .github/workflows/build.yml. Must stay in sync.
MAC_ASSET_NAME = "AirPoint.dmg"
WIN_ASSET_NAME = "AirPoint-Windows.zip"

# User data that must NEVER be overwritten by an update.
PROTECTED_DIRS = {"profiles", "venv", "__pycache__", ".git"}
# unins000.* are the Inno Setup uninstaller; keep them across auto-updates so
# "Uninstall" via Add/Remove Programs keeps working after an in-app update.
PROTECTED_FILES = {".gitignore", "crash.log", ".vc_installed",
                   "unins000.exe", "unins000.dat"}


# ---------- Version helpers ----------

def get_local_version():
    try:
        with open(VERSION_FILE, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0"


def parse_version(v):
    """Parse 'v1.2.3' or '1.2.3' into a tuple (1, 2, 3)."""
    v = v.lstrip("vV").strip()
    return tuple(int(p) for p in v.split("."))


# ---------- GitHub release fetch ----------

def fetch_latest_release():
    """Return the latest release JSON dict, or None on any failure."""
    try:
        req = urllib.request.Request(
            API_URL, headers={"Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError,
            json.JSONDecodeError, OSError):
        return None


def find_asset_url(release_json, asset_name):
    """Find a release asset by exact filename. Returns URL or None."""
    for asset in release_json.get("assets", []) or []:
        if asset.get("name") == asset_name:
            return asset.get("browser_download_url")
    return None


def download_file(url, dest_path):
    """Download a URL to a local path. Returns True on success.

    Uses urlopen (not urlretrieve) so we can pass the certifi-backed SSL
    context - same macOS cert issue as fetch_latest_release."""
    try:
        with urllib.request.urlopen(url, timeout=120, context=_ssl_context()) as resp, \
                open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


# ---------- UI: update prompt + progress ----------

def _prompt_user_to_update(remote_tag, local_tag):
    """Tk dialog. Returns True iff the user clicked Yes."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        answer = messagebox.askyesno(
            "AirPoint Update Available",
            f"A new version of AirPoint is available.\n\n"
            f"  Installed:  {local_tag}\n"
            f"  Available:  {remote_tag}\n\n"
            f"Install now? AirPoint will close and reopen automatically."
        )
        root.destroy()
        return bool(answer)
    except Exception:
        return False


def _show_installing_message():
    """Returns the tk root so caller can destroy() it after work is done."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.title("AirPoint")
        root.geometry("360x100")
        root.resizable(False, False)
        tk.Label(
            root,
            text="Installing update...\nAirPoint will restart in a moment.",
            pady=20,
        ).pack()
        root.update()
        return root
    except Exception:
        return None


# ---------- Platform-specific swap ----------

def _spawn_detached(cmd):
    """Spawn a subprocess fully detached so it survives our exit."""
    if sys.platform == "win32":
        DETACHED = 0x00000008  # DETACHED_PROCESS
        NEW_GROUP = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            cmd,
            creationflags=DETACHED | NEW_GROUP,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    else:
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _apply_update_macos(dmg_path):
    """Mount DMG, copy AirPoint.app to temp, spawn swap script."""
    # Resolve the actually-running .app bundle path from sys.executable
    # (handles the case where the user renamed the .app after install).
    target_app = None
    cursor = sys.executable
    while cursor and cursor != "/":
        if cursor.endswith(".app"):
            target_app = cursor
            break
        cursor = os.path.dirname(cursor)
    if not target_app:
        # Fallback: assume default name next to APP_DIR.
        target_app = os.path.join(APP_DIR, "AirPoint.app")

    mount_dir = tempfile.mkdtemp(prefix="airpoint_dmg_")
    try:
        result = subprocess.run(
            ["hdiutil", "attach", "-nobrowse", "-noautoopen",
             "-mountpoint", mount_dir, dmg_path],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            return False
    except (subprocess.SubprocessError, OSError):
        return False

    new_app_temp = tempfile.mkdtemp(prefix="airpoint_new_")
    new_app = os.path.join(new_app_temp, "AirPoint.app")

    try:
        # Find AirPoint.app inside the mounted DMG (may be at root or nested).
        src_app = os.path.join(mount_dir, "AirPoint.app")
        if not os.path.isdir(src_app):
            return False
        shutil.copytree(src_app, new_app, symlinks=True)
    except (OSError, shutil.Error):
        subprocess.run(["hdiutil", "detach", mount_dir, "-force"],
                       capture_output=True)
        shutil.rmtree(new_app_temp, ignore_errors=True)
        return False

    subprocess.run(["hdiutil", "detach", mount_dir, "-force"],
                   capture_output=True)

    swap_script = os.path.join(tempfile.gettempdir(), "airpoint_swap.sh")
    with open(swap_script, "w") as f:
        f.write(
            "#!/bin/bash\n"
            "# Wait for the parent AirPoint to fully exit.\n"
            "sleep 2\n"
            f'rm -rf "{target_app}"\n'
            f'mv "{new_app}" "{target_app}"\n'
            # Strip the quarantine bit so Gatekeeper doesn't re-prompt.
            f'xattr -dr com.apple.quarantine "{target_app}" 2>/dev/null || true\n'
            f'open -a "{target_app}"\n'
            f'rm -rf "{new_app_temp}"\n'
            f'rm -- "$0"\n'
        )
    os.chmod(swap_script, 0o755)

    _spawn_detached(["/bin/bash", swap_script])
    return True


def _apply_update_windows(zip_path):
    """Extract ZIP to temp, copy user data over, spawn swap .bat."""
    install_dir = APP_DIR  # Folder containing AirPoint.exe.
    exe_path = os.path.join(install_dir, "AirPoint.exe")

    new_dir = tempfile.mkdtemp(prefix="airpoint_new_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(new_dir)
    except (zipfile.BadZipFile, OSError):
        shutil.rmtree(new_dir, ignore_errors=True)
        return False

    # The zip may extract its contents at the root or inside one subfolder.
    contents = [c for c in os.listdir(new_dir) if not c.startswith(".")]
    if len(contents) == 1 and os.path.isdir(os.path.join(new_dir, contents[0])):
        source = os.path.join(new_dir, contents[0])
    else:
        source = new_dir

    # Migrate protected user data from old install -> new install.
    for d in PROTECTED_DIRS:
        old = os.path.join(install_dir, d)
        if os.path.isdir(old):
            new = os.path.join(source, d)
            if os.path.exists(new):
                shutil.rmtree(new, ignore_errors=True)
            try:
                shutil.copytree(old, new)
            except (OSError, shutil.Error):
                pass

    for f in PROTECTED_FILES:
        old = os.path.join(install_dir, f)
        if os.path.isfile(old):
            try:
                shutil.copy2(old, os.path.join(source, f))
            except OSError:
                pass

    swap_bat = os.path.join(tempfile.gettempdir(), "airpoint_swap.bat")
    with open(swap_bat, "w") as f:
        # Wait for the parent to exit, then MIRROR the new build over the install
        # in place with robocopy. The old approach (rmdir the install dir, then
        # move the new one in) deleted the install FIRST, so any failure of the
        # move - a locked file, or %TEMP% on a different volume - left the folder
        # gone and produced "The system cannot find the path specified", wrecking
        # the install. robocopy /MIR updates in place and retries locked files;
        # protected user data was already copied into {source}, so /MIR keeps it.
        f.write(
            "@echo off\r\n"
            "timeout /t 3 /nobreak >nul\r\n"
            f'robocopy "{source}" "{install_dir}" /MIR /R:5 /W:2 '
            "/NFL /NDL /NJH /NJS /NP >nul\r\n"
            f'start "" "{exe_path}"\r\n'
            f'rmdir /s /q "{new_dir}" 2>nul\r\n'
            'del "%~f0"\r\n'
        )

    _spawn_detached(["cmd.exe", "/c", swap_bat])
    return True


# ---------- Main entry: perform update check ----------

def perform_update_check():
    """Run the full update flow. Returns True iff an update was applied
    (caller must SystemExit so the swap script can finish replacing files).
    """
    # Only run for installed (frozen) builds. Source runs are dev mode.
    if not getattr(sys, "frozen", False):
        return False

    if sys.platform == "darwin":
        asset_name = MAC_ASSET_NAME
        applier = _apply_update_macos
        suffix = ".dmg"
    elif sys.platform == "win32":
        asset_name = WIN_ASSET_NAME
        applier = _apply_update_windows
        suffix = ".zip"
    else:
        return False  # Linux not supported.

    release = fetch_latest_release()
    if not release:
        return False

    remote_tag = release.get("tag_name", "")
    if not remote_tag:
        return False

    try:
        remote_ver = parse_version(remote_tag)
        local_ver = parse_version(get_local_version())
    except (ValueError, IndexError):
        return False

    if remote_ver <= local_ver:
        return False  # Already up to date.

    asset_url = find_asset_url(release, asset_name)
    if not asset_url:
        return False

    if not _prompt_user_to_update(remote_tag, get_local_version()):
        return False

    progress_root = _show_installing_message()
    try:
        download_path = os.path.join(
            tempfile.gettempdir(), f"airpoint_update{suffix}"
        )
        if not download_file(asset_url, download_path):
            return False

        return applier(download_path)
    finally:
        if progress_root is not None:
            try:
                progress_root.destroy()
            except Exception:
                pass


# ---------- Legacy standalone entry (kept for old autostart shims) ----------

def main():
    """Plain launcher: just exec main.py. Source-run path."""
    main_py = os.path.join(APP_DIR, "main.py")
    if not os.path.exists(main_py):
        print("Error: main.py not found!")
        sys.exit(1)
    extra_args = [a for a in sys.argv[1:] if a != "--skip-update"]
    os.execv(sys.executable, [sys.executable, main_py] + extra_args)


def _show_launcher_crash(exc_type, exc_value, exc_tb):
    """Write crash log and show a Tk dialog if anything blew up here."""
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
