# ---------- Production mode: suppress warnings before any imports ----------
import sys
import subprocess
FROZEN = getattr(sys, 'frozen', False)
if FROZEN:
    import os as _os
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # silence TensorFlow
    _os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    _os.environ["GLOG_minloglevel"] = "3"                # silence glog (MediaPipe)
    import warnings
    warnings.filterwarnings("ignore")

# MediaPipe MUST be imported before cv2: in a frozen (PyInstaller) build its
# _framework_bindings native module fails to initialize if OpenCV's native
# DLLs are loaded into the process first. See airpoint_entry.run_app().
import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import time
import math
import json
import os
import copy
import argparse
import traceback
import logging
from datetime import datetime
from collections import deque

# Windows DPI awareness — must be set before any GUI calls so pyautogui
# coordinates match the actual screen resolution on high-DPI displays.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

if FROZEN:
    logging.disable(logging.CRITICAL)  # silence all Python logging
    # Help PyQt5 find its platform plugins inside the PyInstaller bundle
    for _candidate in [
        os.path.join(sys._MEIPASS, "PyQt5", "Qt5", "plugins"),
        os.path.join(sys._MEIPASS, "PyQt5", "Qt", "plugins"),
        os.path.join(sys._MEIPASS, "qt5_plugins"),
    ]:
        if os.path.isdir(_candidate):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _candidate
            break

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QLineEdit, QListWidget,
                              QStackedWidget, QProgressBar, QSizePolicy,
                              QCheckBox, QSlider, QInputDialog, QMessageBox,
                              QListWidgetItem, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QEventLoop, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen

# APP_DIR: when frozen, use the folder containing the exe, not the temp bundle dir
if FROZEN:
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(APP_DIR, "profiles")
CRASH_LOG = os.path.join(APP_DIR, "crash.log")

# ---------- Crash logging ----------

def _write_crash_log(exc_type, exc_value, exc_tb):
    """Append crash details to crash.log with timestamp."""
    try:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        entry = (
            f"\n{'='*60}\n"
            f"CRASH  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*60}\n"
            f"{tb_text}\n"
        )
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass  # never let logging itself crash


def show_crash_dialog(exc_type, exc_value, exc_tb):
    """Show a user-friendly PyQt5 error dialog and log the crash."""
    _write_crash_log(exc_type, exc_value, exc_tb)
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        apply_app_theme(app)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(S("crash_title"))
        msg.setText(S("crash_msg"))
        short = f"{exc_type.__name__}: {exc_value}"
        if len(short) > 200:
            short = short[:200] + "..."
        msg.setInformativeText(short)
        msg.setDetailedText("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        msg.setStyleSheet(f"""
            QMessageBox {{ background-color: {T.bg}; color: {T.text}; }}
            QLabel {{ color: {T.text}; font-size: 13px; }}
            QPushButton {{ background-color: {T.surface}; color: {T.text}; border: 1px solid {T.border};
                          border-radius: 6px; padding: 6px 18px; min-width: 80px; }}
            QPushButton:hover {{ background-color: {T.surface_hover}; }}
            QTextEdit {{ background-color: {T.camera_bg}; color: {T.text2}; font-family: monospace; font-size: 11px; }}
        """)
        msg.exec_()
    except Exception:
        # If Qt itself is broken, at least print to stderr
        traceback.print_exception(exc_type, exc_value, exc_tb)

# Single source of truth for all configurable values.
# Every profile JSON follows this schema; missing keys fall back to these defaults.
DEFAULT_CONFIG = {
    "schema_version": 1,
    "calibration": None,
    "sensitivity": 2.5,
    "smoothing_factor": 0.65,
    "thresholds": {
        "pinch_threshold": 0.05,
        "fist_threshold": 0.06,
        "drag_threshold": 0.4,
        "action_cooldown": 0.15,
        "scroll_dead_zone": 0.015,
        "scroll_threshold": 0.035,
        "scroll_amount": 2,
        "screen_edge_margin": 20,
        "cursor_dead_zone": 10,
        "calibration_margin": 0.05,
    },
    "dwell_click": {
        "enabled": False,
        "radius": 30,
        "duration": 1.5,
    },
    "gaze_detection_enabled": True,
    "click_feedback": True,
    # Per-gesture actions. pinch + fist are user-remappable to discrete clicks
    # (see _do_action); the others are structural (drag / scroll / move) and fixed.
    "gesture_actions": {
        "pinch": "left_click",
        "pinch_hold": "drag",
        "fist": "right_click",
        "two_finger_scroll": "scroll",
        "open_hand": "cursor_move",
    },
}

# Actions a remappable gesture (pinch / fist) can be bound to, with UI labels.
ACTION_LABELS = [
    ("left_click", "Left click"),
    ("right_click", "Right click"),
    ("double_click", "Double click"),
    ("middle_click", "Middle click"),
    ("none", "Nothing"),
]

# One-tap "pointer feel" presets. Each bundles the 7 comfort knobs the Settings
# panel also exposes as sliders. Selecting a preset applies these live and the
# Settings sliders move to match; nudging any slider puts you in "Custom".
#   sensitivity      higher = bigger cursor jump per hand move (uncalibrated gain)
#   smoothing_factor higher = steadier but laggier (EMA alpha)
#   cursor_dead_zone higher = more resting-tremor suppression, coarser fine aim
#   drag_threshold   seconds of held pinch before a drag starts
#   action_cooldown  min seconds between discrete clicks (higher = fewer repeats)
#   scroll_amount    scroll clicks per step (higher = faster)
#   dwell_duration   seconds of stillness before an auto (hover) click
PRESETS = {
    "precise":     {"sensitivity": 1.8, "smoothing_factor": 0.80, "cursor_dead_zone": 16,
                    "drag_threshold": 0.50, "action_cooldown": 0.25, "scroll_amount": 1, "dwell_duration": 2.0},
    "balanced":    {"sensitivity": 2.5, "smoothing_factor": 0.65, "cursor_dead_zone": 10,
                    "drag_threshold": 0.40, "action_cooldown": 0.15, "scroll_amount": 2, "dwell_duration": 1.5},
    "fast":        {"sensitivity": 3.5, "smoothing_factor": 0.45, "cursor_dead_zone": 6,
                    "drag_threshold": 0.25, "action_cooldown": 0.12, "scroll_amount": 3, "dwell_duration": 1.2},
    "high_tremor": {"sensitivity": 1.5, "smoothing_factor": 0.88, "cursor_dead_zone": 24,
                    "drag_threshold": 0.60, "action_cooldown": 0.60, "scroll_amount": 1, "dwell_duration": 2.5},
}
PRESET_LABELS = [("precise", "Precise"), ("balanced", "Balanced"),
                 ("fast", "Fast"), ("high_tremor", "High tremor")]

# ---------- Autostart ----------

def set_autostart(enabled):
    """Enable or disable AirPoint starting when the computer turns on.
    Windows: creates/removes a shortcut in the Startup folder.
    macOS: creates/removes a LaunchAgent plist.
    """
    launcher = os.path.join(APP_DIR, "launcher.py")
    if not os.path.exists(launcher):
        launcher = os.path.join(APP_DIR, "main.py")

    if sys.platform == "win32":
        try:
            startup_dir = os.path.join(
                os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu",
                "Programs", "Startup"
            )
            bat_path = os.path.join(startup_dir, "AirPoint.bat")
            if enabled:
                python = sys.executable
                with open(bat_path, "w") as f:
                    f.write(f'@echo off\nstart "" /min "{python}" "{launcher}"\n')
            else:
                if os.path.exists(bat_path):
                    os.remove(bat_path)
        except Exception as e:
            _write_crash_log(type(e), e, e.__traceback__)

    elif sys.platform == "darwin":
        try:
            plist_dir = os.path.expanduser("~/Library/LaunchAgents")
            os.makedirs(plist_dir, exist_ok=True)
            plist_path = os.path.join(plist_dir, "com.chetana.airpoint.plist")
            if enabled:
                python = sys.executable
                plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chetana.airpoint</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{launcher}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/airpoint.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/airpoint.err</string>
</dict>
</plist>
"""
                with open(plist_path, "w") as f:
                    f.write(plist_content)
            else:
                if os.path.exists(plist_path):
                    os.remove(plist_path)
        except Exception as e:
            _write_crash_log(type(e), e, e.__traceback__)


def get_autostart_enabled():
    """Check whether autostart is currently configured."""
    if sys.platform == "win32":
        bat_path = os.path.join(
            os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu",
            "Programs", "Startup", "AirPoint.bat"
        )
        return os.path.exists(bat_path)
    elif sys.platform == "darwin":
        plist_path = os.path.expanduser("~/Library/LaunchAgents/com.chetana.airpoint.plist")
        return os.path.exists(plist_path)
    return False


# ---------- Internationalisation (i18n) ----------

STRINGS = {
    "en": {
        # Language selector
        "lang_title": "Choose your language",
        "lang_subtitle": "You can change this later.",
        "lang_english": "English",
        "lang_hindi": "हिन्दी (Hindi)",
        "lang_malayalam": "മലയാളം (Malayalam)",
        "lang_tamil": "தமிழ் (Tamil)",
        # Profile selector
        "profile_title": "Welcome Back",
        "profile_subtitle": "Choose your profile to get started",
        "profile_new": "+ New Profile",
        "profile_select": "Select",
        "quit": "Quit",
        # Name entry
        "name_title": "What's your name?",
        "name_subtitle": "We'll create a personal profile for you",
        "name_placeholder": "Type your name here...",
        "continue": "Continue",
        # Welcome / How it works
        "welcome_hi": "Hi, {name}!",
        "welcome_default": "Hi there!",
        "welcome_sub": "AirPoint lets you control your computer\nusing just your hand and a camera.",
        "how_title": "How it works",
        "feat_move_title": "Move the cursor",
        "feat_move_desc": "Hold your hand open in front of the camera.\nMove your hand around — the cursor follows.",
        "feat_click_title": "Click",
        "feat_click_desc": "Pinch your thumb and the finger next to it\ntogether. Like tapping a button in the air.",
        "feat_right_title": "Right-click",
        "feat_right_desc": "Make a fist to right-click.\nThis opens menus, just like a normal mouse.",
        "feat_scroll_title": "Scroll",
        "feat_scroll_desc": "Hold up two fingers next to each other and\nmove them up or down to scroll a page.",
        "setup_note": "First, we need a quick 30-second setup so AirPoint\ncan learn how you move your hand.",
        "lets_go": "Let's Go!",
        "skip_setup": "Skip Setup",
        # Calibration
        "cal_step1_title": "Step 1 of 4 — We need to know how far you can reach\nso the cursor covers your whole screen.",
        "cal_step2_title": "Step 2 of 4 — We're checking how steady your hand is\nso we can reduce any shaking.",
        "cal_step3_title": "Step 3 of 4 — This is how you'll click things\nPinching is like tapping a button in the air.",
        "cal_step4_title": "Step 4 of 4 — This is how you'll right-click\nMaking a fist opens menus.",
        "cal_move_left": "Move your hand to the LEFT",
        "cal_move_right": "Now move your hand to the RIGHT",
        "cal_move_up": "Move your hand UP",
        "cal_move_down": "Move your hand DOWN",
        "cal_hint_dir": "Press the spacebar when your hand is as far {dir} as comfortable",
        "cal_hint_steady": "Just wait — the camera is watching your hand",
        "cal_steady_inst": "Hold your hand still and relax",
        "cal_pinch_inst": "Pinch your thumb and the finger next to it together — and keep holding it",
        "cal_fist_inst": "Make a fist — close all your fingers and keep holding it",
        "cal_hint_gesture": "While holding the gesture, tap spacebar to record  (or press N to skip)",
        "cal_hold_still": "Hold still...",
        "cal_hand_lost": "Hand lost — try again",
        "cal_show_hand": "Show your hand to begin",
        "cal_recording": "Recording — keep holding the gesture until the bar fills",
        "cal_gesture_retry": "Didn't catch that — make the gesture and tap spacebar again",
        # Done page
        "done_title": "You're all set!",
        "done_subtitle": "AirPoint is ready. Here's a quick reminder:",
        "done_gesture_open": "Open hand",
        "done_gesture_pinch": "Pinch",
        "done_gesture_fist": "Fist",
        "done_gesture_scroll": "Two fingers up or down",
        "done_action_move": "Move cursor",
        "done_action_click": "Click",
        "done_action_right": "Right-click",
        "done_action_scroll": "Scroll",
        "done_extras_title": "You can also turn these on later:",
        "done_extras_gaze": (
            "<b style='color:#ccc;'>Pause when not looking</b> "
            "<span style='color:#888;'>— AirPoint pauses if you look away "
            "from the screen, so it won't move the cursor by accident.</span>"
        ),
        "done_extras_dwell": (
            "<b style='color:#ccc;'>Auto-click</b> "
            "<span style='color:#888;'>— If you hold your hand still over something "
            "for a moment, AirPoint clicks it for you automatically.</span>"
        ),
        "autostart_label": "Start AirPoint when computer turns on",
        "start_airpoint": "Start AirPoint",
        # Status panel
        "panel_hi": "Hi, {name}",
        "panel_looking": "Looking for your hand...",
        "panel_moving": "Moving cursor",
        "panel_clicked": "Clicked!",
        "panel_dragging": "Dragging...",
        "panel_drag_done": "Done dragging",
        "panel_right_clicked": "Right-clicked!",
        "panel_scrolling": "Scrolling",
        "panel_pinch_drag": "Keep pinching to drag...",
        "panel_auto_clicked": "Auto-clicked!",
        "panel_look_screen": "Look at screen to start",
        "panel_ready": "Ready — move your hand",
        "panel_gaze_on": "  Pause when not looking  —  ON\n  AirPoint pauses if you look away from the screen",
        "panel_gaze_off": "  Pause when not looking  —  OFF\n  Tap here to turn this on",
        "panel_dwell_on": "  Auto-click when you hold still  —  ON\n  Clicks for you after staying in one spot",
        "panel_dwell_off": "  Auto-click when you hold still  —  OFF\n  Tap here to turn this on",
        "panel_redo": "Redo Setup",
        "panel_stop": "Stop AirPoint",
        # Crash
        "crash_title": "AirPoint — Something went wrong",
        "crash_msg": "AirPoint ran into an unexpected error and needs to close.\n\nYour profiles and settings are safe.",
    },
    "hi": {
        # Language selector
        "lang_title": "अपनी भाषा चुनें",
        "lang_subtitle": "आप इसे बाद में बदल सकते हैं।",
        "lang_english": "English",
        "lang_hindi": "हिन्दी (Hindi)",
        "lang_malayalam": "മലയാളം (Malayalam)",
        "lang_tamil": "தமிழ் (Tamil)",
        # Profile selector
        "profile_title": "फिर से स्वागत है",
        "profile_subtitle": "शुरू करने के लिए अपना प्रोफ़ाइल चुनें",
        "profile_new": "+ नया प्रोफ़ाइल",
        "profile_select": "चुनें",
        "quit": "बंद करें",
        # Name entry
        "name_title": "आपका नाम क्या है?",
        "name_subtitle": "हम आपके लिए एक प्रोफ़ाइल बनाएँगे",
        "name_placeholder": "यहाँ अपना नाम लिखें...",
        "continue": "आगे बढ़ें",
        # Welcome / How it works
        "welcome_hi": "नमस्ते, {name}!",
        "welcome_default": "नमस्ते!",
        "welcome_sub": "AirPoint आपको सिर्फ़ अपने हाथ और\nकैमरे से कंप्यूटर चलाने देता है।",
        "how_title": "यह कैसे काम करता है",
        "feat_move_title": "कर्सर हिलाएँ",
        "feat_move_desc": "कैमरे के सामने हाथ खोलकर रखें।\nहाथ हिलाएँ — कर्सर भी हिलेगा।",
        "feat_click_title": "क्लिक करें",
        "feat_click_desc": "अंगूठे और उसके बगल वाली उँगली को\nजोड़ें। हवा में बटन दबाने जैसा है।",
        "feat_right_title": "राइट-क्लिक",
        "feat_right_desc": "मुट्ठी बंद करें — राइट-क्लिक होगा।\nइससे मेनू खुलता है।",
        "feat_scroll_title": "स्क्रॉल करें",
        "feat_scroll_desc": "दो उँगलियाँ ऊपर उठाएँ और\nऊपर-नीचे हिलाकर पेज स्क्रॉल करें।",
        "setup_note": "पहले, एक छोटा-सा 30 सेकंड का सेटअप चाहिए\nताकि AirPoint आपके हाथ की गति सीख सके।",
        "lets_go": "चलो शुरू करें!",
        "skip_setup": "सेटअप छोड़ें",
        # Calibration
        "cal_step1_title": "चरण 1 / 4 — हमें जानना है कि आप कितनी दूर तक पहुँच सकते हैं\nताकि कर्सर पूरी स्क्रीन पर चले।",
        "cal_step2_title": "चरण 2 / 4 — हम देख रहे हैं कि आपका हाथ कितना स्थिर है\nताकि हम कंपन कम कर सकें।",
        "cal_step3_title": "चरण 3 / 4 — इससे आप क्लिक करेंगे\nचुटकी बजाना = हवा में बटन दबाना।",
        "cal_step4_title": "चरण 4 / 4 — इससे आप राइट-क्लिक करेंगे\nमुट्ठी बंद करने से मेनू खुलता है।",
        "cal_move_left": "अपना हाथ बाईं ओर ले जाएँ",
        "cal_move_right": "अब दाईं ओर ले जाएँ",
        "cal_move_up": "अपना हाथ ऊपर ले जाएँ",
        "cal_move_down": "अपना हाथ नीचे ले जाएँ",
        "cal_hint_dir": "जब हाथ {dir} तक पहुँच जाए तो स्पेसबार दबाएँ",
        "cal_hint_steady": "बस रुकें — कैमरा आपका हाथ देख रहा है",
        "cal_steady_inst": "अपना हाथ स्थिर रखें और आराम करें",
        "cal_pinch_inst": "अंगूठे और बगल वाली उँगली को जोड़ें — और ऐसे ही पकड़े रहें",
        "cal_fist_inst": "मुट्ठी बंद करें — सारी उँगलियाँ बंद रखें और पकड़े रहें",
        "cal_hint_gesture": "इशारा बनाए रखते हुए स्पेसबार दबाएँ रिकॉर्ड के लिए  (या N दबाएँ छोड़ने के लिए)",
        "cal_hold_still": "स्थिर रहें...",
        "cal_hand_lost": "हाथ नहीं दिखा — फिर कोशिश करें",
        "cal_show_hand": "अपना हाथ दिखाएँ",
        "cal_recording": "रिकॉर्ड हो रहा है — बार भरने तक इशारा ऐसे ही रखें",
        "cal_gesture_retry": "पकड़ नहीं पाया — इशारा बनाकर फिर स्पेसबार दबाएँ",
        # Done page
        "done_title": "सब तैयार है!",
        "done_subtitle": "AirPoint तैयार है। याद रखें:",
        "done_gesture_open": "हाथ खोलें",
        "done_gesture_pinch": "चुटकी",
        "done_gesture_fist": "मुट्ठी",
        "done_gesture_scroll": "दो उँगलियाँ ऊपर या नीचे",
        "done_action_move": "कर्सर हिलाएँ",
        "done_action_click": "क्लिक",
        "done_action_right": "राइट-क्लिक",
        "done_action_scroll": "स्क्रॉल",
        "done_extras_title": "ये सुविधाएँ बाद में भी चालू कर सकते हैं:",
        "done_extras_gaze": (
            "<b style='color:#ccc;'>न देखने पर रुकें</b> "
            "<span style='color:#888;'>— अगर आप स्क्रीन से नज़र हटाएँ "
            "तो AirPoint रुक जाएगा, ताकि गलती से कर्सर न हिले।</span>"
        ),
        "done_extras_dwell": (
            "<b style='color:#ccc;'>अपने-आप क्लिक</b> "
            "<span style='color:#888;'>— अगर आप हाथ को किसी चीज़ पर "
            "रोककर रखें, तो AirPoint अपने-आप क्लिक कर देगा।</span>"
        ),
        "autostart_label": "कंप्यूटर चालू होने पर AirPoint शुरू करें",
        "start_airpoint": "AirPoint शुरू करें",
        # Status panel
        "panel_hi": "नमस्ते, {name}",
        "panel_looking": "आपका हाथ खोज रहा है...",
        "panel_moving": "कर्सर हिल रहा है",
        "panel_clicked": "क्लिक हुआ!",
        "panel_dragging": "खींच रहा है...",
        "panel_drag_done": "खींचना पूरा",
        "panel_right_clicked": "राइट-क्लिक हुआ!",
        "panel_scrolling": "स्क्रॉल हो रहा है",
        "panel_pinch_drag": "खींचने के लिए चुटकी पकड़े रखें...",
        "panel_auto_clicked": "अपने-आप क्लिक हुआ!",
        "panel_look_screen": "शुरू करने के लिए स्क्रीन देखें",
        "panel_ready": "तैयार — हाथ हिलाएँ",
        "panel_gaze_on": "  न देखने पर रुकें  —  चालू\n  स्क्रीन से नज़र हटाने पर AirPoint रुकेगा",
        "panel_gaze_off": "  न देखने पर रुकें  —  बंद\n  चालू करने के लिए यहाँ दबाएँ",
        "panel_dwell_on": "  रुकने पर अपने-आप क्लिक  —  चालू\n  एक जगह रुकने पर क्लिक होगा",
        "panel_dwell_off": "  रुकने पर अपने-आप क्लिक  —  बंद\n  चालू करने के लिए यहाँ दबाएँ",
        "panel_redo": "फिर से सेटअप करें",
        "panel_stop": "AirPoint बंद करें",
        # Crash
        "crash_title": "AirPoint — कुछ गड़बड़ हो गई",
        "crash_msg": "AirPoint में कोई समस्या आई और इसे बंद करना पड़ा।\n\nआपकी प्रोफ़ाइल और सेटिंग्स सुरक्षित हैं।",
    },
    "ml": {
        "lang_title": "നിങ്ങളുടെ ഭാഷ തിരഞ്ഞെടുക്കൂ",
        "lang_subtitle": "ഇത് പിന്നീട് മാറ്റാം.",
        "lang_english": "English",
        "lang_hindi": "हिन्दी (Hindi)",
        "lang_malayalam": "മലയാളം (Malayalam)",
        "lang_tamil": "தமிழ் (Tamil)",
        "profile_title": "വീണ്ടും സ്വാഗതം",
        "profile_subtitle": "തുടങ്ങാൻ നിങ്ങളുടെ പ്രൊഫൈൽ തിരഞ്ഞെടുക്കൂ",
        "profile_new": "+ പുതിയ പ്രൊഫൈൽ",
        "profile_select": "തിരഞ്ഞെടുക്കൂ",
        "quit": "പുറത്തുകടക്കൂ",
        "name_title": "നിങ്ങളുടെ പേരെന്താണ്?",
        "name_subtitle": "നിങ്ങൾക്കായി ഒരു പ്രൊഫൈൽ ഉണ്ടാക്കാം",
        "name_placeholder": "ഇവിടെ പേര് ടൈപ്പ് ചെയ്യൂ...",
        "continue": "തുടരുക",
        "welcome_hi": "ഹായ്, {name}!",
        "welcome_default": "ഹായ്!",
        "welcome_sub": "കൈയും ക്യാമറയും കൊണ്ട് മാത്രം\nനിങ്ങളുടെ കമ്പ്യൂട്ടർ നിയന്ത്രിക്കാം.",
        "how_title": "ഇത് എങ്ങനെ പ്രവർത്തിക്കുന്നു",
        "feat_move_title": "കഴ്സർ നീക്കുക",
        "feat_move_desc": "ക്യാമറയ്ക്ക് മുന്നിൽ കൈ തുറന്നു പിടിക്കൂ.\nകൈ ചലിപ്പിക്കൂ — കഴ്സർ പിന്തുടരും.",
        "feat_click_title": "ക്ലിക്ക്",
        "feat_click_desc": "തള്ളവിരലും അടുത്ത വിരലും\nചേർത്ത് നുള്ളൂ. വായുവിൽ ബട്ടൺ അമർത്തുംപോലെ.",
        "feat_right_title": "റൈറ്റ്-ക്ലിക്ക്",
        "feat_right_desc": "റൈറ്റ്-ക്ലിക്കിന് മുഷ്ടി ചുരുട്ടൂ.\nസാധാരണ മൗസ് പോലെ മെനു തുറക്കും.",
        "feat_scroll_title": "സ്ക്രോൾ",
        "feat_scroll_desc": "രണ്ട് വിരലുകൾ ചേർത്ത് ഉയർത്തി\nമുകളിലോ താഴെയോ നീക്കി പേജ് സ്ക്രോൾ ചെയ്യൂ.",
        "setup_note": "നിങ്ങളുടെ കൈ ചലനം പഠിക്കാൻ AirPoint-ന്\nആദ്യം 30 സെക്കൻഡ് സജ്ജീകരണം വേണം.",
        "lets_go": "തുടങ്ങാം!",
        "skip_setup": "സജ്ജീകരണം ഒഴിവാക്കൂ",
        "cal_step1_title": "ഘട്ടം 1 / 4 — കഴ്സർ സ്ക്രീൻ മുഴുവൻ എത്താൻ\nനിങ്ങൾ എത്രദൂരം എത്തുമെന്ന് അറിയണം.",
        "cal_step2_title": "ഘട്ടം 2 / 4 — കുലുക്കം കുറയ്ക്കാൻ നിങ്ങളുടെ കൈ\nഎത്ര സ്ഥിരമാണെന്ന് പരിശോധിക്കുന്നു.",
        "cal_step3_title": "ഘട്ടം 3 / 4 — ഇങ്ങനെ ക്ലിക്ക് ചെയ്യാം\nനുള്ളൽ വായുവിൽ ബട്ടൺ അമർത്തുംപോലെ.",
        "cal_step4_title": "ഘട്ടം 4 / 4 — ഇങ്ങനെ റൈറ്റ്-ക്ലിക്ക് ചെയ്യാം\nമുഷ്ടി ചുരുട്ടിയാൽ മെനു തുറക്കും.",
        "cal_move_left": "കൈ ഇടത്തേക്ക് നീക്കൂ",
        "cal_move_right": "ഇനി കൈ വലത്തേക്ക് നീക്കൂ",
        "cal_move_up": "കൈ മുകളിലേക്ക് നീക്കൂ",
        "cal_move_down": "കൈ താഴേക്ക് നീക്കൂ",
        "cal_hint_dir": "സുഖമായി {dir} ഭാഗത്തേക്ക് കൈ എത്തുമ്പോൾ സ്പെയ്സ്ബാർ അമർത്തൂ",
        "cal_hint_steady": "കാത്തിരിക്കൂ — ക്യാമറ നിങ്ങളുടെ കൈ നോക്കുന്നു",
        "cal_steady_inst": "കൈ അനക്കാതെ ശാന്തമായി പിടിക്കൂ",
        "cal_pinch_inst": "തള്ളവിരലും അടുത്ത വിരലും ചേർത്ത് നുള്ളൂ — അങ്ങനെ പിടിച്ചുനിർത്തൂ",
        "cal_fist_inst": "മുഷ്ടി ചുരുട്ടൂ — എല്ലാ വിരലുകളും അടച്ച് പിടിച്ചുനിർത്തൂ",
        "cal_hint_gesture": "ആംഗ്യം പിടിച്ചുകൊണ്ട് റെക്കോർഡ് ചെയ്യാൻ സ്പെയ്സ്ബാർ അമർത്തൂ  (ഒഴിവാക്കാൻ N)",
        "cal_hold_still": "അനക്കാതെ പിടിക്കൂ...",
        "cal_hand_lost": "കൈ കണ്ടില്ല — വീണ്ടും ശ്രമിക്കൂ",
        "cal_show_hand": "തുടങ്ങാൻ കൈ കാണിക്കൂ",
        "cal_recording": "റെക്കോർഡ് ചെയ്യുന്നു — ബാർ നിറയുംവരെ ആംഗ്യം പിടിക്കൂ",
        "cal_gesture_retry": "കിട്ടിയില്ല — ആംഗ്യം കാണിച്ച് വീണ്ടും സ്പെയ്സ്ബാർ അമർത്തൂ",
        "done_title": "എല്ലാം തയ്യാർ!",
        "done_subtitle": "AirPoint തയ്യാർ. ഒരു ചെറിയ ഓർമ്മപ്പെടുത്തൽ:",
        "done_gesture_open": "തുറന്ന കൈ",
        "done_gesture_pinch": "നുള്ളൽ",
        "done_gesture_fist": "മുഷ്ടി",
        "done_gesture_scroll": "രണ്ട് വിരൽ മുകളിലോ താഴെയോ",
        "done_action_move": "കഴ്സർ നീക്കുക",
        "done_action_click": "ക്ലിക്ക്",
        "done_action_right": "റൈറ്റ്-ക്ലിക്ക്",
        "done_action_scroll": "സ്ക്രോൾ",
        "done_extras_title": "ഇവയും പിന്നീട് ഓൺ ചെയ്യാം:",
        "done_extras_gaze": "<b style='color:#ccc;'>നോക്കാത്തപ്പോൾ താൽക്കാലികമായി നിർത്തുക</b> <span style='color:#888;'>— സ്ക്രീനിൽ നിന്ന് നോട്ടം മാറ്റിയാൽ AirPoint നിർത്തും, അബദ്ധത്തിൽ കഴ്സർ നീങ്ങില്ല.</span>",
        "done_extras_dwell": "<b style='color:#ccc;'>ഓട്ടോ-ക്ലിക്ക്</b> <span style='color:#888;'>— ഒരു സാധനത്തിന് മുകളിൽ കൈ അനക്കാതെ നിർത്തിയാൽ AirPoint തനിയെ ക്ലിക്ക് ചെയ്യും.</span>",
        "autostart_label": "കമ്പ്യൂട്ടർ ഓണാകുമ്പോൾ AirPoint തുടങ്ങുക",
        "start_airpoint": "AirPoint തുടങ്ങൂ",
        "panel_hi": "ഹായ്, {name}",
        "panel_looking": "നിങ്ങളുടെ കൈ തിരയുന്നു...",
        "panel_moving": "കഴ്സർ നീക്കുന്നു",
        "panel_clicked": "ക്ലിക്ക് ചെയ്തു!",
        "panel_dragging": "വലിച്ചിടുന്നു...",
        "panel_drag_done": "വലിച്ചിടൽ പൂർത്തിയായി",
        "panel_right_clicked": "റൈറ്റ്-ക്ലിക്ക് ചെയ്തു!",
        "panel_scrolling": "സ്ക്രോൾ ചെയ്യുന്നു",
        "panel_pinch_drag": "വലിച്ചിടാൻ നുള്ളിപ്പിടിക്കൂ...",
        "panel_auto_clicked": "തനിയെ ക്ലിക്ക് ചെയ്തു!",
        "panel_look_screen": "തുടങ്ങാൻ സ്ക്രീനിലേക്ക് നോക്കൂ",
        "panel_ready": "തയ്യാർ — കൈ നീക്കൂ",
        "panel_gaze_on": "  നോക്കാത്തപ്പോൾ നിർത്തുക  —  ഓൺ\n  സ്ക്രീനിൽ നിന്ന് നോട്ടം മാറ്റിയാൽ AirPoint നിർത്തും",
        "panel_gaze_off": "  നോക്കാത്തപ്പോൾ നിർത്തുക  —  ഓഫ്\n  ഓൺ ചെയ്യാൻ ഇവിടെ ടാപ്പ് ചെയ്യൂ",
        "panel_dwell_on": "  അനക്കാതെ നിർത്തുമ്പോൾ ഓട്ടോ-ക്ലിക്ക്  —  ഓൺ\n  ഒരിടത്ത് നിന്നാൽ തനിയെ ക്ലിക്ക് ചെയ്യും",
        "panel_dwell_off": "  അനക്കാതെ നിർത്തുമ്പോൾ ഓട്ടോ-ക്ലിക്ക്  —  ഓഫ്\n  ഓൺ ചെയ്യാൻ ഇവിടെ ടാപ്പ് ചെയ്യൂ",
        "panel_redo": "സജ്ജീകരണം വീണ്ടും ചെയ്യൂ",
        "panel_stop": "AirPoint നിർത്തൂ",
        "crash_title": "AirPoint — എന്തോ പിശക് സംഭവിച്ചു",
        "crash_msg": "AirPoint-ന് അപ്രതീക്ഷിത പിശക് സംഭവിച്ചു, അടയ്ക്കേണ്ടതുണ്ട്.\n\nനിങ്ങളുടെ പ്രൊഫൈലുകളും ക്രമീകരണങ്ങളും സുരക്ഷിതമാണ്.",
    },
    "ta": {
        "lang_title": "உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்",
        "lang_subtitle": "இதை பிறகு மாற்றலாம்.",
        "lang_english": "English",
        "lang_hindi": "हिन्दी (Hindi)",
        "lang_malayalam": "മലയാളം (Malayalam)",
        "lang_tamil": "தமிழ் (Tamil)",
        "profile_title": "மீண்டும் வரவேற்கிறோம்",
        "profile_subtitle": "தொடங்க உங்கள் சுயவிவரத்தைத் தேர்ந்தெடுக்கவும்",
        "profile_new": "+ புதிய சுயவிவரம்",
        "profile_select": "தேர்வு",
        "quit": "வெளியேறு",
        "name_title": "உங்கள் பெயர் என்ன?",
        "name_subtitle": "உங்களுக்கான தனிப்பட்ட சுயவிவரத்தை உருவாக்குவோம்",
        "name_placeholder": "உங்கள் பெயரை இங்கே தட்டச்சு செய்யவும்...",
        "continue": "தொடரவும்",
        "welcome_hi": "வணக்கம், {name}!",
        "welcome_default": "வணக்கம்!",
        "welcome_sub": "உங்கள் கையும் ஒரு கேமராவும் மட்டுமே\nகொண்டு கணினியைக் கட்டுப்படுத்தலாம்.",
        "how_title": "இது எப்படி வேலை செய்கிறது",
        "feat_move_title": "சுட்டியை நகர்த்துதல்",
        "feat_move_desc": "கேமரா முன் கையைத் திறந்து வைக்கவும்.\nகையை நகர்த்தினால் சுட்டியும் பின்தொடரும்.",
        "feat_click_title": "கிளிக் செய்தல்",
        "feat_click_desc": "கட்டைவிரலையும் அதன் அருகிலுள்ள விரலையும்\nஇணைக்கவும். காற்றில் பட்டனை அழுத்துவது போல.",
        "feat_right_title": "வலது கிளிக்",
        "feat_right_desc": "வலது கிளிக்கிற்கு கையை முட்டியாக மூடவும்.\nஇது சாதாரண சுட்டி போல மெனுக்களைத் திறக்கும்.",
        "feat_scroll_title": "ஸ்க்ரோல் செய்தல்",
        "feat_scroll_desc": "இரண்டு விரல்களை அருகருகே நிமிர்த்தி\nமேலே அல்லது கீழே நகர்த்தி பக்கத்தை ஸ்க்ரோல் செய்யவும்.",
        "setup_note": "முதலில், உங்கள் கை அசைவை AirPoint அறிய\nஒரு விரைவான 30-வினாடி அமைப்பு தேவை.",
        "lets_go": "தொடங்குவோம்!",
        "skip_setup": "அமைப்பைத் தவிர்",
        "cal_step1_title": "படி 1/4 — சுட்டி உங்கள் முழு திரையையும் சேர\nநீங்கள் எவ்வளவு தூரம் எட்டுவீர்கள் என அறிய வேண்டும்.",
        "cal_step2_title": "படி 2/4 — அசைவைக் குறைக்க உங்கள் கை எவ்வளவு\nஸ்திரமாக உள்ளது என சரிபார்க்கிறோம்.",
        "cal_step3_title": "படி 3/4 — இப்படித்தான் கிளிக் செய்வீர்கள்\nகிள்ளுவது காற்றில் பட்டனை அழுத்துவது போல.",
        "cal_step4_title": "படி 4/4 — இப்படித்தான் வலது கிளிக் செய்வீர்கள்\nமுட்டி பிடிப்பது மெனுக்களைத் திறக்கும்.",
        "cal_move_left": "உங்கள் கையை இடதுபுறம் நகர்த்தவும்",
        "cal_move_right": "இப்போது கையை வலதுபுறம் நகர்த்தவும்",
        "cal_move_up": "உங்கள் கையை மேலே நகர்த்தவும்",
        "cal_move_down": "உங்கள் கையை கீழே நகர்த்தவும்",
        "cal_hint_dir": "கை வசதியாக எட்டும் {dir} எல்லைக்கு வந்ததும் ஸ்பேஸ்பாரை அழுத்தவும்",
        "cal_hint_steady": "காத்திருங்கள் — கேமரா உங்கள் கையைப் பார்க்கிறது",
        "cal_steady_inst": "கையை அசைக்காமல் வைத்து ஓய்வாக இருங்கள்",
        "cal_pinch_inst": "கட்டைவிரலையும் அதன் அருகிலுள்ள விரலையும் இணைத்து அப்படியே வைத்திருங்கள்",
        "cal_fist_inst": "முட்டி பிடிக்கவும் — அனைத்து விரல்களையும் மூடி அப்படியே வைத்திருங்கள்",
        "cal_hint_gesture": "சைகையை வைத்தபடி பதிவுசெய்ய ஸ்பேஸ்பாரை அழுத்தவும்  (தவிர்க்க N)",
        "cal_hold_still": "அசைக்காமல் வைக்கவும்...",
        "cal_hand_lost": "கை தெரியவில்லை — மீண்டும் முயற்சிக்கவும்",
        "cal_show_hand": "தொடங்க உங்கள் கையைக் காட்டவும்",
        "cal_recording": "பதிவாகிறது — பட்டை நிரம்பும் வரை சைகையை வைத்திருங்கள்",
        "cal_gesture_retry": "பிடிபடவில்லை — சைகையைச் செய்து மீண்டும் ஸ்பேஸ்பாரை அழுத்தவும்",
        "done_title": "எல்லாம் தயார்!",
        "done_subtitle": "AirPoint தயார். ஒரு விரைவான நினைவூட்டல்:",
        "done_gesture_open": "திறந்த கை",
        "done_gesture_pinch": "கிள்ளு",
        "done_gesture_fist": "முட்டி",
        "done_gesture_scroll": "இரண்டு விரல்கள் மேலே/கீழே",
        "done_action_move": "சுட்டியை நகர்த்து",
        "done_action_click": "கிளிக்",
        "done_action_right": "வலது கிளிக்",
        "done_action_scroll": "ஸ்க்ரோல்",
        "done_extras_title": "இவற்றையும் பிறகு இயக்கலாம்:",
        "done_extras_gaze": "<b style='color:#ccc;'>பார்க்காதபோது இடைநிறுத்து</b> <span style='color:#888;'>— திரையை விட்டு வேறு பக்கம் பார்த்தால் AirPoint இடைநிறுத்தும், எனவே தற்செயலாக சுட்டி நகராது.</span>",
        "done_extras_dwell": "<b style='color:#ccc;'>தானியங்கி கிளிக்</b> <span style='color:#888;'>— ஒரு பொருளின் மேல் கையை சிறிது நேரம் அசைக்காமல் வைத்தால், AirPoint தானாகவே கிளிக் செய்யும்.</span>",
        "autostart_label": "கணினி இயங்கும்போது AirPoint-ஐ தொடங்கு",
        "start_airpoint": "AirPoint-ஐ தொடங்கு",
        "panel_hi": "வணக்கம், {name}",
        "panel_looking": "உங்கள் கையைத் தேடுகிறது...",
        "panel_moving": "சுட்டி நகர்கிறது",
        "panel_clicked": "கிளிக் ஆனது!",
        "panel_dragging": "இழுக்கிறது...",
        "panel_drag_done": "இழுத்தல் முடிந்தது",
        "panel_right_clicked": "வலது கிளிக் ஆனது!",
        "panel_scrolling": "ஸ்க்ரோல் ஆகிறது",
        "panel_pinch_drag": "இழுக்க கிள்ளியபடி வைத்திருங்கள்...",
        "panel_auto_clicked": "தானாக கிளிக் ஆனது!",
        "panel_look_screen": "தொடங்க திரையைப் பாருங்கள்",
        "panel_ready": "தயார் — உங்கள் கையை நகர்த்தவும்",
        "panel_gaze_on": "  பார்க்காதபோது இடைநிறுத்து  —  இயக்கப்பட்டது\n  திரையை விட்டு பார்த்தால் AirPoint இடைநிறுத்தும்",
        "panel_gaze_off": "  பார்க்காதபோது இடைநிறுத்து  —  நிறுத்தப்பட்டது\n  இதை இயக்க இங்கே தட்டவும்",
        "panel_dwell_on": "  அசைக்காமல் வைத்தால் தானியங்கி கிளிக்  —  இயக்கப்பட்டது\n  ஒரே இடத்தில் நின்றால் உங்களுக்காக கிளிக் செய்யும்",
        "panel_dwell_off": "  அசைக்காமல் வைத்தால் தானியங்கி கிளிக்  —  நிறுத்தப்பட்டது\n  இதை இயக்க இங்கே தட்டவும்",
        "panel_redo": "அமைப்பை மீண்டும் செய்",
        "panel_stop": "AirPoint-ஐ நிறுத்து",
        "crash_title": "AirPoint — ஏதோ தவறு நடந்தது",
        "crash_msg": "AirPoint எதிர்பாராத பிழையை சந்தித்ததால் மூட வேண்டும்.\n\nஉங்கள் சுயவிவரங்களும் அமைப்புகளும் பாதுகாப்பாக உள்ளன.",
    },
}

# Current language — set during wizard, defaults to English
_current_lang = "en"

def S(key, **kwargs):
    """Get a translated string. Usage: S('welcome_hi', name='Kavin')"""
    text = STRINGS.get(_current_lang, STRINGS["en"]).get(key, STRINGS["en"].get(key, key))
    if kwargs:
        text = text.format(**kwargs)
    return text

def set_language(lang):
    global _current_lang
    _current_lang = lang if lang in STRINGS else "en"


# ============================================================
# PyQt5 Setup Wizard (used for profile selection & calibration)
# ============================================================

# ---------------------------------------------------------------------------
# Theme system
#
# AirPoint follows the OS appearance (light / dark) and uses an accent drawn
# from the app logo (a vivid blue with a cyan glow). Styling is centralized:
# `T` holds the resolved palette, `_font()` returns the platform's system UI
# font, and BASE_QSS is the global stylesheet applied to the QApplication.
# Onboarding is the only screen-heavy surface, so this is kept lightweight —
# one palette object, one stylesheet, native window chrome.
# ---------------------------------------------------------------------------
from types import SimpleNamespace

# Accent comes straight from the logo: azure #1C8CEE with a #2ADDFF cyan glow.
_LIGHT = {
    "bg": "#ECECEE", "card": "#FFFFFF", "surface": "#FFFFFF",
    "surface_hover": "#EEEEF1", "border": "#D6D6DB", "border_strong": "#C2C2C9",
    "text": "#1C1C1E", "text2": "#3A3A3E", "text_dim": "#6E6E76",
    "on_accent": "#FFFFFF",
    "accent": "#0A6FF0", "accent_hover": "#2A82F2", "accent_press": "#0858C8",
    "accent_soft": "#E5F0FF", "accent_soft_hover": "#D6E8FF",
    "danger": "#D9352B", "danger_text": "#C62B22",
    "danger_soft": "#FCEAEA", "danger_soft_hover": "#F8DCDC", "danger_border": "#F0C0BE",
    "warn": "#A4641A", "warn_soft": "#FBF1DE",
    "drag": "#B23A86", "drag_soft": "#FAE5F2",
    "scroll": "#2E5BD0", "scroll_soft": "#E4ECFF",
    "slider_handle": "#FFFFFF", "camera_bg": "#0C0C0E", "focus": "#0A6FF0",
}
_DARK = {
    "bg": "#1E1E22", "card": "#26262C", "surface": "#2A2A31",
    "surface_hover": "#33333C", "border": "#3A3A43", "border_strong": "#4A4A55",
    "text": "#F2F2F5", "text2": "#C9C9D0", "text_dim": "#8A8A93",
    "on_accent": "#FFFFFF",
    "accent": "#2A9AEE", "accent_hover": "#48A9F2", "accent_press": "#1E83D8",
    "accent_soft": "#16344A", "accent_soft_hover": "#1B4060",
    "danger": "#FF6B6B", "danger_text": "#FF8585",
    "danger_soft": "#3A2122", "danger_soft_hover": "#4A2A2B", "danger_border": "#5A2E2E",
    "warn": "#FFCC66", "warn_soft": "#3A3320",
    "drag": "#FF88CC", "drag_soft": "#3A1A30",
    "scroll": "#88AAFF", "scroll_soft": "#1A2A3D",
    "slider_handle": "#F2F2F5", "camera_bg": "#0C0C0E", "focus": "#2ADDFF",
}


def _detect_dark():
    """Best-effort OS dark-mode detection (no QApplication needed)."""
    override = os.environ.get("AIRPOINT_THEME", "").strip().lower()
    if override in ("dark", "light"):
        return override == "dark"
    try:
        if sys.platform == "darwin":
            import subprocess
            r = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                               capture_output=True, text=True, timeout=1.5)
            return r.returncode == 0 and "Dark" in r.stdout
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            val, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return val == 0
    except Exception:
        pass
    return True  # default to dark if the OS won't tell us


IS_DARK = _detect_dark()
T = SimpleNamespace(**(_DARK if IS_DARK else _LIGHT))

# Platform system-UI font stacks — makes the app feel native on each OS while
# letting Qt fall back automatically for Indic scripts (Hindi/Malayalam/Tamil).
# On macOS, ".AppleSystemUIFont" resolves to San Francisco (the real system
# font); Helvetica Neue is the always-present native fallback. The family is
# carried by the application font (set in apply_app_theme) rather than the
# global stylesheet, so the genuine system font isn't overridden by QSS.
if sys.platform == "darwin":
    _UI_FAMILIES = [".AppleSystemUIFont", "SF Pro Text", "Helvetica Neue", "Lucida Grande"]
elif sys.platform == "win32":
    _UI_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Tahoma"]
else:
    _UI_FAMILIES = ["Inter", "Noto Sans", "Ubuntu", "DejaVu Sans"]


def _font(size, weight=QFont.Normal):
    """A QFont in the platform system-UI family at the given point size."""
    f = QFont()
    f.setFamilies(_UI_FAMILIES)
    f.setPointSize(size)
    f.setWeight(weight)
    return f


def _theme_html(s):
    """Recolor the inline-HTML helper colors used in translated rich-text strings
    (#ccc = emphasized term, #888 = description) to the active theme so they stay
    legible in both light and dark mode."""
    return s.replace("#ccc", T.text2).replace("#888", T.text_dim)


def _build_base_qss(t):
    return f"""
QWidget {{ background-color: {t.bg}; color: {t.text}; }}
QLabel {{ background: transparent; }}
QPushButton {{
    background-color: {t.accent}; color: {t.on_accent}; border: none;
    border-radius: 8px; padding: 10px 22px; font-size: 14px; font-weight: 600;
}}
QPushButton:hover {{ background-color: {t.accent_hover}; }}
QPushButton:pressed {{ background-color: {t.accent_press}; }}
QPushButton:disabled {{ background-color: {t.surface}; color: {t.text_dim}; }}
QPushButton#secondary {{
    background-color: {t.surface}; color: {t.text};
    border: 1px solid {t.border}; font-weight: 500;
}}
QPushButton#secondary:hover {{ background-color: {t.surface_hover}; border-color: {t.border_strong}; }}
QPushButton#danger {{
    background-color: {t.danger_soft}; color: {t.danger};
    border: 1px solid {t.danger_border}; font-weight: 600;
}}
QPushButton#danger:hover {{ background-color: {t.danger_soft_hover}; }}
QLineEdit {{
    background-color: {t.surface}; border: 1px solid {t.border}; border-radius: 8px;
    padding: 10px; font-size: 15px; color: {t.text};
    selection-background-color: {t.accent}; selection-color: {t.on_accent};
}}
QLineEdit:focus {{ border: 2px solid {t.accent}; }}
QListWidget {{
    background-color: {t.surface}; border: 1px solid {t.border}; border-radius: 8px;
    outline: none; font-size: 14px; padding: 4px;
}}
QListWidget::item {{ padding: 10px 12px; border-radius: 6px; margin: 2px 2px; color: {t.text}; }}
QListWidget::item:selected {{ background-color: {t.accent_soft}; color: {t.accent}; }}
QListWidget::item:hover {{ background-color: {t.surface_hover}; }}
QProgressBar {{
    border: none; border-radius: 7px; background-color: {t.surface_hover};
    text-align: center; color: {t.text_dim}; font-size: 11px; max-height: 14px;
}}
QProgressBar::chunk {{ background-color: {t.accent}; border-radius: 7px; }}
QCheckBox {{ color: {t.text2}; spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid {t.border_strong}; background: {t.surface};
}}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QComboBox {{
    background-color: {t.surface}; color: {t.text}; border: 1px solid {t.border};
    border-radius: 8px; padding: 7px 12px; font-size: 13px;
}}
QComboBox:hover {{ border-color: {t.accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {t.card}; color: {t.text}; border: 1px solid {t.border};
    outline: none; selection-background-color: {t.accent_soft}; selection-color: {t.accent};
}}
QSlider::groove:horizontal {{ height: 5px; background: {t.surface_hover}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 3px; }}
QSlider::add-page:horizontal {{ background: {t.surface_hover}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {t.slider_handle}; border: 1px solid {t.accent};
    width: 16px; height: 16px; margin: -7px 0; border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{ background: {t.accent}; }}
QToolTip {{ background-color: {t.card}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 8px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.border_strong}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {t.text_dim}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
"""


BASE_QSS = _build_base_qss(T)


def apply_app_theme(app):
    """Apply the resolved theme to a QApplication: Fusion base (so Windows and
    macOS render the same clean, Mac-like styling), matching palette for native
    dialogs, system font, and the global stylesheet."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    from PyQt5.QtGui import QPalette, QColor
    t = T
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(t.bg))
    pal.setColor(QPalette.WindowText, QColor(t.text))
    pal.setColor(QPalette.Base, QColor(t.surface))
    pal.setColor(QPalette.AlternateBase, QColor(t.card))
    pal.setColor(QPalette.Text, QColor(t.text))
    pal.setColor(QPalette.Button, QColor(t.surface))
    pal.setColor(QPalette.ButtonText, QColor(t.text))
    pal.setColor(QPalette.BrightText, QColor(t.danger))
    pal.setColor(QPalette.Highlight, QColor(t.accent))
    pal.setColor(QPalette.HighlightedText, QColor(t.on_accent))
    pal.setColor(QPalette.ToolTipBase, QColor(t.card))
    pal.setColor(QPalette.ToolTipText, QColor(t.text))
    try:
        pal.setColor(QPalette.PlaceholderText, QColor(t.text_dim))
    except Exception:
        pass
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(t.text_dim))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(t.text_dim))
    app.setPalette(pal)
    app.setFont(_font(10))
    app.setStyleSheet(BASE_QSS)


class CameraWidget(QLabel):
    """QLabel subclass that displays OpenCV BGR frames."""

    def __init__(self, width=640, height=360, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {T.camera_bg}; border-radius: 10px;")

    def update_frame(self, cv_frame):
        rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = q_img.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))


class SetupWizard(QWidget):
    """PyQt5 setup wizard for profile selection and calibration."""

    # Emitted with the result ("completed"/"skipped"/"quit") when the wizard
    # closes, so a caller can drive a local QEventLoop instead of nesting
    # app.exec_() (which would tear down the whole app on recalibration).
    finished = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.result = None  # "completed", "skipped", or "quit"
        self.profile_name = None

        # Calibration state
        self.cal_step = 0
        self.dir_index = 0
        self.recorded = {}
        self.capture_countdown = None
        self.tremor_start = None
        self.tremor_samples = []
        self.gesture_sampling = False
        self.gesture_sample_start = None
        self.gesture_samples = []
        self.gesture_skipped = False
        self.gesture_results = {}

        # Key press flags (consumed by timer tick)
        self._space = False
        self._n_key = False

        # Window setup
        self.setWindowTitle("AirPoint Setup")
        self.setFixedSize(720, 540)
        self.setStyleSheet(BASE_QSS)

        # Stacked widget for pages
        self.stacked = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stacked)

        # Camera widget (shared across calibration sub-steps)
        self.camera_widget = CameraWidget(640, 360)

        # Timer for calibration camera loop
        self.timer = QTimer()
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._on_timer_tick)

        # Build pages
        self.stacked.addWidget(self._build_language_page())    # 0
        self.stacked.addWidget(self._build_profile_page())     # 1
        self.stacked.addWidget(self._build_name_page())        # 2
        self.stacked.addWidget(self._build_welcome_page())     # 3
        self.stacked.addWidget(self._build_calibration_page()) # 4
        self.stacked.addWidget(self._build_done_page())        # 5

        # Show correct starting page
        if self.controller.profile_name and self.controller.calibration is None:
            # Profile name given via CLI but not found — go straight to welcome
            self.profile_name = self.controller.profile_name
            self.welcome_title.setText(S("welcome_hi", name=self.profile_name))
            self.stacked.setCurrentIndex(3)
        elif self.controller.list_profiles():
            # Returning user — skip language, go to profile selector
            self.stacked.setCurrentIndex(1)
        else:
            # First-time user — start with language choice
            self.stacked.setCurrentIndex(0)

    # ---- Page Builders ----

    def _build_language_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(80, 60, 80, 40)
        vbox.setSpacing(16)

        vbox.addStretch(1)

        title = QLabel("Choose your language")
        title.setFont(_font(24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")
        vbox.addWidget(title)

        sub = QLabel("अपनी भाषा चुनें")
        sub.setFont(_font(16))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(sub)

        vbox.addSpacing(30)

        en_btn = QPushButton("English")
        en_btn.setFixedHeight(52)
        en_btn.setStyleSheet("""
            QPushButton { font-size: 18px; border-radius: 12px; }
        """)
        en_btn.setCursor(Qt.PointingHandCursor)
        en_btn.clicked.connect(lambda: self._on_language_chosen("en"))
        vbox.addWidget(en_btn)

        vbox.addSpacing(10)

        hi_btn = QPushButton("हिन्दी (Hindi)")
        hi_btn.setFixedHeight(52)
        hi_btn.setStyleSheet("""
            QPushButton { font-size: 18px; border-radius: 12px; }
        """)
        hi_btn.setCursor(Qt.PointingHandCursor)
        hi_btn.clicked.connect(lambda: self._on_language_chosen("hi"))
        vbox.addWidget(hi_btn)

        vbox.addSpacing(10)

        ml_btn = QPushButton("മലയാളം (Malayalam)")
        ml_btn.setFixedHeight(52)
        ml_btn.setStyleSheet("""
            QPushButton { font-size: 18px; border-radius: 12px; }
        """)
        ml_btn.setCursor(Qt.PointingHandCursor)
        ml_btn.clicked.connect(lambda: self._on_language_chosen("ml"))
        vbox.addWidget(ml_btn)

        vbox.addSpacing(10)

        ta_btn = QPushButton("தமிழ் (Tamil)")
        ta_btn.setFixedHeight(52)
        ta_btn.setStyleSheet("""
            QPushButton { font-size: 18px; border-radius: 12px; }
        """)
        ta_btn.setCursor(Qt.PointingHandCursor)
        ta_btn.clicked.connect(lambda: self._on_language_chosen("ta"))
        vbox.addWidget(ta_btn)

        vbox.addStretch(1)
        return page

    def _on_language_chosen(self, lang):
        set_language(lang)
        self._refresh_all_text()
        if self.controller.list_profiles():
            self.stacked.setCurrentIndex(1)
        else:
            self.stacked.setCurrentIndex(2)

    def _refresh_all_text(self):
        """Update every translatable widget after language changes."""
        # Profile page
        self._prof_title.setText(S("profile_title"))
        self._prof_sub.setText(S("profile_subtitle"))
        self._prof_select_btn.setText(S("profile_select"))
        self._prof_quit_btn.setText(S("quit"))
        # Update the "+ New Profile" item in the list
        last_idx = self.profile_list.count() - 1
        if last_idx >= 0:
            self.profile_list.item(last_idx).setText(S("profile_new"))
        # Name page
        self._name_title.setText(S("name_title"))
        self._name_sub.setText(S("name_subtitle"))
        self.name_input.setPlaceholderText(S("name_placeholder"))
        self._name_continue_btn.setText(S("continue"))
        # Welcome page
        if self.profile_name:
            self.welcome_title.setText(S("welcome_hi", name=self.profile_name))
        else:
            self.welcome_title.setText(S("welcome_default"))
        self._welcome_sub.setText(S("welcome_sub"))
        self._how_title.setText(S("how_title"))
        for title_key, t_label, desc_key, d_label in self._feat_labels:
            t_label.setText(f"<b>{S(title_key)}</b>")
            d_label.setText(S(desc_key))
        self._setup_note.setText(S("setup_note"))
        self._begin_btn.setText(S("lets_go"))
        self._skip_btn.setText(S("skip_setup"))
        # Done page
        self._done_title.setText(S("done_title"))
        self.done_subtitle.setText(S("done_subtitle"))
        for g_key, g_label, a_key, a_label in self._done_gesture_labels:
            g_label.setText(S(g_key))
            a_label.setText(S(a_key))
        self._extras_title.setText(S("done_extras_title"))
        self._extras_desc.setText(_theme_html(S("done_extras_gaze") + "<br>" + S("done_extras_dwell")))
        self.autostart_cb.setText(S("autostart_label"))
        self._start_btn.setText(S("start_airpoint"))

    def _build_profile_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(60, 40, 60, 30)
        vbox.setSpacing(12)

        self._prof_title = QLabel(S("profile_title"))
        self._prof_title.setFont(_font(24, QFont.Bold))
        self._prof_title.setAlignment(Qt.AlignCenter)
        self._prof_title.setStyleSheet("color: white;")
        vbox.addWidget(self._prof_title)

        self._prof_sub = QLabel(S("profile_subtitle"))
        self._prof_sub.setFont(_font(12))
        self._prof_sub.setAlignment(Qt.AlignCenter)
        self._prof_sub.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(self._prof_sub)

        vbox.addSpacing(10)

        self.profile_list = QListWidget()
        for name in self.controller.list_profiles():
            self.profile_list.addItem(name)
        self.profile_list.addItem(S("profile_new"))
        self.profile_list.setCurrentRow(0)
        self.profile_list.itemDoubleClicked.connect(self._on_profile_select)
        vbox.addWidget(self.profile_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._prof_select_btn = QPushButton(S("profile_select"))
        self._prof_select_btn.clicked.connect(self._on_profile_select)
        btn_row.addWidget(self._prof_select_btn)

        self._prof_quit_btn = QPushButton(S("quit"))
        self._prof_quit_btn.setObjectName("secondary")
        self._prof_quit_btn.clicked.connect(lambda: self._finish("quit"))
        btn_row.addWidget(self._prof_quit_btn)
        vbox.addLayout(btn_row)

        return page

    def _build_name_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(80, 60, 80, 40)
        vbox.setSpacing(16)

        vbox.addStretch(1)

        self._name_title = QLabel(S("name_title"))
        self._name_title.setFont(_font(22, QFont.Bold))
        self._name_title.setAlignment(Qt.AlignCenter)
        self._name_title.setStyleSheet("color: white;")
        vbox.addWidget(self._name_title)

        self._name_sub = QLabel(S("name_subtitle"))
        self._name_sub.setFont(_font(11))
        self._name_sub.setAlignment(Qt.AlignCenter)
        self._name_sub.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(self._name_sub)

        vbox.addSpacing(10)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(S("name_placeholder"))
        self.name_input.setMaxLength(20)
        self.name_input.setAlignment(Qt.AlignCenter)
        self.name_input.setFixedHeight(48)
        self.name_input.returnPressed.connect(self._on_name_entered)
        vbox.addWidget(self.name_input)

        vbox.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._name_continue_btn = QPushButton(S("continue"))
        self._name_continue_btn.clicked.connect(self._on_name_entered)
        btn_row.addStretch()
        btn_row.addWidget(self._name_continue_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        vbox.addStretch(1)
        return page

    def _build_welcome_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(40, 24, 40, 20)
        vbox.setSpacing(6)

        self.welcome_title = QLabel(S("welcome_default"))
        self.welcome_title.setFont(_font(24, QFont.Bold))
        self.welcome_title.setAlignment(Qt.AlignCenter)
        self.welcome_title.setStyleSheet("color: white;")
        vbox.addWidget(self.welcome_title)

        self._welcome_sub = QLabel(S("welcome_sub"))
        self._welcome_sub.setFont(_font(12))
        self._welcome_sub.setAlignment(Qt.AlignCenter)
        self._welcome_sub.setStyleSheet(f"color: {T.text_dim};")
        self._welcome_sub.setWordWrap(True)
        vbox.addWidget(self._welcome_sub)

        vbox.addSpacing(10)

        self._how_title = QLabel(S("how_title"))
        self._how_title.setFont(_font(14, QFont.Bold))
        self._how_title.setStyleSheet(f"color: {T.accent};")
        vbox.addWidget(self._how_title)

        vbox.addSpacing(4)

        # Feature rows — store labels for translation refresh
        self._feat_labels = []
        feat_keys = [
            ("feat_move_title", "feat_move_desc"),
            ("feat_click_title", "feat_click_desc"),
            ("feat_right_title", "feat_right_desc"),
            ("feat_scroll_title", "feat_scroll_desc"),
        ]
        for title_key, desc_key in feat_keys:
            row = QHBoxLayout()
            row.setSpacing(10)
            row.setContentsMargins(0, 0, 0, 0)
            t = QLabel(f"<b>{S(title_key)}</b>")
            t.setFont(_font(11))
            t.setStyleSheet(f"color: {T.text2};")
            t.setFixedWidth(120)
            t.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            row.addWidget(t)
            d = QLabel(S(desc_key))
            d.setFont(_font(10))
            d.setStyleSheet(f"color: {T.text_dim};")
            d.setWordWrap(True)
            row.addWidget(d, 1)
            self._feat_labels.append((title_key, t, desc_key, d))
            vbox.addLayout(row)
            vbox.addSpacing(2)

        vbox.addSpacing(6)

        self._setup_note = QLabel(S("setup_note"))
        self._setup_note.setFont(_font(11))
        self._setup_note.setAlignment(Qt.AlignCenter)
        self._setup_note.setStyleSheet(f"color: {T.text_dim};")
        self._setup_note.setWordWrap(True)
        vbox.addWidget(self._setup_note)

        vbox.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._begin_btn = QPushButton(S("lets_go"))
        self._begin_btn.clicked.connect(self._start_calibration)
        btn_row.addStretch()
        btn_row.addWidget(self._begin_btn)

        self._skip_btn = QPushButton(S("skip_setup"))
        self._skip_btn.setObjectName("secondary")
        self._skip_btn.clicked.connect(lambda: self._finish("skipped"))
        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        return page

    def _build_calibration_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(20, 16, 20, 12)
        vbox.setSpacing(8)

        # Step dots row
        self.step_dots = []
        dots_row = QHBoxLayout()
        dots_row.setSpacing(8)
        dots_row.addStretch()
        for i in range(4):
            dot = QLabel()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet(f"background-color: {T.border}; border-radius: 7px;")
            self.step_dots.append(dot)
            dots_row.addWidget(dot)
        dots_row.addStretch()
        vbox.addLayout(dots_row)

        # Title and instruction
        self.cal_title = QLabel(S("cal_step1_title"))
        self.cal_title.setFont(_font(11))
        self.cal_title.setAlignment(Qt.AlignCenter)
        self.cal_title.setStyleSheet(f"color: {T.text_dim};")
        self.cal_title.setWordWrap(True)
        vbox.addWidget(self.cal_title)

        self.cal_instruction = QLabel(S("cal_move_left"))
        self.cal_instruction.setFont(_font(20, QFont.Bold))
        self.cal_instruction.setAlignment(Qt.AlignCenter)
        self.cal_instruction.setStyleSheet(f"color: {T.accent};")
        self.cal_instruction.setWordWrap(True)
        vbox.addWidget(self.cal_instruction)

        # Camera
        cam_row = QHBoxLayout()
        cam_row.addStretch()
        cam_row.addWidget(self.camera_widget)
        cam_row.addStretch()
        vbox.addLayout(cam_row)

        # Hint
        self.cal_hint = QLabel(S("cal_hint_dir", dir="LEFT"))
        self.cal_hint.setFont(_font(16, QFont.Bold))
        self.cal_hint.setAlignment(Qt.AlignCenter)
        self.cal_hint.setStyleSheet(f"color: {T.accent};")
        vbox.addWidget(self.cal_hint)

        # Progress bar (hidden by default)
        self.cal_progress = QProgressBar()
        self.cal_progress.setRange(0, 100)
        self.cal_progress.setValue(0)
        self.cal_progress.setFixedHeight(18)
        self.cal_progress.setVisible(False)
        vbox.addWidget(self.cal_progress)

        return page

    def _build_done_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(40, 24, 40, 20)
        vbox.setSpacing(8)

        # Completed dots
        dots_row = QHBoxLayout()
        dots_row.setSpacing(8)
        dots_row.addStretch()
        for _ in range(4):
            dot = QLabel()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet(f"background-color: {T.accent}; border-radius: 7px;")
            dots_row.addWidget(dot)
        dots_row.addStretch()
        vbox.addLayout(dots_row)

        vbox.addSpacing(6)

        self._done_title = QLabel(S("done_title"))
        self._done_title.setFont(_font(24, QFont.Bold))
        self._done_title.setAlignment(Qt.AlignCenter)
        self._done_title.setStyleSheet(f"color: {T.accent};")
        vbox.addWidget(self._done_title)

        self.done_subtitle = QLabel(S("done_subtitle"))
        self.done_subtitle.setFont(_font(11))
        self.done_subtitle.setAlignment(Qt.AlignCenter)
        self.done_subtitle.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(self.done_subtitle)

        vbox.addSpacing(8)

        # Quick reference card
        card = QWidget()
        card.setStyleSheet(f"background-color: {T.card}; border: 1px solid {T.border}; border-radius: 10px; padding: 12px;")
        card_vbox = QVBoxLayout(card)
        card_vbox.setContentsMargins(14, 10, 14, 10)
        card_vbox.setSpacing(6)

        gesture_keys = [
            ("done_gesture_open", "done_action_move"),
            ("done_gesture_pinch", "done_action_click"),
            ("done_gesture_fist", "done_action_right"),
            ("done_gesture_scroll", "done_action_scroll"),
        ]
        self._done_gesture_labels = []
        for g_key, a_key in gesture_keys:
            row = QHBoxLayout()
            g = QLabel(S(g_key))
            g.setFont(_font(11))
            g.setStyleSheet(f"color: {T.text2};")
            row.addWidget(g)
            row.addStretch()
            a = QLabel(S(a_key))
            a.setFont(_font(11, QFont.Bold))
            a.setStyleSheet(f"color: {T.accent};")
            a.setAlignment(Qt.AlignRight)
            row.addWidget(a)
            self._done_gesture_labels.append((g_key, g, a_key, a))
            card_vbox.addLayout(row)

        vbox.addWidget(card)

        vbox.addSpacing(6)

        self._extras_title = QLabel(S("done_extras_title"))
        self._extras_title.setFont(_font(11))
        self._extras_title.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(self._extras_title)

        self._extras_desc = QLabel(_theme_html(S("done_extras_gaze") + "<br>" + S("done_extras_dwell")))
        self._extras_desc.setFont(_font(10))
        self._extras_desc.setWordWrap(True)
        self._extras_desc.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(self._extras_desc)

        vbox.addSpacing(8)

        self.autostart_cb = QCheckBox(S("autostart_label"))
        self.autostart_cb.setFont(_font(11))
        self.autostart_cb.setChecked(get_autostart_enabled())
        cb_row = QHBoxLayout()
        cb_row.addStretch()
        cb_row.addWidget(self.autostart_cb)
        cb_row.addStretch()
        vbox.addLayout(cb_row)

        vbox.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._start_btn = QPushButton(S("start_airpoint"))
        self._start_btn.clicked.connect(lambda: self._finish("completed"))
        btn_row.addStretch()
        btn_row.addWidget(self._start_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        return page

    # ---- Page Actions ----

    def _on_profile_select(self):
        item = self.profile_list.currentItem()
        if item is None:
            return
        text = item.text()
        if text == S("profile_new"):
            self.stacked.setCurrentIndex(2)
            self.name_input.setFocus()
        else:
            self.profile_name = text
            self.controller.load_profile(text)
            self._finish("completed")

    def _on_name_entered(self):
        raw = self.name_input.text().strip()
        name = "".join(c for c in raw if c.isalnum() or c in " _-").strip() or "default"
        self.profile_name = name
        self.controller.profile_name = name
        self.welcome_title.setText(S("welcome_hi", name=name))
        self.stacked.setCurrentIndex(3)

    def _start_calibration(self):
        self.cal_step = 0
        self.dir_index = 0
        self.recorded = {}
        self.capture_countdown = None
        self.tremor_start = None
        self.tremor_samples = []
        self.gesture_results = {}
        self._update_cal_display()
        self.stacked.setCurrentIndex(4)
        self.timer.start()

    def _update_cal_display(self):
        DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
        GESTURE_NAMES = ["Pinch", "Fist"]

        # Update step dots
        step_map = {0: 0, 1: 1, 2: 2, 3: 3}
        current = step_map.get(self.cal_step, 0)
        for i, dot in enumerate(self.step_dots):
            if i < current:
                dot.setStyleSheet(f"background-color: {T.accent}; border-radius: 7px;")
            elif i == current:
                dot.setStyleSheet(f"background-color: {T.accent}; border-radius: 7px; border: 2px solid {T.text};")
            else:
                dot.setStyleSheet(f"background-color: {T.border}; border-radius: 7px;")

        if self.cal_step == 0:
            d = DIRECTIONS[self.dir_index] if self.dir_index < 4 else "DOWN"
            self.cal_title.setText(S("cal_step1_title"))
            inst_map = {
                "LEFT": S("cal_move_left"),
                "RIGHT": S("cal_move_right"),
                "UP": S("cal_move_up"),
                "DOWN": S("cal_move_down"),
            }
            self.cal_instruction.setText(inst_map.get(d, ""))
            # For Hindi, direction names stay in the hint via {dir}
            dir_display = {"LEFT": "LEFT", "RIGHT": "RIGHT", "UP": "UP", "DOWN": "DOWN"}
            _dir_by_lang = {
                "hi": {"LEFT": "बाईं ओर", "RIGHT": "दाईं ओर", "UP": "ऊपर", "DOWN": "नीचे"},
                "ml": {"LEFT": "ഇടത്", "RIGHT": "വലത്", "UP": "മുകൾ", "DOWN": "താഴ്"},
                "ta": {"LEFT": "இடது", "RIGHT": "வலது", "UP": "மேல்", "DOWN": "கீழ்"},
            }
            if _current_lang in _dir_by_lang:
                dir_display = _dir_by_lang[_current_lang]
            self.cal_hint.setText(S("cal_hint_dir", dir=dir_display.get(d, d)))
            self.cal_progress.setVisible(False)
        elif self.cal_step == 1:
            self.cal_title.setText(S("cal_step2_title"))
            self.cal_instruction.setText(S("cal_steady_inst"))
            self.cal_hint.setText(S("cal_hint_steady"))
            self.cal_progress.setVisible(True)
            self.cal_progress.setValue(0)
        elif self.cal_step == 2:
            self.cal_title.setText(S("cal_step3_title"))
            self.cal_instruction.setText(S("cal_pinch_inst"))
            self.cal_hint.setText(S("cal_hint_gesture"))
            self.cal_progress.setVisible(False)
            self.gesture_sampling = False
            self.gesture_samples = []
            self.gesture_skipped = False
        elif self.cal_step == 3:
            self.cal_title.setText(S("cal_step4_title"))
            self.cal_instruction.setText(S("cal_fist_inst"))
            self.cal_hint.setText(S("cal_hint_gesture"))
            self.cal_progress.setVisible(False)
            self.gesture_sampling = False
            self.gesture_samples = []
            self.gesture_skipped = False

    # ---- Timer Tick (Camera + Calibration Logic) ----

    def _on_timer_tick(self):
        try:
            ret, frame = self.controller.cap.read()
            if not ret:
                return
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.controller.hands.process(rgb_frame)
            frame_h, frame_w = frame.shape[:2]

            hand_center = None
            landmarks = None
            if hand_results.multi_hand_landmarks:
                for hl in hand_results.multi_hand_landmarks:
                    self.controller.mp_draw.draw_landmarks(
                        frame, hl, self.controller.mp_hands.HAND_CONNECTIONS,
                        self.controller.mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1),
                        self.controller.mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1)
                    )
                    landmarks = self.controller.get_landmarks(hl)
                    hand_center = self.controller.calculate_hand_center(landmarks)
                    hx = int(hand_center[0] * frame_w)
                    hy = int(hand_center[1] * frame_h)
                    cv2.circle(frame, (hx, hy), 20, (0, 220, 200), 3)
                    cv2.circle(frame, (hx, hy), 5, (0, 220, 200), -1)

            # Draw recorded direction points
            for rec_label, rec_pos in self.recorded.items():
                rx = int(rec_pos[0] * frame_w)
                ry = int(rec_pos[1] * frame_h)
                cv2.circle(frame, (rx, ry), 12, (0, 200, 120), -1)

            self.camera_widget.update_frame(frame)

            # Dispatch to step handler
            if self.cal_step == 0:
                self._tick_movement(hand_center)
            elif self.cal_step == 1:
                self._tick_steadiness(hand_center)
            elif self.cal_step == 2:
                self._tick_gesture(hand_center, landmarks, "PINCH")
            elif self.cal_step == 3:
                self._tick_gesture(hand_center, landmarks, "FIST")

            # Consume key flags
            self._space = False
            self._n_key = False
            self._cal_error_count = 0
        except SystemExit:
            raise
        except Exception as e:
            # One bad frame must never crash the setup wizard (mirrors _tracking_tick).
            _write_crash_log(type(e), e, e.__traceback__)
            self._cal_error_count = getattr(self, '_cal_error_count', 0) + 1
            if self._cal_error_count >= 60:  # ~2s of consecutive failures
                self.timer.stop()
                try:
                    QMessageBox.warning(self, "AirPoint",
                        "AirPoint lost the camera during setup. Reconnect it (or "
                        "close any app using it) and start setup again.")
                except Exception:
                    pass
                self._finish("quit")

    def _tick_movement(self, hand_center):
        DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
        HOLD_TIME = 1.0

        if self.dir_index >= 4:
            self.cal_step = 1
            self._update_cal_display()
            return

        if self._space and self.capture_countdown is None and hand_center is not None:
            self.capture_countdown = time.time()
            self.cal_hint.setText(S("cal_hold_still"))
            self.cal_hint.setStyleSheet(f"color: {T.accent}; font-weight: bold;")

        if self.capture_countdown is not None:
            elapsed = time.time() - self.capture_countdown
            if elapsed >= HOLD_TIME:
                if hand_center is not None:
                    label = DIRECTIONS[self.dir_index]
                    self.recorded[label] = hand_center
                    print(f"  Captured {label}: ({hand_center[0]:.4f}, {hand_center[1]:.4f})")
                    self.dir_index += 1
                    self.capture_countdown = None
                    self.cal_hint.setStyleSheet(f"color: {T.accent}; font-weight: bold;")
                    self._update_cal_display()
                else:
                    self.capture_countdown = None
                    self.cal_hint.setText(S("cal_hand_lost"))
                    self.cal_hint.setStyleSheet(f"color: {T.danger}; font-weight: bold;")

    def _tick_steadiness(self, hand_center):
        TREMOR_DURATION = 5.0

        if hand_center is not None:
            if self.tremor_start is None:
                self.tremor_start = time.time()
            elapsed = time.time() - self.tremor_start
            self.tremor_samples.append(hand_center)

            progress = min(1.0, elapsed / TREMOR_DURATION)
            self.cal_progress.setValue(int(progress * 100))

            if elapsed >= TREMOR_DURATION:
                self._finish_steadiness()
                return
        else:
            if self.tremor_start is not None:
                self.tremor_start = None
                self.tremor_samples.clear()
                self.cal_progress.setValue(0)
            self.cal_hint.setText(S("cal_show_hand"))
            self.cal_hint.setStyleSheet(f"color: {T.danger}; font-weight: bold;")

    def _finish_steadiness(self):
        if len(self.tremor_samples) >= 10:
            xs = [s[0] for s in self.tremor_samples]
            ys = [s[1] for s in self.tremor_samples]
            tremor_std = float(np.sqrt(np.std(xs)**2 + np.std(ys)**2))
        else:
            tremor_std = 0.005

        MIN_STD, MAX_STD = 0.001, 0.015
        MIN_SMOOTH, MAX_SMOOTH = 0.45, 0.85
        clamped = max(MIN_STD, min(MAX_STD, tremor_std))
        t = (clamped - MIN_STD) / (MAX_STD - MIN_STD)
        self.controller.smoothing_factor = MIN_SMOOTH + t * (MAX_SMOOTH - MIN_SMOOTH)
        self.controller.smoothed_screen_pos = None
        self.controller._smoothed_pass2 = None
        self.controller._prev_raw_pos = None
        self.controller._last_output_pos = None
        self.tremor_std = tremor_std

        print(f"  Tremor STD: {tremor_std:.5f} ({len(self.tremor_samples)} samples)")
        print(f"  Auto smoothing_factor: {self.controller.smoothing_factor:.2f}")

        self.cal_step = 2
        self._update_cal_display()

    def _tick_gesture(self, hand_center, landmarks, gesture_name):
        GESTURE_SAMPLE_TIME = 1.5

        measured_value = None
        if landmarks is not None:
            if gesture_name == "PINCH":
                measured_value = self.controller.calculate_distance(landmarks[4], landmarks[8])
            else:  # FIST
                palm_center = landmarks[9]
                dists = [self.controller.calculate_distance(landmarks[tip], palm_center) for tip in [4, 8, 12, 16, 20]]
                measured_value = float(np.mean(dists))

        if self._n_key and not self.gesture_sampling:
            self.gesture_skipped = True
            self.gesture_results[gesture_name] = None
            print(f"  {gesture_name}: SKIPPED")
            self._advance_from_gesture(gesture_name)
            return

        if self._space and not self.gesture_sampling and measured_value is not None:
            self.gesture_sampling = True
            self.gesture_sample_start = time.time()
            self.gesture_samples = []
            self.cal_progress.setVisible(True)
            self.cal_progress.setValue(0)
            self.cal_hint.setText(S("cal_recording"))
            self.cal_hint.setStyleSheet(f"color: {T.accent}; font-weight: bold;")

        if self.gesture_sampling and self.gesture_sample_start is not None:
            elapsed = time.time() - self.gesture_sample_start
            if measured_value is not None:
                self.gesture_samples.append(measured_value)

            progress = min(1.0, elapsed / GESTURE_SAMPLE_TIME)
            self.cal_progress.setValue(int(progress * 100))

            if elapsed >= GESTURE_SAMPLE_TIME:
                if len(self.gesture_samples) >= 5:
                    avg = float(np.mean(self.gesture_samples))
                    self.gesture_results[gesture_name] = avg
                    print(f"  {gesture_name}: measured={avg:.4f} ({len(self.gesture_samples)} samples)")
                    self._advance_from_gesture(gesture_name)
                else:
                    # Hand wasn't visible during recording. Don't silently disable
                    # the gesture — reset and let the user try again (or press N to skip).
                    print(f"  {gesture_name}: too few samples, retrying")
                    self.gesture_sampling = False
                    self.gesture_sample_start = None
                    self.gesture_samples = []
                    self.cal_progress.setVisible(False)
                    self.cal_hint.setText(S("cal_gesture_retry"))
                    self.cal_hint.setStyleSheet(f"color: {T.danger}; font-weight: bold;")

    def _advance_from_gesture(self, gesture_name):
        self.cal_progress.setVisible(False)
        self.cal_hint.setStyleSheet(f"color: {T.accent}; font-weight: bold;")
        if gesture_name == "PINCH":
            self.cal_step = 3
            self._update_cal_display()
        else:
            # Fist done — finalize calibration
            self._finish_calibration()

    def _finish_calibration(self):
        self.timer.stop()

        # Set personal thresholds
        pinch_raw = self.gesture_results.get("PINCH")
        if pinch_raw is not None:
            self.controller.pinch_threshold = pinch_raw * 1.3
            print(f"  Pinch threshold: {self.controller.pinch_threshold:.4f}")
        else:
            self.controller.pinch_threshold = None
            print("  Pinch: DISABLED")

        fist_raw = self.gesture_results.get("FIST")
        if fist_raw is not None:
            self.controller.fist_threshold = fist_raw * 1.3
            print(f"  Fist threshold: {self.controller.fist_threshold:.4f}")
        else:
            self.controller.fist_threshold = None
            print("  Fist: DISABLED")

        tremor_std = getattr(self, 'tremor_std', 0.005)
        self.controller.calibration = {
            "left":   self.recorded["LEFT"][0],
            "right":  self.recorded["RIGHT"][0],
            "top":    self.recorded["UP"][1],
            "bottom": self.recorded["DOWN"][1],
            "tremor_std": tremor_std,
            "calibration_margin": self.controller.calibration_margin,
        }
        if self.controller.calibration["left"] > self.controller.calibration["right"]:
            self.controller.calibration["left"], self.controller.calibration["right"] = \
                self.controller.calibration["right"], self.controller.calibration["left"]
        if self.controller.calibration["top"] > self.controller.calibration["bottom"]:
            self.controller.calibration["top"], self.controller.calibration["bottom"] = \
                self.controller.calibration["bottom"], self.controller.calibration["top"]

        self.controller.save_profile()
        print(f"Calibration complete for '{self.profile_name}'!")

        self.done_subtitle.setText(S("done_subtitle"))
        self.stacked.setCurrentIndex(5)

    # ---- Finish / Close ----

    def _finish(self, result):
        self.timer.stop()
        self.result = result
        # Apply autostart preference
        if result == "completed" and hasattr(self, 'autostart_cb'):
            set_autostart(self.autostart_cb.isChecked())
        self.close()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._space = True
        elif key == Qt.Key_N:
            self._n_key = True
        elif key == Qt.Key_Q:
            self._finish("quit")
        elif key == Qt.Key_Escape:
            self._finish("quit")

    def closeEvent(self, event):
        self.timer.stop()
        if self.result is None:
            self.result = "quit"
        event.accept()
        self.finished.emit(self.result)


class StatusPanel(QWidget):
    """User-friendly control panel shown during tracking."""

    TOGGLE_ON = f"""
        QPushButton {{ background-color: {T.accent_soft}; color: {T.accent}; border: 1px solid {T.accent};
                      border-radius: 12px; padding: 12px; text-align: left; font-size: 14px; font-weight: 600; }}
        QPushButton:hover {{ background-color: {T.accent_soft_hover}; }}
    """
    TOGGLE_OFF = f"""
        QPushButton {{ background-color: {T.surface}; color: {T.text_dim}; border: 1px solid {T.border};
                      border-radius: 12px; padding: 12px; text-align: left; font-size: 14px; font-weight: 500; }}
        QPushButton:hover {{ background-color: {T.surface_hover}; }}
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("AirPoint")
        self.setFixedSize(360, 660)
        self.setStyleSheet(BASE_QSS + f" StatusPanel {{ background-color: {T.bg}; }}")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(24, 20, 24, 18)
        vbox.setSpacing(0)

        # ---- Header + status badge ----
        header = QLabel("AirPoint")
        header.setFont(_font(20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color: {T.accent};")
        vbox.addWidget(header)

        self.profile_label = QLabel()
        self.profile_label.setFont(_font(11))
        self.profile_label.setAlignment(Qt.AlignCenter)
        self.profile_label.setStyleSheet(f"color: {T.text_dim};")
        vbox.addWidget(self.profile_label)

        vbox.addSpacing(10)

        # Big status indicator
        self.status_badge = QLabel(S("panel_looking"))
        self.status_badge.setFont(_font(15, QFont.Bold))
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedHeight(50)
        self.status_badge.setStyleSheet(
            f"background-color: {T.surface}; color: {T.text_dim};"
            f" border: 1px solid {T.border}; border-radius: 12px; padding: 8px;")
        vbox.addWidget(self.status_badge)

        vbox.addSpacing(12)

        # ---- Pause / resume (park the cursor) ----
        self.pause_btn = QPushButton()
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setFixedHeight(48)
        self.pause_btn.clicked.connect(self._toggle_pause)
        vbox.addWidget(self.pause_btn)

        vbox.addSpacing(10)

        # ---- Toggle buttons ----
        self.gaze_btn = QPushButton()
        self.gaze_btn.setCursor(Qt.PointingHandCursor)
        self.gaze_btn.setFixedHeight(52)
        self.gaze_btn.clicked.connect(self._toggle_gaze)
        vbox.addWidget(self.gaze_btn)

        vbox.addSpacing(8)

        self.dwell_btn = QPushButton()
        self.dwell_btn.setCursor(Qt.PointingHandCursor)
        self.dwell_btn.setFixedHeight(52)
        self.dwell_btn.clicked.connect(self._toggle_dwell)
        vbox.addWidget(self.dwell_btn)

        vbox.addStretch(1)

        # ---- Settings / Profiles row ----
        tools_row = QHBoxLayout()
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setObjectName("secondary")
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setFixedHeight(44)
        self.settings_btn.clicked.connect(self._open_settings)
        self.profiles_btn = QPushButton("Profiles")
        self.profiles_btn.setObjectName("secondary")
        self.profiles_btn.setCursor(Qt.PointingHandCursor)
        self.profiles_btn.setFixedHeight(44)
        self.profiles_btn.clicked.connect(self._open_profiles)
        tools_row.addWidget(self.settings_btn)
        tools_row.addWidget(self.profiles_btn)
        vbox.addLayout(tools_row)

        vbox.addSpacing(8)

        # ---- Bottom actions ----
        self.recal_btn = QPushButton(S("panel_redo"))
        self.recal_btn.setObjectName("secondary")
        self.recal_btn.setCursor(Qt.PointingHandCursor)
        self.recal_btn.setFixedHeight(44)
        vbox.addWidget(self.recal_btn)

        vbox.addSpacing(8)

        self.quit_btn = QPushButton(S("panel_stop"))
        self.quit_btn.setObjectName("danger")
        self.quit_btn.setCursor(Qt.PointingHandCursor)
        self.quit_btn.setFixedHeight(44)
        vbox.addWidget(self.quit_btn)

        # Timer for updating status
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._update_status)

    def _toggle_gaze(self):
        self.controller.toggle_gaze_detection()
        self._update_status()

    def _toggle_pause(self):
        self.controller.toggle_pause()
        self._update_status()

    def _toggle_dwell(self):
        self.controller.toggle_dwell_click()
        self._update_status()

    def _open_settings(self):
        if getattr(self, 'settings_panel', None) is None:
            self.settings_panel = SettingsPanel(self.controller)
        else:
            self.settings_panel._sync_from_controller()
        self.settings_panel.show()
        self.settings_panel.raise_()
        self.settings_panel.activateWindow()

    def _open_profiles(self):
        if getattr(self, 'profiles_panel', None) is None:
            self.profiles_panel = ProfilesPanel(self.controller)
        else:
            self.profiles_panel._refresh()
        self.profiles_panel.show()
        self.profiles_panel.raise_()
        self.profiles_panel.activateWindow()

    def start(self):
        self._update_status()
        self.timer.start()

    def _set_badge(self, bg, fg, border=None):
        self.status_badge.setStyleSheet(
            f"background-color: {bg}; color: {fg};"
            f" border: 1px solid {border or bg}; border-radius: 12px; padding: 8px;")

    def _update_status(self):
        c = self.controller
        self.profile_label.setText(S("panel_hi", name=c.profile_name or "User"))

        # -- Pause button --
        if getattr(c, 'paused', False):
            self.pause_btn.setText("Resume tracking")
            self.pause_btn.setStyleSheet(self.TOGGLE_ON)
        else:
            self.pause_btn.setText("Pause tracking")
            self.pause_btn.setStyleSheet(self.TOGGLE_OFF)

        # -- Status badge --
        gesture = getattr(c, '_last_gesture', 'no_hand')
        if getattr(c, 'paused', False):
            self.status_badge.setText("Paused")
            self._set_badge(T.warn_soft, T.warn)
        elif gesture == 'no_hand':
            self.status_badge.setText(S("panel_looking"))
            self._set_badge(T.surface, T.text_dim, border=T.border)
        else:
            friendly = {
                "cursor_control": S("panel_moving"),
                "left_click": S("panel_clicked"),
                "drag_start": S("panel_dragging"),
                "dragging": S("panel_dragging"),
                "drag_end": S("panel_drag_done"),
                "right_click": S("panel_right_clicked"),
                "two_finger_scroll": S("panel_scrolling"),
                "scroll_active": S("panel_scrolling"),
                "pinch_wait": S("panel_pinch_drag"),
                "dwell_click": S("panel_auto_clicked"),
                "safety_disabled": S("panel_look_screen"),
                "idle": S("panel_ready"),
            }.get(gesture, S("panel_ready"))
            self.status_badge.setText(friendly)
            bg, fg = T.accent_soft, T.accent
            if "drag" in gesture.lower():
                bg, fg = T.drag_soft, T.drag
            elif "scroll" in gesture.lower():
                bg, fg = T.scroll_soft, T.scroll
            elif "safety" in gesture.lower():
                bg, fg = T.warn_soft, T.warn
            self._set_badge(bg, fg)

        # -- Gaze toggle --
        if c.gaze_detection_enabled:
            self.gaze_btn.setText(S("panel_gaze_on"))
            self.gaze_btn.setStyleSheet(self.TOGGLE_ON)
        else:
            self.gaze_btn.setText(S("panel_gaze_off"))
            self.gaze_btn.setStyleSheet(self.TOGGLE_OFF)

        # -- Dwell toggle --
        if c.dwell_click_enabled:
            self.dwell_btn.setText(S("panel_dwell_on"))
            self.dwell_btn.setStyleSheet(self.TOGGLE_ON)
        else:
            self.dwell_btn.setText(S("panel_dwell_off"))
            self.dwell_btn.setStyleSheet(self.TOGGLE_OFF)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_G:
            self._toggle_gaze()
        elif key == Qt.Key_D:
            self._toggle_dwell()
        elif key == Qt.Key_C:
            self.recal_btn.click()
        elif key == Qt.Key_Q or key == Qt.Key_Escape:
            self.quit_btn.click()

    def closeEvent(self, event):
        self.timer.stop()
        for attr in ('settings_panel', 'profiles_panel'):
            p = getattr(self, attr, None)
            if p is not None:
                p.close()
        # Closing the panel by ANY means (incl. the OS X-button) must stop the
        # tracking loop — otherwise the cursor keeps moving with no UI to stop it.
        cb = getattr(self, '_on_close_quit', None)
        if cb is not None:
            self._on_close_quit = None  # one-shot: on_quit calls panel.close() again
            cb()
        event.accept()


class SettingsPanel(QWidget):
    """Live tuning: pointer-feel presets + fine sliders. Every change applies
    instantly to the running controller (the tracking loop reads these attrs
    each frame); Save persists them to the active profile."""

    PRESET_ON = (f"QPushButton {{ background:{T.accent_soft}; color:{T.accent}; border:1px solid {T.accent};"
                 f" border-radius:9px; font-size:12px; font-weight:600; }}")
    PRESET_OFF = (f"QPushButton {{ background:{T.surface}; color:{T.text_dim}; border:1px solid {T.border};"
                  f" border-radius:9px; font-size:12px; }} QPushButton:hover {{ background:{T.surface_hover}; }}")

    # caption, controller attr, lo, hi, step, scale, is_int, value formatter
    SLIDERS = [
        ("Cursor speed",            "sensitivity",          0.5, 5.0,  0.1,  10,  False, lambda x: f"{x:.1f}x"),
        ("Steadiness (smoothing)",  "smoothing_factor",     0.0, 0.95, 0.05, 100, False, lambda x: f"{int(round(x*100))}%"),
        ("Ignore tiny movements",   "cursor_dead_zone",     0,   40,   1,    1,   True,  lambda x: f"{int(x)} px"),
        ("Scroll speed",            "scroll_amount",        1,   10,   1,    1,   True,  lambda x: f"{int(x)}"),
        ("Hold-to-drag time",       "drag_threshold",       0.2, 1.0,  0.05, 100, False, lambda x: f"{x:.2f}s"),
        ("Time between clicks",     "action_cooldown",      0.1, 0.8,  0.05, 100, False, lambda x: f"{x:.2f}s"),
        ("Hover-click time",        "dwell_click_duration", 0.5, 4.0,  0.1,  10,  False, lambda x: f"{x:.1f}s"),
    ]

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._loading = False
        self.setWindowTitle("AirPoint Settings")
        # Fixed width, content-driven height so Save/Reset can never be clipped
        # off the bottom under HiDPI / large-font scaling.
        self.setFixedWidth(400)
        self.setMinimumHeight(560)
        self.setStyleSheet(BASE_QSS + f" SettingsPanel {{ background-color: {T.bg}; }}")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)

        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        v.setSpacing(7)

        header = QLabel("Settings")
        header.setFont(_font(18, QFont.Bold))
        header.setStyleSheet(f"color: {T.accent};")
        v.addWidget(header)

        # ---- Pointer-feel presets ----
        v.addWidget(self._caption("Pointer feel"))
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self.preset_btns = {}
        for key, label in PRESET_LABELS:
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(34)
            b.clicked.connect(lambda _=False, k=key: self._on_preset(k))
            self.preset_btns[key] = b
            preset_row.addWidget(b)
        v.addLayout(preset_row)
        self.custom_lbl = QLabel("")
        self.custom_lbl.setAlignment(Qt.AlignCenter)
        self.custom_lbl.setStyleSheet(f"color:{T.text_dim}; font-size:11px;")
        v.addWidget(self.custom_lbl)

        # ---- Fine sliders ----
        self.sliders = {}  # attr -> (slider, value_label, scale, is_int, fmt)
        for caption, attr, lo, hi, step, scale, is_int, fmt in self.SLIDERS:
            v.addWidget(self._caption(caption))
            row = QHBoxLayout()
            sl = QSlider(Qt.Horizontal)
            sl.setMinimum(int(round(lo * scale)))
            sl.setMaximum(int(round(hi * scale)))
            sl.setSingleStep(max(1, int(round(step * scale))))
            sl.setPageStep(max(1, int(round(step * scale))))
            sl.setCursor(Qt.PointingHandCursor)
            val = QLabel("")
            val.setFixedWidth(56)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setStyleSheet(f"color:{T.accent}; font-size:12px;")
            sl.valueChanged.connect(lambda raw, a=attr: self._on_slider(a, raw))
            row.addWidget(sl)
            row.addWidget(val)
            v.addLayout(row)
            self.sliders[attr] = (sl, val, scale, is_int, fmt)

        self.dwell_cb = QCheckBox("Auto-click by hovering (dwell)")
        self.dwell_cb.toggled.connect(self._on_dwell_toggle)
        v.addWidget(self.dwell_cb)

        self.feedback_cb = QCheckBox("Show a ring when I click")
        self.feedback_cb.toggled.connect(self._on_feedback_toggle)
        v.addWidget(self.feedback_cb)

        # ---- Gesture actions (remap the click gestures) ----
        v.addWidget(self._caption("Pinch does"))
        self.pinch_combo = self._make_action_combo("pinch")
        v.addWidget(self.pinch_combo)
        v.addWidget(self._caption("Fist does"))
        self.fist_combo = self._make_action_combo("fist")
        v.addWidget(self.fist_combo)

        v.addStretch(1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.setObjectName("secondary")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(self._on_reset)
        self.save_btn = QPushButton("Save")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(self.save_btn)
        v.addLayout(btn_row)

        self._sync_from_controller()

        # Keep sliders/checkbox in sync if controller attrs change elsewhere
        # (preset applied from another path, dwell toggled on the status panel,
        # a live recalibration) while this panel stays open. The _loading guard
        # makes the programmatic setValue calls side-effect-free, and setValue to
        # the current value is a no-op so it won't fight an in-progress drag.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(300)
        self._refresh_timer.timeout.connect(self._sync_from_controller)

    def showEvent(self, event):
        self._sync_from_controller()
        self._refresh_timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._refresh_timer.stop()
        super().hideEvent(event)

    def _caption(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{T.text2}; font-size:12px;")
        return lbl

    def _on_slider(self, attr, raw):
        if self._loading:
            return
        sl, val, scale, is_int, fmt = self.sliders[attr]
        real = int(round(raw / scale)) if is_int else raw / scale
        setattr(self.controller, attr, real)
        val.setText(fmt(real))
        self._highlight_preset()

    def _on_dwell_toggle(self, checked):
        if self._loading:
            return
        self.controller.dwell_click_enabled = bool(checked)
        self.controller._reset_dwell()

    def _on_feedback_toggle(self, checked):
        if self._loading:
            return
        self.controller.click_feedback_enabled = bool(checked)

    def _make_action_combo(self, gesture):
        combo = QComboBox()
        combo.setCursor(Qt.PointingHandCursor)
        for action, label in ACTION_LABELS:
            combo.addItem(label, action)
        combo.currentIndexChanged.connect(
            lambda _idx, g=gesture, c=combo: self._on_action_changed(g, c))
        return combo

    def _on_action_changed(self, gesture, combo):
        if self._loading:
            return
        self.controller.gesture_actions[gesture] = combo.currentData()

    def _on_preset(self, key):
        self.controller.apply_preset(key)
        self._sync_from_controller()

    def _sync_from_controller(self):
        self._loading = True
        for attr, (sl, val, scale, is_int, fmt) in self.sliders.items():
            real = getattr(self.controller, attr)
            sl.setValue(int(round(real * scale)))
            val.setText(fmt(real))
        self.dwell_cb.setChecked(bool(self.controller.dwell_click_enabled))
        self.feedback_cb.setChecked(bool(getattr(self.controller, "click_feedback_enabled", True)))
        ga = self.controller.gesture_actions
        for gesture, combo, default in (("pinch", self.pinch_combo, "left_click"),
                                        ("fist", self.fist_combo, "right_click")):
            idx = combo.findData(ga.get(gesture, default))
            combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._loading = False
        self._highlight_preset()

    def _highlight_preset(self):
        active = self.controller.detect_preset()
        for key, b in self.preset_btns.items():
            b.setStyleSheet(self.PRESET_ON if key == active else self.PRESET_OFF)
        self.custom_lbl.setText("Custom — your own tuning" if active == "custom" else "")

    def _on_reset(self):
        c = self.controller
        c.sensitivity = DEFAULT_CONFIG["sensitivity"]
        c.smoothing_factor = DEFAULT_CONFIG["smoothing_factor"]
        th = DEFAULT_CONFIG["thresholds"]
        c.cursor_dead_zone = th["cursor_dead_zone"]
        c.scroll_amount = th["scroll_amount"]
        c.drag_threshold = th["drag_threshold"]
        c.action_cooldown = th["action_cooldown"]
        c.dwell_click_duration = DEFAULT_CONFIG["dwell_click"]["duration"]
        self._sync_from_controller()

    def _on_save(self):
        if self.controller.profile_name is not None:
            self.controller.save_profile()
        self.save_btn.setText("Saved")
        QTimer.singleShot(1200, lambda: self.save_btn.setText("Save"))


class ProfilesPanel(QWidget):
    """Manage saved profiles: switch, rename, duplicate, delete, set default."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("AirPoint Profiles")
        self.setFixedSize(360, 470)
        self.setStyleSheet(BASE_QSS + f" ProfilesPanel {{ background-color: {T.bg}; }}")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)

        v = QVBoxLayout(self)
        v.setContentsMargins(22, 18, 22, 16)
        v.setSpacing(10)

        header = QLabel("Profiles")
        header.setFont(_font(18, QFont.Bold))
        header.setStyleSheet(f"color: {T.accent};")
        v.addWidget(header)

        self.listw = QListWidget()
        v.addWidget(self.listw, 1)

        def mkbtn(text, slot, danger=False):
            b = QPushButton(text)
            b.setCursor(Qt.PointingHandCursor)
            b.setObjectName("danger" if danger else "secondary")
            b.clicked.connect(slot)
            return b

        row1 = QHBoxLayout()
        row1.addWidget(mkbtn("Switch to", self._switch))
        row1.addWidget(mkbtn("Set default", self._set_default))
        v.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(mkbtn("Rename", self._rename))
        row2.addWidget(mkbtn("Duplicate", self._duplicate))
        v.addLayout(row2)
        v.addWidget(mkbtn("Delete", self._delete, danger=True))

        self._refresh()

    def _selected(self):
        it = self.listw.currentItem()
        return it.data(Qt.UserRole) if it is not None else None

    def _refresh(self, select=None):
        self.listw.clear()
        default = HandCenterGestureController.get_default_profile()
        active = self.controller.profile_name
        for name in self.controller.list_profiles():
            tags = []
            if name == active:
                tags.append("active")
            if name == default:
                tags.append("default")
            label = name + (f"   ({', '.join(tags)})" if tags else "")
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, name)
            self.listw.addItem(item)
            if name == (select or active):
                self.listw.setCurrentItem(item)

    def _warn(self, msg):
        QMessageBox.warning(self, "AirPoint", msg)

    def _switch(self):
        name = self._selected()
        if not name or name == self.controller.profile_name:
            return
        if self.controller.load_profile(name):
            self._refresh()  # StatusPanel's own timer will refresh its label
            if self.controller.calibration is None:
                self._warn(f"Profile '{name}' isn't set up yet, so the pointer "
                           f"won't track properly. Use \"Redo setup\" on the main "
                           f"panel to calibrate it.")

    def _set_default(self):
        name = self._selected()
        if name and HandCenterGestureController.set_default_profile(name):
            self._refresh()

    def _rename(self):
        name = self._selected()
        if not name:
            return
        new, ok = QInputDialog.getText(self, "Rename profile", "New name:", text=name)
        if not ok:
            return
        ok2, msg, final = self.controller.rename_profile(name, new)
        if not ok2:
            self._warn(msg)
        else:
            self._refresh(select=final)

    def _duplicate(self):
        name = self._selected()
        if not name:
            return
        ok, msg, final = self.controller.duplicate_profile(name)
        if not ok:
            self._warn(msg)
        else:
            self._refresh(select=final)

    def _delete(self):
        name = self._selected()
        if not name:
            return
        resp = QMessageBox.question(
            self, "Delete profile", f"Delete profile '{name}'? This can't be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        ok, msg = self.controller.delete_profile(name)
        if not ok:
            self._warn(msg)
        else:
            self._refresh()


class ClickFeedbackOverlay(QWidget):
    """Transparent, click-through, always-on-top ring animation drawn at the
    point of each click/right-click/drag/scroll. It NEVER intercepts mouse input
    (so it can't break the clicks it visualizes) and costs nothing when idle —
    the repaint timer only runs while ripples are alive."""

    DURATION = 0.34   # seconds per ripple
    BASE_R = 10       # starting radius (px)
    GROW_R = 34       # growth over the animation
    MAX_RIPPLES = 8
    COLORS = {
        "left":   (0, 220, 200),    # cyan
        "right":  (255, 170, 68),   # orange
        "drag":   (255, 136, 204),  # pink
        "scroll": (136, 170, 255),  # blue
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ripples = []  # each: {"x","y","kind","t0"}
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
            Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setGeometry(QApplication.desktop().geometry())  # cover all monitors

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)  # ~60fps, only while ripples are live
        self._anim_timer.timeout.connect(self._on_tick)

    def flash(self, gx, gy, kind):
        """Add a ripple at global (gx, gy). Called directly from the tracker."""
        if kind not in self.COLORS:
            kind = "left"
        self._ripples.append({"x": int(gx), "y": int(gy), "kind": kind,
                              "t0": time.monotonic()})
        if len(self._ripples) > self.MAX_RIPPLES:
            self._ripples = self._ripples[-self.MAX_RIPPLES:]
        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def _on_tick(self):
        now = time.monotonic()
        self._ripples = [r for r in self._ripples if (now - r["t0"]) < self.DURATION]
        self.update()
        if not self._ripples:
            self._anim_timer.stop()

    def paintEvent(self, event):
        if not self._ripples:
            return
        now = time.monotonic()
        ox, oy = self.x(), self.y()  # virtual-desktop origin -> widget-local
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        for r in self._ripples:
            prog = (now - r["t0"]) / self.DURATION
            prog = 0.0 if prog < 0 else 1.0 if prog > 1 else prog
            radius = self.BASE_R + prog * self.GROW_R
            alpha = int(220 * (1.0 - prog))
            cr, cg, cb = self.COLORS[r["kind"]]
            cx, cy = r["x"] - ox, r["y"] - oy
            pen = QPen(QColor(cr, cg, cb, alpha))
            pen.setWidth(3)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPoint(cx, cy), int(radius), int(radius))
            if prog < 0.3:  # brief center dot
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(cr, cg, cb, alpha))
                p.drawEllipse(QPoint(cx, cy), 3, 3)
        p.end()

    def showEvent(self, event):
        super().showEvent(event)
        if sys.platform == "win32":
            # Belt-and-suspenders click-through on Windows (WindowTransparentForInput
            # has been flaky in some frozen builds): force WS_EX_TRANSPARENT.
            try:
                import ctypes
                GWL_EXSTYLE = -20
                WS_EX_TRANSPARENT = 0x00000020
                WS_EX_LAYERED = 0x00080000
                WS_EX_NOACTIVATE = 0x08000000
                hwnd = int(self.winId())
                u = ctypes.windll.user32
                ex = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
                u.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                 ex | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE)
            except Exception:
                pass


class _BenchLog:
    """Opt-in per-frame benchmark logger (enabled by --benchmark, inert otherwise).
    Records per-stage latency, FPS basis, hand-detection, raw-vs-smoothed cursor
    position (for jitter), and CPU/RAM, then writes a CSV that bench/analyze.py
    turns into latency / FPS / jitter-RMS / detection-rate figures."""

    def __init__(self, out_path, duration_s):
        self.out_path = out_path
        self.duration = float(duration_s)
        self.t_start = time.perf_counter()
        self.rows = []
        self._last_sample = 0.0
        self._cpu = 0.0
        self._mem = 0.0
        try:
            import psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)  # prime the % meter
        except Exception:
            self._proc = None  # psutil optional; cpu/mem columns stay blank

    def record(self, t0, t1, t2, t3, detected, raw, out):
        now = time.perf_counter()
        if self._proc is not None and (now - self._last_sample) > 0.25:
            try:
                self._cpu = self._proc.cpu_percent(None)
                self._mem = self._proc.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
            self._last_sample = now
        self.rows.append({
            "t_rel": round(now - self.t_start, 4),
            "capture_ms": round((t1 - t0) * 1000.0, 3),
            "inference_ms": round((t2 - t1) * 1000.0, 3),
            "post_ms": round((t3 - t2) * 1000.0, 3),
            "total_ms": round((t3 - t0) * 1000.0, 3),
            "detected": int(bool(detected)),
            "raw_x": "" if raw is None else round(raw[0], 2),
            "raw_y": "" if raw is None else round(raw[1], 2),
            "out_x": "" if out is None else round(out[0], 2),
            "out_y": "" if out is None else round(out[1], 2),
            "cpu_pct": round(self._cpu, 1) if self._proc else "",
            "mem_mb": round(self._mem, 1) if self._proc else "",
        })

    def done(self):
        return (time.perf_counter() - self.t_start) >= self.duration

    def save(self):
        if not self.rows:
            return None
        import csv as _csv
        d = os.path.dirname(self.out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        return self.out_path


class HandCenterGestureController:
    @staticmethod
    def _show_startup_error(title, message, settings_url=None, settings_label="Open Settings"):
        """Show a startup error dialog using PyQt5 (or tkinter as fallback).
        If settings_url is provided, an extra button opens that URL so the user
        can jump straight to the right system settings pane.
        """
        try:
            app = QApplication.instance() or QApplication(sys.argv)
            apply_app_theme(app)
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle(title)
            msg.setText(message)
            settings_btn = None
            if settings_url:
                settings_btn = msg.addButton(settings_label, QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Ok)
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {T.bg}; color: {T.text}; }}
                QLabel {{ color: {T.text}; font-size: 13px; }}
                QPushButton {{ background-color: {T.surface}; color: {T.text}; border: 1px solid {T.border};
                              border-radius: 6px; padding: 6px 18px; min-width: 80px; }}
                QPushButton:hover {{ background-color: {T.surface_hover}; }}
            """)
            msg.exec_()
            if settings_btn is not None and msg.clickedButton() is settings_btn:
                try:
                    if sys.platform == "darwin":
                        subprocess.Popen(["open", settings_url])
                    elif sys.platform == "win32":
                        os.startfile(settings_url)
                    else:
                        import webbrowser
                        webbrowser.open(settings_url)
                except Exception:
                    pass
        except Exception:
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(title, message)
                root.destroy()
            except Exception:
                print(f"{title}\n{message}")

    def __init__(self, enable_gaze_detection=True):
        # Apply all defaults from DEFAULT_CONFIG first (sets every configurable attribute)
        self._apply_config(DEFAULT_CONFIG)

        # Override gaze setting from constructor arg
        self.gaze_detection_enabled = enable_gaze_detection

        # Non-configurable state
        self.face_detected = False
        self.looking_at_screen = False
        self.face_detection_history = deque(maxlen=5)
        self.gaze_cooldown = 0

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Face detection for gaze awareness (only if enabled)
        if self.gaze_detection_enabled:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.mp_face_mesh = None
            self.face_mesh = None

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            cam_url = (
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
                if sys.platform == "darwin"
                else "ms-settings:privacy-webcam"
                if sys.platform == "win32"
                else None
            )
            self._show_startup_error(
                S("crash_title"),
                "AirPoint could not access your camera.\n\n"
                "1. Make sure your computer has a working camera.\n"
                "2. Close any other app that may be using it (Zoom, Teams, FaceTime, browser tabs, etc.).\n"
                "3. Allow camera access for AirPoint in your system settings, then relaunch.",
                settings_url=cam_url,
                settings_label="Open Camera Settings",
            )
            raise SystemExit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Verify we can actually read a frame (catches macOS permission denied)
        ret, _ = self.cap.read()
        if not ret:
            self.cap.release()
            cam_url = (
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
                if sys.platform == "darwin"
                else "ms-settings:privacy-webcam"
                if sys.platform == "win32"
                else None
            )
            self._show_startup_error(
                S("crash_title"),
                "AirPoint could not read from your camera.\n\n"
                "This usually means camera permission is off.\n\n"
                "macOS: System Settings, then Privacy & Security, then Camera. Turn on AirPoint (or Terminal/Python if running from source).\n\n"
                "Windows: Settings, then Privacy & security, then Camera. Make sure camera access is allowed.\n\n"
                "Then relaunch AirPoint.",
                settings_url=cam_url,
                settings_label="Open Camera Settings",
            )
            raise SystemExit(1)

        # Control settings
        # FAILSAFE intentionally disabled: in a gesture-driven mouse, brief tracking
        # glitches can fling the cursor to (0,0) and would kill the app. The user
        # can always quit with 'q' in the camera window or by closing the status panel.
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        # macOS: check Accessibility permission (needed for cursor control).
        # Uses AXIsProcessTrusted from ApplicationServices — the real Accessibility
        # check. (The previous osascript-based check tested Automation permission,
        # which is a different grant.)
        if sys.platform == "darwin":
            try:
                import ctypes
                import ctypes.util
                lib_path = ctypes.util.find_library("ApplicationServices")
                if lib_path:
                    appkit = ctypes.cdll.LoadLibrary(lib_path)
                    appkit.AXIsProcessTrusted.restype = ctypes.c_bool
                    is_trusted = appkit.AXIsProcessTrusted()
                else:
                    is_trusted = True  # framework unavailable — assume OK
                if not is_trusted:
                    self.cap.release()
                    self._show_startup_error(
                        S("crash_title"),
                        "AirPoint needs Accessibility permission to move the cursor.\n\n"
                        "Open System Settings, then Privacy & Security, then Accessibility, and turn on AirPoint (or Terminal/Python if you're running from source).\n\n"
                        "Then relaunch AirPoint.",
                        settings_url="x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
                        settings_label="Open Accessibility Settings",
                    )
                    raise SystemExit(1)
            except SystemExit:
                raise
            except Exception:
                pass  # If check itself fails, don't block — let pyautogui try anyway

        # Screen dimensions (DPI-aware on Windows)
        self.screen_width, self.screen_height = pyautogui.size()

        # Gesture state (not configurable — runtime state)
        self.pinch_start_time = None
        self._pinch_active = False  # Schmitt-trigger latch for pinch hysteresis
        self.is_dragging = False
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        self.prev_hand_center = None
        self.last_action_time = 0
        self.fist_history = deque(maxlen=8)
        self.overlay = None  # ClickFeedbackOverlay, created in run() once a QApplication exists
        self.paused = False  # when True the tracking tick does no detection/cursor control
        self._bench = None          # _BenchLog while --benchmark is active
        self._bench_seconds = 0     # >0 enables benchmark logging in run()
        self._bench_out = None      # optional CSV path for --benchmark

        # Two-finger scroll state
        self.scroll_reference_y = None
        self.scroll_accumulated = 0
        self.scroll_exit_counter = 0
        self.scroll_enter_counter = 0  # consecutive in-pose frames before scroll arms

        # Profile tracking
        self.profile_name = None

        # EMA cursor smoothing state (double-EMA: two cascaded passes)
        self.smoothed_screen_pos = None
        self._smoothed_pass2 = None  # second EMA pass for extra jitter removal
        self._last_output_pos = None  # for dead-zone filtering
        self._prev_raw_pos = None  # for velocity-adaptive smoothing

        # Dwell-click state
        self.dwell_reference_pos = None
        self.dwell_start_time = None
        self.dwell_triggered = False

        print("AirPoint - Hand Center Tracking Controller")
        print(f"  Gaze detection: {'ON' if self.gaze_detection_enabled else 'OFF'}")
        print(f"  Dwell-click: {'ON' if self.dwell_click_enabled else 'OFF'}"
              f" (radius={self.dwell_click_radius}px, duration={self.dwell_click_duration}s)")
        print(f"  Dead zone: {self.cursor_dead_zone}px | Smoothing: {self.smoothing_factor}")
        print("  Gestures: open hand=move, pinch=click, hold pinch=drag, fist=right-click")
        print("  Two fingers (index+middle) = scroll")
        print("  Keys: 'g'=toggle gaze, 'c'=switch profile, 'd'=toggle dwell, 'q'=quit")

    def toggle_gaze_detection(self):
        """Toggle gaze detection on/off"""
        self.gaze_detection_enabled = not self.gaze_detection_enabled

        if self.gaze_detection_enabled:
            # Initialize face detection if it wasn't already
            if self.mp_face_mesh is None:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            print("👁️ GAZE DETECTION ENABLED - Only works when looking at screen")
        else:
            print("👁️ GAZE DETECTION DISABLED - Always active")
            # Reset gaze state to allow control
            self.looking_at_screen = True
            self.face_detected = True

        # Auto-save preference to profile if one is loaded
        if self.profile_name is not None:
            self.save_profile()

    def toggle_dwell_click(self):
        """Toggle dwell-click on/off"""
        self.dwell_click_enabled = not self.dwell_click_enabled
        self._reset_dwell()
        state = "ON" if self.dwell_click_enabled else "OFF"
        print(f"Dwell-click: {state} (radius={self.dwell_click_radius}px, duration={self.dwell_click_duration}s)")
        # Auto-save preference to profile if one is loaded
        if self.profile_name is not None:
            self.save_profile()

    def toggle_pause(self):
        """Pause/resume tracking. Paused = the tick does no detection or cursor
        control, so the cursor parks where it is until resumed."""
        self.paused = not self.paused
        print(f"Tracking: {'PAUSED' if self.paused else 'RESUMED'}")

    def _apply_config(self, config):
        """Apply a config dict to instance attributes, filling missing keys from DEFAULT_CONFIG."""
        merged = copy.deepcopy(DEFAULT_CONFIG)
        for key in config:
            if config[key] is not None and isinstance(config[key], dict) and isinstance(merged.get(key), dict):
                merged[key].update(config[key])
            else:
                merged[key] = config[key]

        # Clamp user-tunable numerics so a corrupt / hand-edited / out-of-range
        # profile can never crash the loop or make the cursor uncontrollable.
        def _clamp(v, lo, hi, default):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(v):
                return default
            return max(lo, min(hi, v))

        th = merged["thresholds"]
        self.sensitivity = _clamp(merged["sensitivity"], 0.5, 5.0, DEFAULT_CONFIG["sensitivity"])
        self.smoothing_factor = _clamp(merged["smoothing_factor"], 0.0, 0.95, DEFAULT_CONFIG["smoothing_factor"])
        self.drag_threshold = _clamp(th["drag_threshold"], 0.1, 2.0, 0.4)
        self.action_cooldown = _clamp(th["action_cooldown"], 0.05, 1.0, 0.15)
        self.pinch_threshold = th["pinch_threshold"]   # may be None (calibration-derived)
        self.fist_threshold = th["fist_threshold"]     # may be None
        self.scroll_dead_zone = _clamp(th["scroll_dead_zone"], 0.0, 0.2, 0.015)
        self.scroll_threshold_val = _clamp(th["scroll_threshold"], 0.005, 0.2, 0.035)
        self.scroll_amount = int(_clamp(th["scroll_amount"], 1, 10, 2))
        self.screen_edge_margin = int(_clamp(th["screen_edge_margin"], 0, 200, 20))
        self.cursor_dead_zone = int(_clamp(th["cursor_dead_zone"], 0, 60, 10))
        self.calibration_margin = _clamp(th["calibration_margin"], 0.0, 0.5, 0.05)

        dw = merged["dwell_click"]
        self.dwell_click_enabled = bool(dw["enabled"])
        self.dwell_click_radius = int(_clamp(dw["radius"], 5, 200, 30))
        self.dwell_click_duration = _clamp(dw["duration"], 0.3, 6.0, 1.5)

        self.gaze_detection_enabled = merged["gaze_detection_enabled"]
        self.click_feedback_enabled = merged.get("click_feedback", True)
        self.gesture_actions = dict(merged["gesture_actions"])
        self.calibration = merged["calibration"]

    @staticmethod
    def _migrate_v0_profile(old):
        """Convert old flat profile format (no schema_version) to new nested schema."""
        new = copy.deepcopy(DEFAULT_CONFIG)
        new["schema_version"] = 1
        # Old format requires at minimum: left, right, top, bottom
        if all(k in old for k in ("left", "right", "top", "bottom")):
            new["calibration"] = {
                "left": old["left"],
                "right": old["right"],
                "top": old["top"],
                "bottom": old["bottom"],
                "tremor_std": old.get("tremor_std"),
                "calibration_margin": old.get("calibration_margin", 0.05),
            }
        else:
            print("  Warning: old profile missing calibration keys, using uncalibrated defaults")
            new["calibration"] = None
        if "smoothing_factor" in old:
            new["smoothing_factor"] = old["smoothing_factor"]
        if "pinch_threshold" in old:
            new["thresholds"]["pinch_threshold"] = old["pinch_threshold"]
        if "fist_threshold" in old:
            new["thresholds"]["fist_threshold"] = old["fist_threshold"]
        return new

    @staticmethod
    def list_profiles():
        """Return sorted list of profile names found in the profiles directory."""
        if not os.path.isdir(PROFILES_DIR):
            return []
        names = []
        for f in sorted(os.listdir(PROFILES_DIR)):
            if f.endswith(".json"):
                names.append(f[:-5])  # strip .json
        return names

    def load_profile(self, name):
        """Load a profile by name. Auto-migrates old flat format to new schema."""
        path = os.path.join(PROFILES_DIR, f"{name}.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Corrupted profile '{name}': {e}")
            return False

        if not isinstance(raw, dict):
            print(f"Error: Profile '{name}' is not a valid JSON object")
            return False

        # Detect and migrate old-format profiles
        if "schema_version" not in raw:
            print(f"  Migrating old profile '{name}' to new schema...")
            raw = self._migrate_v0_profile(raw)

        self._apply_config(raw)
        self.profile_name = name

        # Restore language preference
        if "language" in raw:
            set_language(raw["language"])

        # Reset smoothing state for new profile
        self.smoothed_screen_pos = None
        self._smoothed_pass2 = None
        self._prev_raw_pos = None
        self._last_output_pos = None
        self.dwell_reference_pos = None
        self.dwell_start_time = None
        self.dwell_triggered = False

        # Reset gesture runtime latches too, so switching profiles live (Profiles
        # panel) can't strand a held mouse button or carry stale pinch/scroll
        # pose state into the new profile (mirrors the hand-lost cleanup).
        if getattr(self, "is_dragging", False):
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
            self.is_dragging = False
        self.pinch_start_time = None
        self._pinch_active = False
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        self.scroll_reference_y = None
        self.scroll_accumulated = 0
        self.scroll_exit_counter = 0
        self.scroll_enter_counter = 0

        # Print summary
        cal = self.calibration
        if cal and all(k in cal for k in ('left', 'right', 'top', 'bottom')):
            tremor_str = f", tremor={cal['tremor_std']:.5f}" if cal.get("tremor_std") else ""
            print(f"Loaded profile '{name}': L={cal['left']:.3f} R={cal['right']:.3f} "
                  f"T={cal['top']:.3f} B={cal['bottom']:.3f}{tremor_str}")
        else:
            print(f"Loaded profile '{name}' (uncalibrated)")
        pinch_str = f"{self.pinch_threshold:.4f}" if self.pinch_threshold is not None else "DISABLED"
        fist_str = f"{self.fist_threshold:.4f}" if self.fist_threshold is not None else "DISABLED"
        print(f"  smoothing={self.smoothing_factor:.2f}, pinch={pinch_str}, fist={fist_str}"
              f", dwell={'ON' if self.dwell_click_enabled else 'OFF'}"
              f", dead_zone={self.cursor_dead_zone}px")
        return True

    def save_profile(self):
        """Save full config (calibration + all settings) to the current profile file."""
        os.makedirs(PROFILES_DIR, exist_ok=True)
        config = {
            "schema_version": 1,
            "calibration": self.calibration,
            "sensitivity": self.sensitivity,
            "smoothing_factor": self.smoothing_factor,
            "thresholds": {
                "pinch_threshold": self.pinch_threshold,
                "fist_threshold": self.fist_threshold,
                "drag_threshold": self.drag_threshold,
                "action_cooldown": self.action_cooldown,
                "scroll_dead_zone": self.scroll_dead_zone,
                "scroll_threshold": self.scroll_threshold_val,
                "scroll_amount": self.scroll_amount,
                "screen_edge_margin": self.screen_edge_margin,
                "cursor_dead_zone": self.cursor_dead_zone,
                "calibration_margin": self.calibration_margin,
            },
            "dwell_click": {
                "enabled": self.dwell_click_enabled,
                "radius": self.dwell_click_radius,
                "duration": self.dwell_click_duration,
            },
            "gaze_detection_enabled": self.gaze_detection_enabled,
            "click_feedback": self.click_feedback_enabled,
            "gesture_actions": self.gesture_actions,
            "language": _current_lang,
        }
        path = os.path.join(PROFILES_DIR, f"{self.profile_name}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Profile '{self.profile_name}' saved to {path}")

    # ---- Sensitivity presets ----

    def apply_preset(self, name):
        """Apply a named 'pointer feel' preset to the live attributes and persist."""
        preset = PRESETS.get(name)
        if preset is None:
            return False
        self.sensitivity = preset["sensitivity"]
        self.smoothing_factor = preset["smoothing_factor"]
        self.cursor_dead_zone = preset["cursor_dead_zone"]
        self.drag_threshold = preset["drag_threshold"]
        self.action_cooldown = preset["action_cooldown"]
        self.scroll_amount = preset["scroll_amount"]
        self.dwell_click_duration = preset["dwell_duration"]
        # Reset smoothing / dead-zone state so the new alpha & radius start clean.
        self.smoothed_screen_pos = None
        self._smoothed_pass2 = None
        self._prev_raw_pos = None
        self._last_output_pos = None
        if self.profile_name is not None:
            self.save_profile()
        print(f"Preset applied: {name}")
        return True

    def detect_preset(self):
        """Return the preset key matching the current live values, else 'custom'."""
        def close(a, b):
            return abs(float(a) - float(b)) < 1e-6
        for key, p in PRESETS.items():
            if (close(self.sensitivity, p["sensitivity"]) and
                    close(self.smoothing_factor, p["smoothing_factor"]) and
                    close(self.cursor_dead_zone, p["cursor_dead_zone"]) and
                    close(self.drag_threshold, p["drag_threshold"]) and
                    close(self.action_cooldown, p["action_cooldown"]) and
                    int(self.scroll_amount) == int(p["scroll_amount"]) and
                    close(self.dwell_click_duration, p["dwell_duration"])):
                return key
        return "custom"

    # ---- Profile management (rename / delete / duplicate / default) ----

    @staticmethod
    def _sanitize_name(raw):
        """Clean a user-supplied profile name (same rule as the setup wizard)."""
        name = "".join(c for c in str(raw) if c.isalnum() or c in " _-").strip()
        return name[:20].strip() or "default"

    @staticmethod
    def _default_ptr_path():
        return os.path.join(PROFILES_DIR, "default_profile.txt")

    @staticmethod
    def get_default_profile():
        try:
            with open(HandCenterGestureController._default_ptr_path(), "r", encoding="utf-8") as f:
                name = f.read().strip()
        except OSError:
            return None
        if name and os.path.exists(os.path.join(PROFILES_DIR, f"{name}.json")):
            return name
        return None

    @staticmethod
    def set_default_profile(name):
        if not os.path.exists(os.path.join(PROFILES_DIR, f"{name}.json")):
            return False
        try:
            os.makedirs(PROFILES_DIR, exist_ok=True)
            with open(HandCenterGestureController._default_ptr_path(), "w", encoding="utf-8") as f:
                f.write(name)
            return True
        except OSError:
            return False

    @staticmethod
    def clear_default_profile():
        try:
            os.remove(HandCenterGestureController._default_ptr_path())
        except OSError:
            pass

    def delete_profile(self, name):
        """Delete a profile file. Returns (ok, message)."""
        if name == self.profile_name:
            return False, "Can't delete the profile you're currently using."
        path = os.path.join(PROFILES_DIR, f"{name}.json")
        if not os.path.exists(path):
            return False, "That profile no longer exists."
        try:
            os.remove(path)
        except OSError as e:
            return False, str(e)
        if HandCenterGestureController.get_default_profile() == name:
            HandCenterGestureController.clear_default_profile()
        return True, ""

    def rename_profile(self, old, new):
        """Rename a profile. Returns (ok, message, final_name)."""
        new_name = self._sanitize_name(new)
        src = os.path.join(PROFILES_DIR, f"{old}.json")
        if not os.path.exists(src):
            return False, "That profile no longer exists.", old
        if new_name == old:
            return True, "", old  # no change
        case_only = new_name.lower() == old.lower()
        dst = os.path.join(PROFILES_DIR, f"{new_name}.json")
        if os.path.exists(dst) and not case_only:
            return False, f"A profile named '{new_name}' already exists.", old
        try:
            if case_only:  # two-step for case-insensitive filesystems (Win/macOS)
                tmp = os.path.join(PROFILES_DIR, f"{old}.__tmp__.json")
                if os.path.exists(tmp):
                    os.remove(tmp)  # clear any stale temp from a prior failure
                os.rename(src, tmp)
                os.rename(tmp, dst)
            else:
                os.rename(src, dst)
        except OSError as e:
            return False, str(e), old
        if self.profile_name == old:
            self.profile_name = new_name
        if HandCenterGestureController.get_default_profile() == old:
            HandCenterGestureController.set_default_profile(new_name)
        return True, "", new_name

    def duplicate_profile(self, name, newname=None):
        """Copy a profile to a new, non-colliding name. Returns (ok, message, new_name)."""
        src = os.path.join(PROFILES_DIR, f"{name}.json")
        if not os.path.exists(src):
            return False, "That profile no longer exists.", None
        base = self._sanitize_name(newname) if newname else self._sanitize_name(f"{name} copy")
        final, i = base, 2
        while os.path.exists(os.path.join(PROFILES_DIR, f"{final}.json")):
            # Reserve room for the suffix so _sanitize_name's 20-char cap can't
            # truncate it away (which would spin the loop on a long base name).
            suffix = f" {i}"
            final = self._sanitize_name(base[:20 - len(suffix)].rstrip() + suffix)
            i += 1
            if i > 99:
                return False, "Couldn't find a free name.", None
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = f.read()
            with open(os.path.join(PROFILES_DIR, f"{final}.json"), "w", encoding="utf-8") as f:
                f.write(data)
        except OSError as e:
            return False, str(e), None
        return True, "", final

    # ---- Click feedback + remappable gesture actions ----

    def _emit_click(self, kind):
        """Flash a click-feedback ripple at the current cursor (if enabled)."""
        if self.overlay is None or not getattr(self, "click_feedback_enabled", True):
            return
        try:
            x, y = pyautogui.position()
            self.overlay.flash(x, y, kind)
        except Exception:
            pass

    def _do_action(self, action):
        """Perform a discrete click action (for the remappable pinch / fist
        gestures) and flash matching feedback. 'none'/unknown is a no-op."""
        try:
            if action == "left_click":
                pyautogui.click(); kind = "left"
            elif action == "right_click":
                pyautogui.rightClick(); kind = "right"
            elif action == "double_click":
                pyautogui.doubleClick(); kind = "left"
            elif action == "middle_click":
                pyautogui.middleClick(); kind = "left"
            else:  # "none" or unrecognized
                return
        except Exception as e:
            print(f"Action '{action}' failed: {e}")
            return
        self._emit_click(kind)

    def _fatal_exit(self, title, message, settings_url=None, settings_label="Open Settings"):
        """Stop tracking, then show a final error dialog and quit — without
        re-entrant stacking. Called from inside the tracking tick on
        unrecoverable camera/tracking loss; stopping the timer BEFORE the modal
        dialog prevents the tick from re-firing during exec_() and piling up
        dialogs (and trapping a user who has just lost their only pointer)."""
        if getattr(self, "_fatal_shown", False):
            return
        self._fatal_shown = True
        t = getattr(self, "_tracking_timer", None)
        if t is not None:
            t.stop()
        try:
            self._show_startup_error(title, message,
                                     settings_url=settings_url,
                                     settings_label=settings_label)
        finally:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def map_to_screen(self, hand_x, hand_y):
        """Map hand center coordinates to screen position using calibration bounding box,
        then apply EMA smoothing based on the tremor-derived smoothing_factor."""
        if self.calibration is None:
            # Fallback: use full normalized range
            raw_x = hand_x * self.screen_width
            raw_y = hand_y * self.screen_height
        else:
            cal = self.calibration
            # Add a small margin so edges are reachable
            range_x = cal["right"] - cal["left"]
            range_y = cal["bottom"] - cal["top"]

            # Guard against zero range (bad calibration data)
            if range_x == 0 or range_y == 0:
                raw_x = hand_x * self.screen_width
                raw_y = hand_y * self.screen_height
            else:
                margin_x = range_x * self.calibration_margin
                margin_y = range_y * self.calibration_margin

                norm_x = (hand_x - (cal["left"] - margin_x)) / (range_x + 2 * margin_x)
                norm_y = (hand_y - (cal["top"] - margin_y)) / (range_y + 2 * margin_y)

                # Clamp to [0, 1]
                norm_x = max(0.0, min(1.0, norm_x))
                norm_y = max(0.0, min(1.0, norm_y))

                raw_x = norm_x * self.screen_width
                raw_y = norm_y * self.screen_height

        # Velocity-adaptive smoothing: more smoothing when hand is slow (tremor),
        # less smoothing when hand moves fast (intentional movement).
        base_alpha = self.smoothing_factor  # e.g. 0.65
        if self._prev_raw_pos is not None:
            velocity = math.sqrt((raw_x - self._prev_raw_pos[0]) ** 2 +
                                 (raw_y - self._prev_raw_pos[1]) ** 2)
            # Map velocity to alpha: slow movement → alpha up to 0.85, fast → base_alpha or lower
            # Threshold of ~80px/frame distinguishes tremor from intentional movement
            speed_ratio = min(velocity / 80.0, 1.0)
            alpha = base_alpha + (0.85 - base_alpha) * (1.0 - speed_ratio)
        else:
            alpha = base_alpha
        self._prev_raw_pos = [raw_x, raw_y]

        # First EMA pass
        if self.smoothed_screen_pos is None:
            self.smoothed_screen_pos = [raw_x, raw_y]
        else:
            self.smoothed_screen_pos[0] = alpha * self.smoothed_screen_pos[0] + (1 - alpha) * raw_x
            self.smoothed_screen_pos[1] = alpha * self.smoothed_screen_pos[1] + (1 - alpha) * raw_y

        # Second EMA pass (double-EMA) for extra jitter removal
        alpha2 = base_alpha * 0.8  # lighter second pass to avoid excessive lag
        if self._smoothed_pass2 is None:
            self._smoothed_pass2 = list(self.smoothed_screen_pos)
        else:
            self._smoothed_pass2[0] = alpha2 * self._smoothed_pass2[0] + (1 - alpha2) * self.smoothed_screen_pos[0]
            self._smoothed_pass2[1] = alpha2 * self._smoothed_pass2[1] + (1 - alpha2) * self.smoothed_screen_pos[1]

        # Radial dead-zone with a soft ease-out. Inside `inner` the cursor holds
        # still (kills resting tremor). Between inner and outer it eases out
        # proportionally instead of snapping when the boundary is crossed, so a
        # shaky hand can settle onto a small target instead of getting stuck just
        # shy of it. Beyond outer the smoothed position passes straight through.
        # (Replaces the old anisotropic dx<dz AND dy<dz square latch.)
        target_x, target_y = self._smoothed_pass2[0], self._smoothed_pass2[1]
        if self._last_output_pos is not None:
            dx = target_x - self._last_output_pos[0]
            dy = target_y - self._last_output_pos[1]
            dist = math.hypot(dx, dy)
            inner = self.cursor_dead_zone
            outer = 2.0 * self.cursor_dead_zone
            if dist < inner:
                return self._last_output_pos[0], self._last_output_pos[1]
            elif dist < outer:
                frac = (dist - inner) / (outer - inner)
                target_x = self._last_output_pos[0] + dx * frac
                target_y = self._last_output_pos[1] + dy * frac

        # Clamp to screen bounds with margin
        m = self.screen_edge_margin
        screen_x = max(m, min(self.screen_width - m, target_x))
        screen_y = max(m, min(self.screen_height - m, target_y))
        self._last_output_pos = [screen_x, screen_y]
        return screen_x, screen_y

    def get_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        return np.array(landmarks)

    def calculate_hand_center(self, landmarks):
        """Calculate the center of the hand using key landmarks"""
        # Use palm landmarks and wrist for a stable center point
        key_points = [
            landmarks[0],   # Wrist
            landmarks[5],   # Index MCP
            landmarks[9],   # Middle MCP
            landmarks[13],  # Ring MCP
            landmarks[17],  # Pinky MCP
        ]

        center_x = np.mean([point[0] for point in key_points])
        center_y = np.mean([point[1] for point in key_points])

        return [center_x, center_y]

    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def count_extended_fingers(self, landmarks):
        """Count extended fingers with LOOSER thresholds for better two-finger detection"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]

        extended_fingers = 0
        finger_states = []
        wrist = landmarks[0]

        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # Thumb - make it harder to be "extended"
                thumb_wrist_dist = self.calculate_distance(landmarks[tip], wrist)
                thumb_pip_wrist_dist = self.calculate_distance(landmarks[pip], wrist)
                if thumb_wrist_dist > thumb_pip_wrist_dist + 0.03:  # Increased from 0.02
                    extended_fingers += 1
                    finger_states.append(True)
                else:
                    finger_states.append(False)
            else:  # Other fingers - make it easier to be "extended"
                if landmarks[tip][1] < landmarks[pip][1] + 0.01:  # Reduced from 0.02
                    extended_fingers += 1
                    finger_states.append(True)
                else:
                    finger_states.append(False)

        return extended_fingers, finger_states

    def detect_fist(self, landmarks):
        """Simple fist detection using personal threshold"""
        if self.fist_threshold is None:
            return False
        palm_center = landmarks[9]
        fingertip_distances = []
        for tip_idx in [4, 8, 12, 16, 20]:
            dist = self.calculate_distance(landmarks[tip_idx], palm_center)
            fingertip_distances.append(dist)

        avg_distance = np.mean(fingertip_distances)
        extended_count, _ = self.count_extended_fingers(landmarks)

        return extended_count <= 1 and avg_distance < self.fist_threshold

    def detect_open_hand(self, landmarks):
        """Simple open hand detection"""
        extended_count, _ = self.count_extended_fingers(landmarks)
        return extended_count >= 3

    def detect_face_and_gaze(self, frame):
        """Detect if user's face is visible and roughly looking at screen"""
        # If gaze detection is disabled, always return True
        if not self.gaze_detection_enabled:
            self.face_detected = True
            self.looking_at_screen = True
            return True

        # If face_mesh is not initialized, return True (fallback)
        if self.face_mesh is None:
            self.face_detected = True
            self.looking_at_screen = True
            return True

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)

        face_detected = False
        looking_forward = False

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                face_detected = True

                # Get key landmarks for gaze estimation
                landmarks = face_landmarks.landmark

                # Key points for gaze direction (simplified approach)
                nose_tip = landmarks[1]  # Nose tip
                left_eye = landmarks[33]  # Left eye corner
                right_eye = landmarks[263]  # Right eye corner

                # Simple forward-facing detection based on eye symmetry and nose position
                eye_distance = abs(left_eye.x - right_eye.x)
                nose_center_x = nose_tip.x
                face_center_x = (left_eye.x + right_eye.x) / 2

                # Check if face is roughly centered and forward-facing
                symmetry_threshold = 0.02
                nose_offset = abs(nose_center_x - face_center_x)
                eye_width_threshold = 0.08  # Minimum eye distance for forward face

                if eye_distance > eye_width_threshold and nose_offset < symmetry_threshold:
                    looking_forward = True

                break

        # Update detection history for smoothing
        self.face_detection_history.append((face_detected, looking_forward))

        # Smooth decision based on recent history
        recent_detections = list(self.face_detection_history)
        if len(recent_detections) >= 3:
            face_count = sum(1 for fd, _ in recent_detections if fd)
            gaze_count = sum(1 for _, lf in recent_detections if lf)

            # Need majority of recent frames to have face + forward gaze
            self.face_detected = face_count >= 3
            self.looking_at_screen = gaze_count >= 2

        return self.face_detected and self.looking_at_screen

    def _reset_dwell(self):
        """Reset dwell-click state."""
        self.dwell_reference_pos = None
        self.dwell_start_time = None
        self.dwell_triggered = False

    def is_safe_to_control(self):
        """Check if it's safe to perform gestures"""
        # If gaze detection is disabled, always return True
        if not self.gaze_detection_enabled:
            return True

        # Otherwise, check if user is looking at screen
        return self.looking_at_screen

    def detect_two_finger_scroll(self, landmarks):
        """Detect two-finger scroll gesture - index and middle up, IGNORE thumb position"""
        extended_count, finger_states = self.count_extended_fingers(landmarks)

        # Check: index + middle up, ring + pinky down, IGNORE thumb completely
        is_two_finger_pose = (
            finger_states[1] and  # Index finger extended (MUST have)
            finger_states[2] and  # Middle finger extended (MUST have)
            not finger_states[3] and  # Ring finger not extended
            not finger_states[4]      # Pinky not extended
            # DON'T CARE about thumb (finger_states[0]) at all!
        )

        if not is_two_finger_pose:
            # Out of pose — require a fresh run of in-pose frames before re-arming.
            self.scroll_enter_counter = 0
            # Only reset if we've been out of pose for a bit (sticky mode)
            if not hasattr(self, 'scroll_exit_counter'):
                self.scroll_exit_counter = 0

            self.scroll_exit_counter += 1

            # Give it 3 frames of grace before exiting (prevents accidental exits)
            if self.scroll_exit_counter > 3:
                if self.scroll_reference_y is not None:
                    print("📱 Exiting scroll mode")
                self.scroll_reference_y = None
                self.scroll_accumulated = 0
                self.scroll_exit_counter = 0
            return False
        else:
            # Reset exit counter when back in pose
            self.scroll_exit_counter = 0

        # Calculate average Y position of index and middle fingertips
        index_tip_y = landmarks[8][1]  # Index fingertip
        middle_tip_y = landmarks[12][1]  # Middle fingertip
        current_fingers_y = (index_tip_y + middle_tip_y) / 2

        # Initialize reference position on first detection — but only after a
        # short entry debounce. Exit already has a 3-frame grace; entry used to
        # be instant, so a single mis-classified frame (mid-pinch, hand reshaping)
        # could hijack the cursor into scroll mode and capture a bogus reference.
        # Require 2 consecutive in-pose frames before arming; until then keep
        # cursor control (return False).
        if self.scroll_reference_y is None:
            self.scroll_enter_counter += 1
            if self.scroll_enter_counter < 2:
                return False
            self.scroll_reference_y = current_fingers_y
            self.scroll_accumulated = 0
            print("📱 Two-finger scroll mode activated")
            return True

        # Calculate movement since reference
        movement = current_fingers_y - self.scroll_reference_y

        # Apply dead zone threshold (smaller movements ignored)
        dead_zone = self.scroll_dead_zone
        if abs(movement) < dead_zone:
            return True  # In scroll mode but not moving enough

        # Accumulate movement for bigger-movement requirement
        self.scroll_accumulated += movement

        # Require larger accumulated movement before scrolling
        scroll_threshold = self.scroll_threshold_val

        if abs(self.scroll_accumulated) >= scroll_threshold:
            # Determine scroll direction and amount
            if self.scroll_accumulated > 0:
                # Fingers moved down -> scroll down
                scroll_amount = -self.scroll_amount
                direction = "down"
            else:
                # Fingers moved up -> scroll up
                scroll_amount = self.scroll_amount
                direction = "up"

            # Perform the scroll
            pyautogui.scroll(scroll_amount)
            self._emit_click("scroll")
            print(f"📜 Two-finger scroll {direction} (movement: {self.scroll_accumulated:.3f})")

            # Reset accumulator but keep reference for continuous scrolling
            self.scroll_accumulated = 0
            self.scroll_reference_y = current_fingers_y

        return True

    def detect_gestures(self, landmarks):
        """Gesture detection using HAND CENTER tracking - respects gaze setting"""

        # SAFETY CHECK: Only proceed if it's safe to control
        if not self.is_safe_to_control():
            # Clean up any active gestures for safety
            if self.is_dragging:
                try:
                    pyautogui.mouseUp(button='left')
                    if self.gaze_detection_enabled:
                        print("🛑 SAFETY: Stopped drag - user not looking at screen")
                    else:
                        print("🛑 Stopped drag")
                except Exception:
                    pass
                self.is_dragging = False
                self.drag_start_hand_pos = None
                self.drag_start_screen_pos = None
                self.pinch_start_time = None

            # Reset scroll mode
            if self.scroll_reference_y is not None:
                if self.gaze_detection_enabled:
                    print("🛑 SAFETY: Exited scroll - user not looking at screen")
                else:
                    print("🛑 Exited scroll")
                self.scroll_reference_y = None
                self.scroll_accumulated = 0

            # Reset other states
            self.prev_hand_center = None
            self.smoothed_screen_pos = None
            self._smoothed_pass2 = None
            self._prev_raw_pos = None
            self._last_output_pos = None
            self._pinch_active = False
            self.scroll_enter_counter = 0
            self.dwell_reference_pos = None
            self.dwell_start_time = None
            self.dwell_triggered = False
            return "safety_disabled" if self.gaze_detection_enabled else "disabled"

        current_time = time.time()

        # Calculate hand center
        hand_center = self.calculate_hand_center(landmarks)

        # Get key finger positions for pinch detection
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        # Calculate pinch — with Schmitt-trigger hysteresis. Enter the pinched
        # state below the threshold, but only LEAVE it once the fingers open past
        # threshold * 1.25. A single hard compare made a hand resting near the
        # threshold flicker pinched/unpinched every frame, firing false clicks and
        # flickering drag start/stop — the hysteresis band absorbs that jitter.
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        if self.pinch_threshold is None:
            self._pinch_active = False
        elif self._pinch_active:
            if pinch_distance > self.pinch_threshold * 1.25:
                self._pinch_active = False
        else:
            if pinch_distance < self.pinch_threshold:
                self._pinch_active = True
        is_pinched = self._pinch_active

        extended_count, finger_states = self.count_extended_fingers(landmarks)

        # 1. FIST DETECTION for RIGHT CLICK
        is_fist = self.detect_fist(landmarks)
        self.fist_history.append(is_fist)

        if len(self.fist_history) >= 5:
            recent_states = list(self.fist_history)[-5:]
            if (not recent_states[-1] and not recent_states[-2] and
                any(recent_states[:-2]) and
                current_time - self.last_action_time > self.action_cooldown):

                self._do_action(self.gesture_actions.get("fist", "right_click"))
                self.last_action_time = current_time
                self._reset_dwell()
                return "fist_right_click"

        # 2. PINCH/DRAG SYSTEM using HAND CENTER
        if is_pinched:
            # Start timing pinch
            if self.pinch_start_time is None:
                self.pinch_start_time = current_time
                print("🤏 Pinch started - timing...")

            pinch_duration = current_time - self.pinch_start_time

            # Start drag after threshold
            if pinch_duration >= self.drag_threshold and not self.is_dragging:
                try:
                    # Get current screen position
                    current_screen_x, current_screen_y = pyautogui.position()

                    # Store HAND CENTER position at drag start
                    self.drag_start_hand_pos = hand_center.copy()
                    self.drag_start_screen_pos = [current_screen_x, current_screen_y]

                    # Start drag
                    pyautogui.mouseDown(button='left')
                    self._emit_click("drag")
                    self.is_dragging = True
                    self.last_action_time = current_time

                    print(f"🖱️ DRAG STARTED! Hand center at ({hand_center[0]:.4f}, {hand_center[1]:.4f})")
                    print(f"🖱️ Screen position: ({current_screen_x}, {current_screen_y})")

                    self._reset_dwell()
                    return "drag_started"

                except Exception as e:
                    print(f"❌ Drag start failed: {e}")
                    return "drag_failed"

            # Continue dragging - track HAND CENTER movement
            if self.is_dragging and self.drag_start_hand_pos is not None and self.drag_start_screen_pos is not None:
                if self.calibration is not None:
                    # Absolute mapping via calibration bounding box
                    new_screen_x, new_screen_y = self.map_to_screen(hand_center[0], hand_center[1])
                else:
                    # Fallback: delta-based drag movement
                    hand_delta_x = hand_center[0] - self.drag_start_hand_pos[0]
                    hand_delta_y = hand_center[1] - self.drag_start_hand_pos[1]
                    screen_delta_x = hand_delta_x * self.screen_width * 3.0
                    screen_delta_y = hand_delta_y * self.screen_height * 3.0
                    new_screen_x = self.drag_start_screen_pos[0] + screen_delta_x
                    new_screen_y = self.drag_start_screen_pos[1] + screen_delta_y
                    new_screen_x = max(self.screen_edge_margin, min(self.screen_width - self.screen_edge_margin, new_screen_x))
                    new_screen_y = max(self.screen_edge_margin, min(self.screen_height - self.screen_edge_margin, new_screen_y))

                try:
                    pyautogui.moveTo(new_screen_x, new_screen_y, duration=0)

                    actual_x, actual_y = pyautogui.position()
                    total_moved = abs(actual_x - self.drag_start_screen_pos[0]) + abs(actual_y - self.drag_start_screen_pos[1])

                    if total_moved > 5:
                        print(f"🖱️ Dragging → screen ({actual_x},{actual_y}) [moved {total_moved:.0f}px]")

                except Exception as e:
                    print(f"❌ Drag move failed: {e}")

                return "dragging"

            # Waiting for drag threshold
            remaining = self.drag_threshold - pinch_duration
            if remaining > 0:
                return f"pinch_waiting_{remaining:.1f}"

        else:
            # Pinch released
            if self.pinch_start_time is not None:
                pinch_duration = current_time - self.pinch_start_time

                # End drag if was dragging
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp(button='left')

                        # Calculate total drag distance
                        if self.drag_start_screen_pos is not None:
                            final_x, final_y = pyautogui.position()
                            total_distance = abs(final_x - self.drag_start_screen_pos[0]) + abs(final_y - self.drag_start_screen_pos[1])
                            print(f"🖱️ DRAG ENDED! Total distance: {total_distance} pixels")

                        self.is_dragging = False
                        self.drag_start_hand_pos = None
                        self.drag_start_screen_pos = None
                        self.pinch_start_time = None

                        self._reset_dwell()
                        return "drag_ended"

                    except Exception as e:
                        print(f"❌ Drag end failed: {e}")
                        self.is_dragging = False
                        return "drag_end_failed"

                # Quick click if short pinch
                elif (pinch_duration < self.drag_threshold and
                      current_time - self.last_action_time > self.action_cooldown):

                    try:
                        self._do_action(self.gesture_actions.get("pinch", "left_click"))
                        self.last_action_time = current_time
                        print("🖱️ CLICK!")
                        self.pinch_start_time = None
                        self._reset_dwell()
                        return "pinch_click"
                    except Exception as e:
                        print(f"❌ Click failed: {e}")
                        return "click_failed"

                # Clean up
                self.pinch_start_time = None

        # 3. TWO-FINGER SCROLL DETECTION
        if not is_pinched and not self.is_dragging:
            if self.detect_two_finger_scroll(landmarks):
                if self.scroll_reference_y is not None:
                    self._reset_dwell()
                    return "two_finger_scroll"

        # 4. NORMAL CURSOR CONTROL using HAND CENTER
        if not is_pinched and not self.is_dragging and self.detect_open_hand(landmarks):

            # Don't control cursor if in scroll mode
            if self.scroll_reference_y is not None:
                return "scroll_mode_active"

            if self.calibration is not None:
                # Absolute mapping via calibration bounding box
                new_x, new_y = self.map_to_screen(hand_center[0], hand_center[1])
                try:
                    pyautogui.moveTo(new_x, new_y, duration=0)
                except Exception as e:
                    print(f"❌ Cursor move failed: {e}")
            elif self.prev_hand_center is not None:
                # Fallback: relative delta-based movement (no calibration)
                hand_delta_x = hand_center[0] - self.prev_hand_center[0]
                hand_delta_y = hand_center[1] - self.prev_hand_center[1]

                # Convert pixel dead zone to normalized hand-coordinate threshold
                norm_dz = self.cursor_dead_zone / (max(self.screen_width, self.screen_height) * self.sensitivity)
                if abs(hand_delta_x) > norm_dz or abs(hand_delta_y) > norm_dz:
                    screen_delta_x = hand_delta_x * self.screen_width * self.sensitivity
                    screen_delta_y = hand_delta_y * self.screen_height * self.sensitivity

                    try:
                        current_x, current_y = pyautogui.position()
                        new_x = max(self.screen_edge_margin, min(self.screen_width - self.screen_edge_margin, current_x + screen_delta_x))
                        new_y = max(self.screen_edge_margin, min(self.screen_height - self.screen_edge_margin, current_y + screen_delta_y))

                        pyautogui.moveTo(new_x, new_y, duration=0)
                    except Exception as e:
                        print(f"❌ Cursor move failed: {e}")

            # Update previous HAND CENTER position
            self.prev_hand_center = hand_center.copy()

            # 5. DWELL-CLICK: if cursor stays still long enough, click
            if self.dwell_click_enabled:
                current_pos = pyautogui.position()
                if self.dwell_reference_pos is None:
                    self.dwell_reference_pos = current_pos
                    self.dwell_start_time = current_time
                    self.dwell_triggered = False
                else:
                    dist = math.sqrt(
                        (current_pos[0] - self.dwell_reference_pos[0]) ** 2 +
                        (current_pos[1] - self.dwell_reference_pos[1]) ** 2
                    )
                    # Re-arm hysteresis: once a dwell click has fired, the cursor
                    # must move CLEARLY away (2x radius) before another can fire,
                    # so an edge-of-radius tremor wobble can't repeat-click a target.
                    exit_radius = (self.dwell_click_radius * 2.0
                                   if self.dwell_triggered else self.dwell_click_radius)
                    if dist > exit_radius:
                        # Cursor moved outside radius — reset
                        self.dwell_reference_pos = current_pos
                        self.dwell_start_time = current_time
                        self.dwell_triggered = False
                    elif not self.dwell_triggered and self.dwell_start_time is not None:
                        elapsed = current_time - self.dwell_start_time
                        if elapsed >= self.dwell_click_duration:
                            self._do_action("left_click")
                            self.last_action_time = current_time
                            self.dwell_triggered = True
                            # Reset timer so moving cursor out and back allows another click
                            self.dwell_start_time = current_time
                            print(f"DWELL CLICK at ({current_pos[0]}, {current_pos[1]})")
                            return "dwell_click"

            return "cursor_control"

        else:
            # Reset hand center tracking when not controlling cursor (unless dragging)
            if not self.is_dragging:
                self.prev_hand_center = None
                self.smoothed_screen_pos = None
                self._smoothed_pass2 = None
                self._prev_raw_pos = None
                self._last_output_pos = None

        return "idle"

    def draw_debug_info(self, frame, landmarks, gesture, extended_count, finger_states):
        """Debug visualization with toggleable gaze detection status"""
        frame_height, frame_width = frame.shape[:2]

        # Calculate hand center
        hand_center = self.calculate_hand_center(landmarks)

        # Get finger positions for pinch visualization
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        is_pinched = self.pinch_threshold is not None and pinch_distance < self.pinch_threshold

        # Clean background - larger for gaze info
        cv2.rectangle(frame, (10, 10), (650, 300), (0, 0, 0), -1)

        # GAZE DETECTION STATUS
        if self.gaze_detection_enabled:
            gaze_color = (0, 255, 0) if self.looking_at_screen else (0, 0, 255)
            gaze_text = "ACTIVE" if self.looking_at_screen else "DISABLED"
            cv2.putText(frame, f"👁️ Gaze Control: {gaze_text}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)

            face_status = "DETECTED" if self.face_detected else "NOT FOUND"
            face_color = (255, 255, 255) if self.face_detected else (100, 100, 100)
            cv2.putText(frame, f"Face: {face_status}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        else:
            cv2.putText(frame, "👁️ Gaze Control: OFF (Always Active)", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'g' to enable gaze detection", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # Current gesture
        if self.is_safe_to_control():
            color = (0, 255, 0) if gesture != "idle" else (255, 255, 255)
            if "drag" in gesture:
                color = (255, 0, 128)  # Pink for drag
            elif "pinch_waiting" in gesture:
                color = (255, 128, 0)  # Orange for waiting
            elif "two_finger_scroll" in gesture or "scroll_mode" in gesture:
                color = (0, 255, 255)  # Cyan for scroll
            elif gesture == "dwell_click":
                color = (0, 200, 100)  # Green for dwell click
        else:
            color = (100, 100, 100)  # Gray when disabled
            gesture = "safety_disabled" if self.gaze_detection_enabled else "disabled"

        cv2.putText(frame, f"Gesture: {gesture}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # HAND CENTER focus (only show if active)
        if self.is_safe_to_control():
            cv2.putText(frame, f"Hand Center: ({hand_center[0]:.4f}, {hand_center[1]:.4f})", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Two-finger scroll status
        if self.scroll_reference_y is not None and self.is_safe_to_control():
            cv2.putText(frame, f"📱 SCROLL MODE - Move 2 fingers up/down", (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Ref Y: {self.scroll_reference_y:.4f}, Acc: {self.scroll_accumulated:.3f}", (20, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Drag state
        elif self.is_dragging and self.is_safe_to_control():
            cv2.putText(frame, "🖱️ ACTIVELY DRAGGING!", (20, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 128), 2)
            if self.drag_start_hand_pos is not None:
                start_x, start_y = self.drag_start_hand_pos
                cv2.putText(frame, f"Drag start: ({start_x:.4f}, {start_y:.4f})", (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif self.pinch_start_time is not None and self.is_safe_to_control():
            remaining = max(0, self.drag_threshold - (time.time() - self.pinch_start_time))
            if remaining > 0:
                cv2.putText(frame, f"Hold {remaining:.1f}s more for drag", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Show control status
        elif not self.is_safe_to_control():
            if self.gaze_detection_enabled:
                cv2.putText(frame, "👁️ Look at screen to activate controls", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
            else:
                cv2.putText(frame, "✋ Controls active (gaze detection OFF)", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Extended fingers info (always show for debugging)
        is_open_hand = self.detect_open_hand(landmarks)
        cv2.putText(frame, f"Extended fingers: {extended_count}/5", (20, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show finger states for two-finger detection (highlight thumb ignorance)
        finger_names = ["Thumb*", "Index", "Middle", "Ring", "Pinky"]
        finger_text = ""
        for i, (name, state) in enumerate(zip(finger_names, finger_states)):
            status = "✓" if state else "✗"
            if i == 0:  # Thumb
                finger_text += f"{name}: {status}(ignored) | "
            else:
                finger_text += f"{name}: {status} | "

        cv2.putText(frame, finger_text[:-3], (20, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        open_hand_color = (0, 255, 0) if is_open_hand else (255, 0, 0)
        cv2.putText(frame, f"Open hand: {'YES' if is_open_hand else 'NO'}", (20, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, open_hand_color, 2)

        # Dwell-click status
        dwell_color = (0, 200, 100) if self.dwell_click_enabled else (100, 100, 100)
        dwell_text = f"Dwell: {'ON' if self.dwell_click_enabled else 'OFF'} ({self.dwell_click_duration}s, {self.dwell_click_radius}px) | 'd' to toggle"
        cv2.putText(frame, dwell_text, (20, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, dwell_color, 1)

        # Instructions
        if self.gaze_detection_enabled:
            cv2.putText(frame, "GAZE MODE | 'g'=toggle gaze, 'c'=profile, 'd'=dwell, 'q'=quit", (20, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "FREE MODE | 'g'=toggle gaze, 'c'=profile, 'd'=dwell, 'q'=quit", (20, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Visual indicators
        hand_center_pos = (int(hand_center[0] * frame_width), int(hand_center[1] * frame_height))
        thumb_pos = (int(thumb_tip[0] * frame_width), int(thumb_tip[1] * frame_height))
        index_pos = (int(index_tip[0] * frame_width), int(index_tip[1] * frame_height))
        middle_pos = (int(middle_tip[0] * frame_width), int(middle_tip[1] * frame_height))

        # Only show active visuals when safe to control
        if self.is_safe_to_control():
            # Pinch line
            line_color = (255, 0, 128) if self.is_dragging else (0, 255, 0) if is_pinched else (255, 255, 255)
            line_thickness = 6 if self.is_dragging else 3 if is_pinched else 2
            cv2.line(frame, thumb_pos, index_pos, line_color, line_thickness)

            # Two-finger scroll indicators
            if self.scroll_reference_y is not None:
                # Highlight the two scroll fingers
                cv2.circle(frame, index_pos, 15, (0, 255, 255), 4)
                cv2.circle(frame, middle_pos, 15, (0, 255, 255), 4)
                cv2.line(frame, index_pos, middle_pos, (0, 255, 255), 4)
                cv2.putText(frame, "SCROLL", (index_pos[0] + 20, index_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Highlight HAND CENTER as the main tracking point
            center_color = (255, 0, 128) if self.is_dragging else (0, 255, 255) if self.scroll_reference_y else (0, 255, 255)
            center_size = 20 if self.is_dragging else 15
            cv2.circle(frame, hand_center_pos, center_size, center_color, 4)
            cv2.putText(frame, "HAND CENTER", (hand_center_pos[0] + 25, hand_center_pos[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_color, 2)

            # Show drag start position if dragging
            if self.is_dragging and self.drag_start_hand_pos is not None:
                start_screen_x = int(self.drag_start_hand_pos[0] * frame_width)
                start_screen_y = int(self.drag_start_hand_pos[1] * frame_height)
                cv2.circle(frame, (start_screen_x, start_screen_y), 12, (255, 255, 0), 3)
                cv2.line(frame, (start_screen_x, start_screen_y), hand_center_pos, (255, 255, 0), 3)
                cv2.putText(frame, "START", (start_screen_x + 15, start_screen_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

            # Dwell-click progress indicator (radial arc around hand center)
            if (self.dwell_click_enabled and self.dwell_start_time is not None
                    and not self.dwell_triggered and self.dwell_reference_pos is not None):
                elapsed = time.time() - self.dwell_start_time
                progress = min(1.0, elapsed / self.dwell_click_duration)
                if progress > 0.1:  # Only show after 10% to avoid flicker
                    angle = int(360 * progress)
                    dwell_color = (0, 255, 0) if progress < 0.9 else (0, 255, 255)
                    cv2.ellipse(frame, hand_center_pos, (30, 30), -90, 0, angle, dwell_color, 3)
                    cv2.putText(frame, f"DWELL {progress*100:.0f}%",
                               (hand_center_pos[0] + 35, hand_center_pos[1] + 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, dwell_color, 1)
        else:
            # Show dimmed hand when not safe to control
            cv2.circle(frame, hand_center_pos, 10, (100, 100, 100), 2)
            cv2.line(frame, thumb_pos, index_pos, (100, 100, 100), 1)

    def _tracking_tick(self):
        """Single frame of the tracking loop, driven by QTimer.
        Any exception inside is caught and logged. One bad frame from MediaPipe
        or OpenCV should never crash the whole app.
        """
        try:
            _bench = self._bench
            _t0 = time.perf_counter() if _bench is not None else 0.0
            _t1 = _t2 = 0.0
            ret, frame = self.cap.read()
            if not ret:
                self._cam_fail_count = getattr(self, '_cam_fail_count', 0) + 1
                if self._cam_fail_count >= 90:  # ~3 seconds at 30fps
                    cam_url = (
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
                        if sys.platform == "darwin"
                        else "ms-settings:privacy-webcam"
                        if sys.platform == "win32"
                        else None
                    )
                    self._fatal_exit(
                        S("crash_title"),
                        "AirPoint lost access to your camera.\n\n"
                        "Another app may have taken it (Zoom, Teams, FaceTime, etc.), "
                        "or the camera was unplugged. Close that app or reconnect the "
                        "camera, then relaunch AirPoint.",
                        settings_url=cam_url,
                        settings_label="Open Camera Settings",
                    )
                    return
                return
            self._cam_fail_count = 0

            frame = cv2.flip(frame, 1)
            if _bench is not None:
                _t1 = time.perf_counter()

            if self.paused:
                # Parked: release any held action, reset latches so nothing fires
                # on resume, and skip detection entirely (camera stays open).
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp(button='left')
                    except Exception:
                        pass
                    self.is_dragging = False
                self.pinch_start_time = None
                self._pinch_active = False
                self.drag_start_hand_pos = None
                self.drag_start_screen_pos = None
                self.prev_hand_center = None
                self.smoothed_screen_pos = None
                self._smoothed_pass2 = None
                self._prev_raw_pos = None
                self._last_output_pos = None
                self.scroll_reference_y = None
                self.scroll_accumulated = 0
                self.scroll_exit_counter = 0
                self.scroll_enter_counter = 0
                self._reset_dwell()
                self._last_gesture = "paused"
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = self.hands.process(rgb_frame)
            if _bench is not None:
                _t2 = time.perf_counter()
            self.detect_face_and_gaze(frame)

            gesture = "no_hand"

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    landmarks = self.get_landmarks(hand_landmarks)
                    # detect_gestures computes its own finger states; the extra
                    # count_extended_fingers call here was discarded every frame.
                    gesture = self.detect_gestures(landmarks)
            else:
                # Clean up when hand lost
                if self.is_dragging:
                    try:
                        pyautogui.mouseUp(button='left')
                    except Exception:
                        pass
                    self.is_dragging = False

                self.pinch_start_time = None
                self._pinch_active = False
                self.drag_start_hand_pos = None
                self.drag_start_screen_pos = None
                self.prev_hand_center = None
                self.smoothed_screen_pos = None
                self._smoothed_pass2 = None
                self._prev_raw_pos = None
                self._last_output_pos = None
                self.scroll_reference_y = None
                self.scroll_accumulated = 0
                self.scroll_exit_counter = 0
                self.scroll_enter_counter = 0
                self._reset_dwell()

            if _bench is not None:
                _bench.record(_t0, _t1, _t2, time.perf_counter(),
                              bool(hand_results.multi_hand_landmarks),
                              self._prev_raw_pos, self._last_output_pos)
                if _bench.done():
                    _p = _bench.save()
                    print(f"[benchmark] complete: {len(_bench.rows)} frames -> {_p}")
                    self._bench = None
                    if getattr(self, '_tracking_timer', None) is not None:
                        self._tracking_timer.stop()
                    _app = QApplication.instance()
                    if _app is not None:
                        _app.quit()

            self._last_gesture = gesture
            self._tick_error_count = 0
        except SystemExit:
            raise
        except Exception as e:
            # One bad frame must not kill the app. Log and keep going.
            _write_crash_log(type(e), e, e.__traceback__)
            self._tick_error_count = getattr(self, '_tick_error_count', 0) + 1
            # If errors keep coming for ~2 seconds straight, something is
            # really wrong; show a dialog and exit cleanly.
            if self._tick_error_count >= 60:
                self._fatal_exit(
                    S("crash_title"),
                    "AirPoint kept running into an error while tracking your hand.\n\n"
                    "Details have been saved to crash.log next to the app.\n"
                    "Please relaunch AirPoint, and if this keeps happening, email "
                    "kavinvenkatesanofficial@gmail.com with the crash.log file attached."
                )
                return

    def run(self):
        """Main control loop using PyQt5 status panel."""
        print("Starting AirPoint controller...")

        app = QApplication.instance() or QApplication(sys.argv)
        apply_app_theme(app)

        # --- Profile selection / calibration phase ---
        # If the user marked a profile as default (Profiles panel), auto-load it
        # and skip the picker. First-time users (no default set) still see the wizard.
        if self.profile_name is None:
            default = self.get_default_profile()
            if default and self.load_profile(default):
                if self.calibration is None:
                    # Default profile was never calibrated — don't trap the user
                    # in its calibration flow; fall back to the normal picker.
                    self.profile_name = None
                else:
                    print(f"Auto-loaded default profile '{default}'.")
        if self.profile_name is not None and self.calibration is not None:
            print(f"Using pre-loaded profile '{self.profile_name}'.")
        else:
            wizard = SetupWizard(self)
            wizard.show()
            app.exec_()
            if wizard.result == "quit":
                self.cap.release()
                raise SystemExit("Quit during setup")
            print(f"Profile '{self.profile_name}' active.")

        # --- Tracking phase with status panel ---
        self._last_gesture = "no_hand"
        panel = StatusPanel(self)

        # Click-feedback overlay: transparent + click-through, always shown (it's
        # invisible until a ripple fires; _emit_click gates on click_feedback_enabled).
        self.overlay = ClickFeedbackOverlay()
        self.overlay.show()

        # Tracking timer (~30fps)
        tracking_timer = QTimer()
        tracking_timer.setInterval(16)
        tracking_timer.timeout.connect(self._tracking_tick)
        self._tracking_timer = tracking_timer  # reachable from _fatal_exit in the tick

        if self._bench_seconds:
            out = self._bench_out or os.path.join(APP_DIR, "bench",
                                                  f"trace_{int(time.time())}.csv")
            self._bench = _BenchLog(out, self._bench_seconds)
            print(f"[benchmark] logging ~{self._bench_seconds:.0f}s of frames -> {out}")

        # Wire up panel buttons
        def on_recalibrate():
            tracking_timer.stop()
            panel.timer.stop()
            panel.hide()
            # Close any open Settings/Profiles tool windows so they can't cover
            # or race the calibration wizard.
            for _attr in ('settings_panel', 'profiles_panel'):
                _p = getattr(panel, _attr, None)
                if _p is not None:
                    _p.close()
            # Run recalibration WITHOUT nesting the application event loop.
            # A local QEventLoop returns here when the wizard closes (instead of
            # app.exec_(), whose quit would tear down the main loop), and
            # disabling auto-quit stops Qt from quitting the whole app when the
            # wizard closes while the panel is hidden — the cause of the
            # "restart setup quits the app" crash.
            prev_quit = app.quitOnLastWindowClosed()
            app.setQuitOnLastWindowClosed(False)
            wizard = SetupWizard(self)
            loop = QEventLoop()
            wizard.finished.connect(lambda _result: loop.quit())
            wizard.show()
            loop.exec_()
            app.setQuitOnLastWindowClosed(prev_quit)
            if wizard.result == "quit":
                on_quit()
                return
            panel.show()
            panel.start()
            tracking_timer.start()

        def on_quit():
            tracking_timer.stop()
            panel.timer.stop()
            panel.close()
            if self.overlay is not None:
                self.overlay.close()
            app.quit()

        panel.recal_btn.clicked.connect(on_recalibrate)
        panel.quit_btn.clicked.connect(on_quit)
        panel._on_close_quit = on_quit  # X-button / closeEvent also stops tracking

        panel.show()
        panel.start()
        tracking_timer.start()
        app.exec_()

        # Final cleanup
        if self.is_dragging:
            try:
                pyautogui.mouseUp(button='left')
            except Exception:
                pass

        self.cap.release()
        if self.overlay is not None:
            self.overlay.close()
        print("AirPoint stopped.")

if __name__ == "__main__":
    # In production (frozen exe), suppress all console output
    if FROZEN:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # Install global exception hook so crashes inside Qt event loops also get caught
    sys.excepthook = show_crash_dialog

    parser = argparse.ArgumentParser(description="AirPoint - Gesture-powered mouse controller")
    parser.add_argument("--profile", type=str, default=None,
                        help="Load a saved profile by name (skips profile selector)")
    parser.add_argument("--no-gaze", action="store_true",
                        help="Start with gaze detection disabled")
    parser.add_argument("--dwell", action="store_true",
                        help="Start with dwell-click enabled")
    parser.add_argument("--generate-default", action="store_true",
                        help="Write default.json to profiles/ directory and exit")
    parser.add_argument("--benchmark", type=float, default=0, metavar="SECONDS",
                        help="Log per-frame latency/FPS/jitter/detection for N seconds "
                             "to a CSV (bench/analyze.py reads it), then exit. Needs psutil "
                             "for CPU/RAM. See bench/METHODS.md.")
    parser.add_argument("--benchmark-out", type=str, default=None,
                        help="CSV path for --benchmark (default: bench/trace_<ts>.csv)")
    args = parser.parse_args()

    if args.generate_default:
        os.makedirs(PROFILES_DIR, exist_ok=True)
        path = os.path.join(PROFILES_DIR, "default.json")
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Default profile written to {path}")
        raise SystemExit(0)

    try:
        gaze = not args.no_gaze
        controller = HandCenterGestureController(enable_gaze_detection=gaze)

        if args.dwell:
            controller.dwell_click_enabled = True

        if args.benchmark:
            controller._bench_seconds = args.benchmark
            controller._bench_out = args.benchmark_out

        if args.profile:
            if not controller.load_profile(args.profile):
                print(f"Profile '{args.profile}' not found. Will start setup wizard.")
                controller.profile_name = args.profile
                # Wizard will run inside run() since calibration is None
            else:
                print(f"Loaded profile '{args.profile}' from CLI.")

        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except SystemExit:
        pass
    except Exception:
        show_crash_dialog(*sys.exc_info())
