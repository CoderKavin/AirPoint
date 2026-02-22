# ---------- Production mode: suppress warnings before any imports ----------
import sys
FROZEN = getattr(sys, 'frozen', False)
if FROZEN:
    import os as _os
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # silence TensorFlow
    _os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    _os.environ["GLOG_minloglevel"] = "3"                # silence glog (MediaPipe)
    import warnings
    warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
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

if FROZEN:
    logging.disable(logging.CRITICAL)  # silence all Python logging

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QLineEdit, QListWidget,
                              QStackedWidget, QProgressBar, QSizePolicy,
                              QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

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
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(S("crash_title"))
        msg.setText(S("crash_msg"))
        short = f"{exc_type.__name__}: {exc_value}"
        if len(short) > 200:
            short = short[:200] + "..."
        msg.setInformativeText(short)
        msg.setDetailedText("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        msg.setStyleSheet("""
            QMessageBox { background-color: #1e1e23; color: #ddd; }
            QLabel { color: #ddd; font-size: 13px; }
            QPushButton { background-color: #2a2a32; color: #ddd; border: 1px solid #444;
                          border-radius: 6px; padding: 6px 18px; min-width: 80px; }
            QPushButton:hover { background-color: #3a3a44; }
            QTextEdit { background-color: #111; color: #ccc; font-family: monospace; font-size: 11px; }
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
    # Stored for forward compatibility; dispatch not yet wired up.
    "gesture_actions": {
        "pinch": "left_click",
        "pinch_hold": "drag",
        "fist": "right_click",
        "two_finger_scroll": "scroll",
        "open_hand": "cursor_move",
    },
}

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
        "cal_pinch_inst": "Pinch your thumb and the finger next to it together",
        "cal_fist_inst": "Make a fist — close all your fingers",
        "cal_hint_gesture": "Press spacebar to record  or  press N to skip",
        "cal_hold_still": "Hold still...",
        "cal_hand_lost": "Hand lost — try again",
        "cal_show_hand": "Show your hand to begin",
        "cal_recording": "Recording... hold the gesture",
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
        "cal_pinch_inst": "अंगूठे और बगल वाली उँगली को जोड़ें",
        "cal_fist_inst": "मुट्ठी बंद करें — सारी उँगलियाँ बंद",
        "cal_hint_gesture": "स्पेसबार दबाएँ रिकॉर्ड के लिए  या  N दबाएँ छोड़ने के लिए",
        "cal_hold_still": "स्थिर रहें...",
        "cal_hand_lost": "हाथ नहीं दिखा — फिर कोशिश करें",
        "cal_show_hand": "अपना हाथ दिखाएँ",
        "cal_recording": "रिकॉर्ड हो रहा है... हाथ ऐसे ही रखें",
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

DARK_STYLE = """
QWidget { background-color: #1e1e23; color: #e0e0e5; }
QPushButton {
    background-color: #00dcc8; color: #1a1a1f; border-radius: 8px;
    padding: 10px 28px; font-weight: bold; font-size: 14px; border: none;
}
QPushButton:hover { background-color: #00f0d8; }
QPushButton:pressed { background-color: #00b8a6; }
QPushButton#secondary {
    background-color: #35353d; color: #aaa; border: 1px solid #555;
}
QPushButton#secondary:hover { background-color: #44444d; color: #ccc; }
QLineEdit {
    background-color: #2a2a30; border: 2px solid #00dcc8; border-radius: 8px;
    padding: 10px; font-size: 16px; color: white;
}
QLineEdit:focus { border-color: #00f0d8; }
QListWidget {
    background-color: #2a2a30; border-radius: 8px; border: 1px solid #3a3a42;
    font-size: 14px; outline: none;
}
QListWidget::item { padding: 12px 16px; border-radius: 6px; margin: 2px 4px; }
QListWidget::item:selected { background-color: rgba(0,220,200,0.15); color: #00dcc8; }
QListWidget::item:hover { background-color: #2e2e36; }
QProgressBar {
    border: none; border-radius: 9px; background-color: #2a2a30;
    text-align: center; color: #888; font-size: 11px; max-height: 18px;
}
QProgressBar::chunk { background-color: #00dcc8; border-radius: 9px; }
"""


class CameraWidget(QLabel):
    """QLabel subclass that displays OpenCV BGR frames."""

    def __init__(self, width=640, height=360, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111; border-radius: 10px;")

    def update_frame(self, cv_frame):
        rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = q_img.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))


class SetupWizard(QWidget):
    """PyQt5 setup wizard for profile selection and calibration."""

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
        self.setStyleSheet(DARK_STYLE)

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
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")
        vbox.addWidget(title)

        sub = QLabel("अपनी भाषा चुनें")
        sub.setFont(QFont("Segoe UI", 16))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #aaa;")
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
        self._extras_desc.setText(S("done_extras_gaze") + "<br>" + S("done_extras_dwell"))
        self.autostart_cb.setText(S("autostart_label"))
        self._start_btn.setText(S("start_airpoint"))

    def _build_profile_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(60, 40, 60, 30)
        vbox.setSpacing(12)

        self._prof_title = QLabel(S("profile_title"))
        self._prof_title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self._prof_title.setAlignment(Qt.AlignCenter)
        self._prof_title.setStyleSheet("color: white;")
        vbox.addWidget(self._prof_title)

        self._prof_sub = QLabel(S("profile_subtitle"))
        self._prof_sub.setFont(QFont("Segoe UI", 12))
        self._prof_sub.setAlignment(Qt.AlignCenter)
        self._prof_sub.setStyleSheet("color: #999;")
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
        self._name_title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        self._name_title.setAlignment(Qt.AlignCenter)
        self._name_title.setStyleSheet("color: white;")
        vbox.addWidget(self._name_title)

        self._name_sub = QLabel(S("name_subtitle"))
        self._name_sub.setFont(QFont("Segoe UI", 11))
        self._name_sub.setAlignment(Qt.AlignCenter)
        self._name_sub.setStyleSheet("color: #888;")
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
        self.welcome_title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.welcome_title.setAlignment(Qt.AlignCenter)
        self.welcome_title.setStyleSheet("color: white;")
        vbox.addWidget(self.welcome_title)

        self._welcome_sub = QLabel(S("welcome_sub"))
        self._welcome_sub.setFont(QFont("Segoe UI", 12))
        self._welcome_sub.setAlignment(Qt.AlignCenter)
        self._welcome_sub.setStyleSheet("color: #aaa;")
        self._welcome_sub.setWordWrap(True)
        vbox.addWidget(self._welcome_sub)

        vbox.addSpacing(10)

        self._how_title = QLabel(S("how_title"))
        self._how_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self._how_title.setStyleSheet("color: #00dcc8;")
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
            t.setFont(QFont("Segoe UI", 11))
            t.setStyleSheet("color: #ddd;")
            t.setFixedWidth(120)
            t.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            row.addWidget(t)
            d = QLabel(S(desc_key))
            d.setFont(QFont("Segoe UI", 10))
            d.setStyleSheet("color: #999;")
            d.setWordWrap(True)
            row.addWidget(d, 1)
            self._feat_labels.append((title_key, t, desc_key, d))
            vbox.addLayout(row)
            vbox.addSpacing(2)

        vbox.addSpacing(6)

        self._setup_note = QLabel(S("setup_note"))
        self._setup_note.setFont(QFont("Segoe UI", 11))
        self._setup_note.setAlignment(Qt.AlignCenter)
        self._setup_note.setStyleSheet("color: #aaa;")
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
            dot.setStyleSheet("background-color: #3a3a42; border-radius: 7px;")
            self.step_dots.append(dot)
            dots_row.addWidget(dot)
        dots_row.addStretch()
        vbox.addLayout(dots_row)

        # Title and instruction
        self.cal_title = QLabel(S("cal_step1_title"))
        self.cal_title.setFont(QFont("Segoe UI", 11))
        self.cal_title.setAlignment(Qt.AlignCenter)
        self.cal_title.setStyleSheet("color: #999;")
        self.cal_title.setWordWrap(True)
        vbox.addWidget(self.cal_title)

        self.cal_instruction = QLabel(S("cal_move_left"))
        self.cal_instruction.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.cal_instruction.setAlignment(Qt.AlignCenter)
        self.cal_instruction.setStyleSheet("color: #00dcc8;")
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
        self.cal_hint.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.cal_hint.setAlignment(Qt.AlignCenter)
        self.cal_hint.setStyleSheet("color: #00dcc8;")
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
            dot.setStyleSheet("background-color: #00dcc8; border-radius: 7px;")
            dots_row.addWidget(dot)
        dots_row.addStretch()
        vbox.addLayout(dots_row)

        vbox.addSpacing(6)

        self._done_title = QLabel(S("done_title"))
        self._done_title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self._done_title.setAlignment(Qt.AlignCenter)
        self._done_title.setStyleSheet("color: #00e8a0;")
        vbox.addWidget(self._done_title)

        self.done_subtitle = QLabel(S("done_subtitle"))
        self.done_subtitle.setFont(QFont("Segoe UI", 11))
        self.done_subtitle.setAlignment(Qt.AlignCenter)
        self.done_subtitle.setStyleSheet("color: #999;")
        vbox.addWidget(self.done_subtitle)

        vbox.addSpacing(8)

        # Quick reference card
        card = QWidget()
        card.setStyleSheet("background-color: #252530; border-radius: 10px; padding: 12px;")
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
            g.setFont(QFont("Segoe UI", 11))
            g.setStyleSheet("color: #bbb;")
            row.addWidget(g)
            row.addStretch()
            a = QLabel(S(a_key))
            a.setFont(QFont("Segoe UI", 11, QFont.Bold))
            a.setStyleSheet("color: #00dcc8;")
            a.setAlignment(Qt.AlignRight)
            row.addWidget(a)
            self._done_gesture_labels.append((g_key, g, a_key, a))
            card_vbox.addLayout(row)

        vbox.addWidget(card)

        vbox.addSpacing(6)

        self._extras_title = QLabel(S("done_extras_title"))
        self._extras_title.setFont(QFont("Segoe UI", 11))
        self._extras_title.setStyleSheet("color: #999;")
        vbox.addWidget(self._extras_title)

        self._extras_desc = QLabel(S("done_extras_gaze") + "<br>" + S("done_extras_dwell"))
        self._extras_desc.setFont(QFont("Segoe UI", 10))
        self._extras_desc.setWordWrap(True)
        self._extras_desc.setStyleSheet("color: #888;")
        vbox.addWidget(self._extras_desc)

        vbox.addSpacing(8)

        self.autostart_cb = QCheckBox(S("autostart_label"))
        self.autostart_cb.setFont(QFont("Segoe UI", 11))
        self.autostart_cb.setStyleSheet("""
            QCheckBox { color: #bbb; spacing: 8px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px;
                                   border: 2px solid #555; background: #2a2a32; }
            QCheckBox::indicator:checked { background: #00dcc8; border-color: #00dcc8; }
        """)
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
                dot.setStyleSheet("background-color: #00dcc8; border-radius: 7px;")
            elif i == current:
                dot.setStyleSheet("background-color: #00dcc8; border-radius: 7px; border: 2px solid white;")
            else:
                dot.setStyleSheet("background-color: #3a3a42; border-radius: 7px;")

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
            if _current_lang == "hi":
                dir_display = {"LEFT": "बाईं ओर", "RIGHT": "दाईं ओर",
                               "UP": "ऊपर", "DOWN": "नीचे"}
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
            self.cal_hint.setStyleSheet("color: #00dcc8; font-weight: bold;")

        if self.capture_countdown is not None:
            elapsed = time.time() - self.capture_countdown
            if elapsed >= HOLD_TIME:
                if hand_center is not None:
                    label = DIRECTIONS[self.dir_index]
                    self.recorded[label] = hand_center
                    print(f"  Captured {label}: ({hand_center[0]:.4f}, {hand_center[1]:.4f})")
                    self.dir_index += 1
                    self.capture_countdown = None
                    self.cal_hint.setStyleSheet("color: #00dcc8; font-weight: bold;")
                    self._update_cal_display()
                else:
                    self.capture_countdown = None
                    self.cal_hint.setText(S("cal_hand_lost"))
                    self.cal_hint.setStyleSheet("color: #ff6666; font-weight: bold;")

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
            self.cal_hint.setStyleSheet("color: #ff6666; font-weight: bold;")

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
            self.cal_hint.setStyleSheet("color: #00dcc8; font-weight: bold;")

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
                else:
                    self.gesture_results[gesture_name] = None
                    print(f"  {gesture_name}: too few samples, gesture disabled")
                self._advance_from_gesture(gesture_name)

    def _advance_from_gesture(self, gesture_name):
        self.cal_progress.setVisible(False)
        self.cal_hint.setStyleSheet("color: #00dcc8; font-weight: bold;")
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


class StatusPanel(QWidget):
    """User-friendly control panel shown during tracking."""

    TOGGLE_ON = """
        QPushButton { background-color: #1a3d38; color: #00e8a0; border: 2px solid #00dcc8;
                      border-radius: 12px; padding: 12px; text-align: left; font-size: 14px; }
        QPushButton:hover { background-color: #1f4a43; }
    """
    TOGGLE_OFF = """
        QPushButton { background-color: #2a2a32; color: #888; border: 2px solid #3a3a44;
                      border-radius: 12px; padding: 12px; text-align: left; font-size: 14px; }
        QPushButton:hover { background-color: #333340; }
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("AirPoint")
        self.setFixedSize(360, 520)
        self.setStyleSheet(DARK_STYLE + " StatusPanel { background-color: #1e1e23; }")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(24, 20, 24, 18)
        vbox.setSpacing(0)

        # ---- Header + status badge ----
        header = QLabel("AirPoint")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #00dcc8;")
        vbox.addWidget(header)

        self.profile_label = QLabel()
        self.profile_label.setFont(QFont("Segoe UI", 11))
        self.profile_label.setAlignment(Qt.AlignCenter)
        self.profile_label.setStyleSheet("color: #888;")
        vbox.addWidget(self.profile_label)

        vbox.addSpacing(10)

        # Big status indicator
        self.status_badge = QLabel(S("panel_looking"))
        self.status_badge.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setFixedHeight(50)
        self.status_badge.setStyleSheet("""
            background-color: #2a2a32; color: #888; border-radius: 12px; padding: 8px;
        """)
        vbox.addWidget(self.status_badge)

        vbox.addSpacing(14)

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

        # ---- Bottom actions ----
        self.recal_btn = QPushButton(S("panel_redo"))
        self.recal_btn.setCursor(Qt.PointingHandCursor)
        self.recal_btn.setFixedHeight(44)
        self.recal_btn.setStyleSheet("""
            QPushButton { background-color: #2a2a32; color: #ccc; border: 2px solid #3a3a44;
                          border-radius: 12px; font-size: 15px; }
            QPushButton:hover { background-color: #333340; }
        """)
        vbox.addWidget(self.recal_btn)

        vbox.addSpacing(8)

        self.quit_btn = QPushButton(S("panel_stop"))
        self.quit_btn.setCursor(Qt.PointingHandCursor)
        self.quit_btn.setFixedHeight(44)
        self.quit_btn.setStyleSheet("""
            QPushButton { background-color: #3a2020; color: #ff7777; border: 2px solid #552a2a;
                          border-radius: 12px; font-size: 15px; }
            QPushButton:hover { background-color: #4a2a2a; }
        """)
        vbox.addWidget(self.quit_btn)

        # Timer for updating status
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._update_status)

    def _toggle_gaze(self):
        self.controller.toggle_gaze_detection()
        self._update_status()

    def _toggle_dwell(self):
        self.controller.toggle_dwell_click()
        self._update_status()

    def start(self):
        self._update_status()
        self.timer.start()

    def _update_status(self):
        c = self.controller
        self.profile_label.setText(S("panel_hi", name=c.profile_name or "User"))

        # -- Status badge --
        gesture = getattr(c, '_last_gesture', 'no_hand')
        if gesture == 'no_hand':
            self.status_badge.setText(S("panel_looking"))
            self.status_badge.setStyleSheet(
                "background-color: #2a2a32; color: #888; border-radius: 12px; padding: 8px;")
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
            bg = "#1a3d38"
            fg = "#00e8a0"
            if "click" in gesture.lower():
                bg, fg = "#1a3d2a", "#00e8a0"
            elif "drag" in gesture.lower():
                bg, fg = "#3a1a30", "#ff88cc"
            elif "scroll" in gesture.lower():
                bg, fg = "#1a2a3d", "#88aaff"
            elif "safety" in gesture.lower():
                bg, fg = "#3d2a1a", "#ffaa44"
            self.status_badge.setStyleSheet(
                f"background-color: {bg}; color: {fg}; border-radius: 12px; padding: 8px;")

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
        event.accept()


class HandCenterGestureController:
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Control settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Gesture state (not configurable — runtime state)
        self.pinch_start_time = None
        self.is_dragging = False
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        self.prev_hand_center = None
        self.last_action_time = 0
        self.fist_history = deque(maxlen=8)

        # Two-finger scroll state
        self.scroll_reference_y = None
        self.scroll_accumulated = 0
        self.scroll_exit_counter = 0

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

    def _apply_config(self, config):
        """Apply a config dict to instance attributes, filling missing keys from DEFAULT_CONFIG."""
        merged = copy.deepcopy(DEFAULT_CONFIG)
        for key in config:
            if config[key] is not None and isinstance(config[key], dict) and isinstance(merged.get(key), dict):
                merged[key].update(config[key])
            else:
                merged[key] = config[key]

        self.sensitivity = merged["sensitivity"]
        self.smoothing_factor = merged["smoothing_factor"]

        th = merged["thresholds"]
        self.drag_threshold = th["drag_threshold"]
        self.action_cooldown = th["action_cooldown"]
        self.pinch_threshold = th["pinch_threshold"]
        self.fist_threshold = th["fist_threshold"]
        self.scroll_dead_zone = th["scroll_dead_zone"]
        self.scroll_threshold_val = th["scroll_threshold"]
        self.scroll_amount = th["scroll_amount"]
        self.screen_edge_margin = th["screen_edge_margin"]
        self.cursor_dead_zone = th["cursor_dead_zone"]
        self.calibration_margin = th["calibration_margin"]

        dw = merged["dwell_click"]
        self.dwell_click_enabled = dw["enabled"]
        self.dwell_click_radius = dw["radius"]
        self.dwell_click_duration = dw["duration"]

        self.gaze_detection_enabled = merged["gaze_detection_enabled"]
        self.gesture_actions = merged["gesture_actions"]
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
        with open(path, "r") as f:
            raw = json.load(f)

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

        # Print summary
        cal = self.calibration
        if cal:
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
            "gesture_actions": self.gesture_actions,
            "language": _current_lang,
        }
        path = os.path.join(PROFILES_DIR, f"{self.profile_name}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Profile '{self.profile_name}' saved to {path}")

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

        # Dead-zone filter: ignore micro-movements below threshold
        if self._last_output_pos is not None:
            dx = abs(self._smoothed_pass2[0] - self._last_output_pos[0])
            dy = abs(self._smoothed_pass2[1] - self._last_output_pos[1])
            if dx < self.cursor_dead_zone and dy < self.cursor_dead_zone:
                return self._last_output_pos[0], self._last_output_pos[1]

        # Clamp to screen bounds with margin
        m = self.screen_edge_margin
        screen_x = max(m, min(self.screen_width - m, self._smoothed_pass2[0]))
        screen_y = max(m, min(self.screen_height - m, self._smoothed_pass2[1]))
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

        # Initialize reference position on first detection
        if self.scroll_reference_y is None:
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
                except:
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

        # Calculate pinch
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        is_pinched = self.pinch_threshold is not None and pinch_distance < self.pinch_threshold

        extended_count, finger_states = self.count_extended_fingers(landmarks)

        # 1. FIST DETECTION for RIGHT CLICK
        is_fist = self.detect_fist(landmarks)
        self.fist_history.append(is_fist)

        if len(self.fist_history) >= 5:
            recent_states = list(self.fist_history)[-5:]
            if (not recent_states[-1] and not recent_states[-2] and
                any(recent_states[:-2]) and
                current_time - self.last_action_time > self.action_cooldown):

                pyautogui.rightClick()
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
                        pyautogui.click()
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
                    if dist > self.dwell_click_radius:
                        # Cursor moved outside radius — reset
                        self.dwell_reference_pos = current_pos
                        self.dwell_start_time = current_time
                        self.dwell_triggered = False
                    elif not self.dwell_triggered and self.dwell_start_time is not None:
                        elapsed = current_time - self.dwell_start_time
                        if elapsed >= self.dwell_click_duration:
                            pyautogui.click()
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
        """Single frame of the tracking loop, driven by QTimer."""
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb_frame)
        self.detect_face_and_gaze(frame)

        gesture = "no_hand"

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = self.get_landmarks(hand_landmarks)
                extended_count, finger_states = self.count_extended_fingers(landmarks)
                gesture = self.detect_gestures(landmarks)
        else:
            # Clean up when hand lost
            if self.is_dragging:
                try:
                    pyautogui.mouseUp(button='left')
                except:
                    pass
                self.is_dragging = False

            self.pinch_start_time = None
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
            self._reset_dwell()

        self._last_gesture = gesture

    def run(self):
        """Main control loop using PyQt5 status panel."""
        print("Starting AirPoint controller...")

        app = QApplication.instance() or QApplication(sys.argv)

        # --- Profile selection / calibration phase ---
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

        # Tracking timer (~30fps)
        tracking_timer = QTimer()
        tracking_timer.setInterval(16)
        tracking_timer.timeout.connect(self._tracking_tick)

        # Wire up panel buttons
        def on_recalibrate():
            tracking_timer.stop()
            panel.timer.stop()
            panel.hide()
            wizard = SetupWizard(self)
            wizard.show()
            app.exec_()
            if wizard.result == "quit":
                app.quit()
                return
            panel.show()
            panel.start()
            tracking_timer.start()

        def on_quit():
            tracking_timer.stop()
            panel.timer.stop()
            panel.close()
            app.quit()

        panel.recal_btn.clicked.connect(on_recalibrate)
        panel.quit_btn.clicked.connect(on_quit)

        panel.show()
        panel.start()
        tracking_timer.start()
        app.exec_()

        # Final cleanup
        if self.is_dragging:
            try:
                pyautogui.mouseUp(button='left')
            except:
                pass

        self.cap.release()
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
