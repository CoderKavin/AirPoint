import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import json
import os
import sys
import copy
import argparse
from collections import deque

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QLineEdit, QListWidget,
                              QStackedWidget, QProgressBar, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")

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
        self.stacked.addWidget(self._build_profile_page())    # 0
        self.stacked.addWidget(self._build_name_page())        # 1
        self.stacked.addWidget(self._build_welcome_page())     # 2
        self.stacked.addWidget(self._build_calibration_page()) # 3
        self.stacked.addWidget(self._build_done_page())        # 4

        # Show correct starting page
        if self.controller.profile_name and self.controller.calibration is None:
            # Profile name given via CLI but not found — go straight to welcome
            self.profile_name = self.controller.profile_name
            self.welcome_title.setText(f"Hi, {self.profile_name}!")
            self.stacked.setCurrentIndex(2)
        elif self.controller.list_profiles():
            self.stacked.setCurrentIndex(0)
        else:
            self.stacked.setCurrentIndex(1)

    # ---- Page Builders ----

    def _build_profile_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(60, 40, 60, 30)
        vbox.setSpacing(12)

        title = QLabel("Welcome Back")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")
        vbox.addWidget(title)

        sub = QLabel("Choose your profile to get started")
        sub.setFont(QFont("Segoe UI", 12))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #999;")
        vbox.addWidget(sub)

        vbox.addSpacing(10)

        self.profile_list = QListWidget()
        for name in self.controller.list_profiles():
            self.profile_list.addItem(name)
        new_item = "+ New Profile"
        self.profile_list.addItem(new_item)
        self.profile_list.setCurrentRow(0)
        self.profile_list.itemDoubleClicked.connect(self._on_profile_select)
        vbox.addWidget(self.profile_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        select_btn = QPushButton("Select")
        select_btn.clicked.connect(self._on_profile_select)
        btn_row.addWidget(select_btn)

        quit_btn = QPushButton("Quit")
        quit_btn.setObjectName("secondary")
        quit_btn.clicked.connect(lambda: self._finish("quit"))
        btn_row.addWidget(quit_btn)
        vbox.addLayout(btn_row)

        return page

    def _build_name_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(80, 60, 80, 40)
        vbox.setSpacing(16)

        vbox.addStretch(1)

        title = QLabel("What's your name?")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white;")
        vbox.addWidget(title)

        sub = QLabel("We'll create a personal profile for you")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #888;")
        vbox.addWidget(sub)

        vbox.addSpacing(10)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Kavin, Priya, User1...")
        self.name_input.setMaxLength(20)
        self.name_input.setAlignment(Qt.AlignCenter)
        self.name_input.setFixedHeight(48)
        self.name_input.returnPressed.connect(self._on_name_entered)
        vbox.addWidget(self.name_input)

        vbox.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        continue_btn = QPushButton("Continue")
        continue_btn.clicked.connect(self._on_name_entered)
        btn_row.addStretch()
        btn_row.addWidget(continue_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        vbox.addStretch(1)
        return page

    def _build_welcome_page(self):
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(60, 40, 60, 30)
        vbox.setSpacing(12)

        vbox.addStretch(1)

        self.welcome_title = QLabel("Hi there!")
        self.welcome_title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.welcome_title.setAlignment(Qt.AlignCenter)
        self.welcome_title.setStyleSheet("color: white;")
        vbox.addWidget(self.welcome_title)

        sub = QLabel("Let's set up AirPoint for you.")
        sub.setFont(QFont("Segoe UI", 13))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #aaa;")
        vbox.addWidget(sub)

        desc = QLabel("This quick setup takes about 30 seconds.\nWe'll calibrate the cursor to your hand movements.")
        desc.setFont(QFont("Segoe UI", 11))
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #777;")
        desc.setWordWrap(True)
        vbox.addWidget(desc)

        vbox.addSpacing(20)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        begin_btn = QPushButton("Begin Setup")
        begin_btn.clicked.connect(self._start_calibration)
        btn_row.addStretch()
        btn_row.addWidget(begin_btn)

        skip_btn = QPushButton("Skip")
        skip_btn.setObjectName("secondary")
        skip_btn.clicked.connect(lambda: self._finish("skipped"))
        btn_row.addWidget(skip_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        vbox.addStretch(1)
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
        self.cal_title = QLabel("Step 1 of 4: Movement Range")
        self.cal_title.setFont(QFont("Segoe UI", 13))
        self.cal_title.setAlignment(Qt.AlignCenter)
        self.cal_title.setStyleSheet("color: #999;")
        vbox.addWidget(self.cal_title)

        self.cal_instruction = QLabel("Reach as far LEFT as comfortable")
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
        self.cal_hint = QLabel("Press SPACE when your hand is at your LEFT limit")
        self.cal_hint.setFont(QFont("Segoe UI", 13))
        self.cal_hint.setAlignment(Qt.AlignCenter)
        self.cal_hint.setStyleSheet("color: #aaa;")
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
        vbox.setContentsMargins(60, 40, 60, 30)
        vbox.setSpacing(12)

        vbox.addStretch(1)

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

        vbox.addSpacing(10)

        title = QLabel("You're all set!")
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00e8a0;")
        vbox.addWidget(title)

        self.done_subtitle = QLabel("Your profile is ready to use.")
        self.done_subtitle.setFont(QFont("Segoe UI", 12))
        self.done_subtitle.setAlignment(Qt.AlignCenter)
        self.done_subtitle.setStyleSheet("color: #999;")
        vbox.addWidget(self.done_subtitle)

        vbox.addSpacing(20)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        start_btn = QPushButton("Start AirPoint")
        start_btn.clicked.connect(lambda: self._finish("completed"))
        btn_row.addStretch()
        btn_row.addWidget(start_btn)
        btn_row.addStretch()
        vbox.addLayout(btn_row)

        vbox.addStretch(1)
        return page

    # ---- Page Actions ----

    def _on_profile_select(self):
        item = self.profile_list.currentItem()
        if item is None:
            return
        text = item.text()
        if text == "+ New Profile":
            self.stacked.setCurrentIndex(1)
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
        self.welcome_title.setText(f"Hi, {name}!")
        self.stacked.setCurrentIndex(2)

    def _start_calibration(self):
        self.cal_step = 0
        self.dir_index = 0
        self.recorded = {}
        self.capture_countdown = None
        self.tremor_start = None
        self.tremor_samples = []
        self.gesture_results = {}
        self._update_cal_display()
        self.stacked.setCurrentIndex(3)
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
            self.cal_title.setText(f"Step 1 of 4: Movement Range")
            inst_map = {"LEFT": "Reach as far LEFT as comfortable",
                        "RIGHT": "Now reach to the RIGHT",
                        "UP": "Reach UP",
                        "DOWN": "And reach DOWN"}
            self.cal_instruction.setText(inst_map.get(d, ""))
            self.cal_hint.setText(f"Press SPACE when your hand is at your {d} limit")
            self.cal_progress.setVisible(False)
        elif self.cal_step == 1:
            self.cal_title.setText("Step 2 of 4: Steadiness Check")
            self.cal_instruction.setText("Hold your hand still and relax")
            self.cal_hint.setText("This is automatic — just hold still")
            self.cal_progress.setVisible(True)
            self.cal_progress.setValue(0)
        elif self.cal_step == 2:
            self.cal_title.setText("Step 3 of 4: Pinch Gesture")
            self.cal_instruction.setText("Pinch your thumb and index finger together")
            self.cal_hint.setText("Press SPACE to record  |  N to skip")
            self.cal_progress.setVisible(False)
            self.gesture_sampling = False
            self.gesture_samples = []
            self.gesture_skipped = False
        elif self.cal_step == 3:
            self.cal_title.setText("Step 4 of 4: Fist Gesture")
            self.cal_instruction.setText("Make a fist — close all your fingers")
            self.cal_hint.setText("Press SPACE to record  |  N to skip")
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
            self.cal_hint.setText("Hold still...")
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
                    self.cal_hint.setStyleSheet("color: #777;")
                    self._update_cal_display()
                else:
                    self.capture_countdown = None
                    self.cal_hint.setText("Hand lost — try again")
                    self.cal_hint.setStyleSheet("color: #ff6666;")

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
            self.cal_hint.setText("Show your hand to begin")
            self.cal_hint.setStyleSheet("color: #ff6666;")

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
            self.cal_hint.setText("Recording... hold the gesture")
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
        self.cal_hint.setStyleSheet("color: #777;")
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

        self.done_subtitle.setText(f"Profile '{self.profile_name}' is ready to use.")
        self.stacked.setCurrentIndex(4)

    # ---- Finish / Close ----

    def _finish(self, result):
        self.timer.stop()
        self.result = result
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
    """Clean status dashboard shown during tracking (replaces raw cv2 debug window)."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("AirPoint")
        self.setFixedSize(340, 420)
        self.setStyleSheet(DARK_STYLE + """
            StatusPanel { background-color: #1e1e23; }
        """)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(20, 20, 20, 16)
        vbox.setSpacing(10)

        # Header
        header = QLabel("AirPoint")
        header.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #00dcc8;")
        vbox.addWidget(header)

        # Profile name
        self.profile_label = QLabel()
        self.profile_label.setFont(QFont("Segoe UI", 12))
        self.profile_label.setAlignment(Qt.AlignCenter)
        self.profile_label.setStyleSheet("color: #999;")
        vbox.addWidget(self.profile_label)

        # Divider
        div = QLabel()
        div.setFixedHeight(1)
        div.setStyleSheet("background-color: #333;")
        vbox.addWidget(div)

        vbox.addSpacing(4)

        # Status rows
        self.gesture_label = self._make_row("Gesture", "Waiting...")
        vbox.addLayout(self.gesture_label["layout"])

        self.gaze_label = self._make_row("Gaze Lock", "OFF")
        vbox.addLayout(self.gaze_label["layout"])

        self.dwell_label = self._make_row("Dwell Click", "OFF")
        vbox.addLayout(self.dwell_label["layout"])

        self.smoothing_label = self._make_row("Smoothing", "0.65")
        vbox.addLayout(self.smoothing_label["layout"])

        self.hand_label = self._make_row("Hand", "Not detected")
        vbox.addLayout(self.hand_label["layout"])

        vbox.addStretch(1)

        # Divider
        div2 = QLabel()
        div2.setFixedHeight(1)
        div2.setStyleSheet("background-color: #333;")
        vbox.addWidget(div2)

        # Keyboard shortcuts hint
        hints = QLabel("C = recalibrate    G = gaze toggle\nD = dwell toggle    Q = quit")
        hints.setFont(QFont("Segoe UI", 9))
        hints.setAlignment(Qt.AlignCenter)
        hints.setStyleSheet("color: #555;")
        vbox.addWidget(hints)

        # Quit / Recalibrate buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.recal_btn = QPushButton("Recalibrate")
        self.recal_btn.setObjectName("secondary")
        self.recal_btn.setFixedHeight(34)
        btn_row.addWidget(self.recal_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setObjectName("secondary")
        self.quit_btn.setFixedHeight(34)
        self.quit_btn.setStyleSheet("""
            QPushButton { background-color: #442222; color: #ff6666; border: 1px solid #663333; border-radius: 8px; }
            QPushButton:hover { background-color: #553333; }
        """)
        btn_row.addWidget(self.quit_btn)
        vbox.addLayout(btn_row)

        # Timer for updating status
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._update_status)

    def _make_row(self, label_text, value_text):
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 11))
        label.setStyleSheet("color: #777;")
        row.addWidget(label)
        row.addStretch()
        value = QLabel(value_text)
        value.setFont(QFont("Segoe UI", 11, QFont.Bold))
        value.setStyleSheet("color: #ddd;")
        value.setAlignment(Qt.AlignRight)
        row.addWidget(value)
        return {"layout": row, "value": value}

    def start(self):
        self._update_status()
        self.timer.start()

    def _update_status(self):
        c = self.controller
        self.profile_label.setText(f"Profile: {c.profile_name or 'None'}")
        self.smoothing_label["value"].setText(f"{c.smoothing_factor:.2f}")

        # Gaze
        if c.gaze_detection_enabled:
            looking = getattr(c, 'looking_at_screen', False)
            self.gaze_label["value"].setText("ACTIVE" if looking else "WATCHING")
            self.gaze_label["value"].setStyleSheet(
                "color: #00e8a0; font-weight: bold;" if looking else "color: #ffaa00; font-weight: bold;")
        else:
            self.gaze_label["value"].setText("OFF")
            self.gaze_label["value"].setStyleSheet("color: #777; font-weight: bold;")

        # Dwell
        if c.dwell_click_enabled:
            self.dwell_label["value"].setText(f"ON ({c.dwell_click_duration:.1f}s)")
            self.dwell_label["value"].setStyleSheet("color: #00e8a0; font-weight: bold;")
        else:
            self.dwell_label["value"].setText("OFF")
            self.dwell_label["value"].setStyleSheet("color: #777; font-weight: bold;")

        # Hand / Gesture
        gesture = getattr(c, '_last_gesture', 'no_hand')
        if gesture == 'no_hand':
            self.hand_label["value"].setText("Not detected")
            self.hand_label["value"].setStyleSheet("color: #666; font-weight: bold;")
            self.gesture_label["value"].setText("--")
            self.gesture_label["value"].setStyleSheet("color: #666; font-weight: bold;")
        else:
            self.hand_label["value"].setText("Tracking")
            self.hand_label["value"].setStyleSheet("color: #00e8a0; font-weight: bold;")
            friendly = {
                "cursor_control": "Moving cursor",
                "left_click": "Click!",
                "drag_start": "Dragging...",
                "dragging": "Dragging...",
                "drag_end": "Drag ended",
                "right_click": "Right click!",
                "two_finger_scroll": "Scrolling",
                "scroll_active": "Scrolling",
                "pinch_wait": "Pinch hold...",
                "dwell_click": "Dwell click!",
                "safety_disabled": "Paused (look at screen)",
                "idle": "Ready",
            }.get(gesture, gesture)
            self.gesture_label["value"].setText(friendly)
            color = "#00dcc8"
            if "click" in gesture.lower():
                color = "#00e8a0"
            elif "drag" in gesture.lower():
                color = "#ff88cc"
            elif "scroll" in gesture.lower():
                color = "#88aaff"
            elif "safety" in gesture.lower():
                color = "#ff6666"
            self.gesture_label["value"].setStyleSheet(f"color: {color}; font-weight: bold;")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_G:
            self.controller.toggle_gaze_detection()
            self._update_status()
        elif key == Qt.Key_D:
            self.controller.toggle_dwell_click()
            self._update_status()
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
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
