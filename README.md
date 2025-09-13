# AirPoint
AirPoint enables users to turn their webcam into a gesture mouse. This Python tool tracks your hand for smooth, natural cursor control—with functionalities such as clicking, dragging, and scrolling.

This Python tool uses **MediaPipe + OpenCV + PyAutoGUI** to track the **center of your hand** for smooth cursor movement, natural drag, pinch-to-click, fist right-click, and a reliable two-finger scroll.

---

## ✨ Features
- **Hand-Center Cursor Control** – smoother, less jittery than fingertip tracking
- **🤏 Quick Pinch → Left Click**
- **🤏 Hold Pinch (0.4s) → Drag Mode**
- **✊ Fist → Right Click**
- **✌️ Two-Finger Scroll** (Index + Middle extended, Ring + Pinky folded, Thumb ignored)

---

## 🖥️ How It Works
- **Open Hand** → Move cursor (hand center deltas → screen deltas)  
- **Pinch** → Short = Click; Hold = Drag  
- **Fist** → Right Click  
- **Two-Finger Gesture** → Smooth scroll mode with dead zone + sticky grace frames  

Behind the scenes:
- Uses **MediaPipe Hands** for landmarks  
- Stabilizes cursor using **palm MCP joints + wrist**  
- Gesture logic powered by distance thresholds, cooldowns, and movement accumulators  
- Mouse/scroll actions simulated with **PyAutoGUI**

---

## 📦 Installation
Clone the repo and install requirements:

```bash
git clone https://github.com/yourusername/hand-center-gesture-controller.git
cd hand-center-gesture-controller
pip install -r requirements.txt
