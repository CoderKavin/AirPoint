# AirPoint

**AirPoint** turns your webcam into a **gesture-powered mouse**.  
Track your **hand center** (not just fingertips) for smooth cursor control—with **click**, **drag**, **right-click**, **scroll**, and even an optional **gaze-aware safety lock** you can toggle anytime.

Built with **MediaPipe**, **OpenCV**, and **PyAutoGUI**.

---

## ✨ Features

- **🖱️ Hand-Center Cursor Control** – smoother, less jittery than fingertip tracking
- **🤏 Pinch → Left Click** (quick pinch = click)
- **🤏 Hold Pinch (0.4s) → Drag Mode**
- **✊ Fist → Right Click**
- **✌️ Two-Finger Scroll**  
  - Index + Middle extended  
  - Ring + Pinky folded  
  - **Thumb ignored** for reliability  
  - Includes dead zone + movement accumulator + sticky frames for smooth scrolling
- **👁️ Gaze Detection (toggle with `g`)**  
  - When enabled → gestures **only work while you’re looking at the screen**  
  - When disabled → gestures are **always active**  
  - Toggle anytime with the **`g` key**  
- **🎛️ On-Screen Debug Overlay** – gesture label, gaze status, hand center coords, finger states, scroll mode, and drag info

---

## 🖥️ How It Works

- **Open Hand** → Cursor follows **hand center deltas → screen deltas**  
- **Pinch** → Short pinch = **click**; Hold (≥0.4s) = **drag mode**  
- **Fist** → **Right click**  
- **Two-Finger Scroll** → Vertical finger movement scrolls up/down  
- **Gaze Detection** → Uses **MediaPipe Face Mesh** to check if you’re facing the screen before allowing input (optional, toggle with `g`)

---

## ⌨️ Keybindings

| Key | Action |
|-----|--------|
| **g** | Toggle **gaze detection** |
| **q** | Quit program |

---

## 📦 Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/yourusername/airpoint.git
cd airpoint
pip install -r requirements.txt
