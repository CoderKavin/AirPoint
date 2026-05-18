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
git clone https://github.com/coderkavin/airpoint.git
cd airpoint
pip install -r requirements.txt
python main.py
```

Or download a prebuilt release from the [Releases page](https://github.com/coderkavin/airpoint/releases) — `AirPoint-Windows.zip` for Windows, `AirPoint.dmg` for macOS.

---

## 🪟 Windows: First-Run Troubleshooting

The Windows build is not yet code-signed, so Windows treats it with suspicion on first run. These are all expected and safe to work around:

**"Windows protected your PC" (SmartScreen)**
Click **More info → Run anyway**. This happens because the app isn't signed yet.

**"AirPoint needs the Microsoft Visual C++ Runtime"**
Let the app install it, or install it yourself (64-bit, latest):
<https://aka.ms/vs/17/release/vc_redist.x64.exe> — then **restart your PC**.

**"AirPoint can't load required libraries" — even after installing VC++**
This means Windows Defender **quarantined a file inside the `AirPoint\_internal` folder**. The `.exe` survives but the app is left incomplete. To fix it permanently on your machine:

1. **Windows Security → Virus & threat protection → Manage settings → Exclusions** → add the folder you keep AirPoint in (e.g. your `Downloads` folder).
2. **Protection history** (same screen) → **Restore** anything listed for AirPoint.
3. **Delete the broken AirPoint folder**, then **re-extract `AirPoint-Windows.zip` fresh** into the excluded folder.
4. Run `AirPoint.exe` again.

> Always extract the **whole** zip and keep `AirPoint.exe` next to its `_internal` folder — the app will not run if they're separated.

If startup still fails, AirPoint writes a `dll_error.log` next to the `.exe` — open an issue with its contents.
