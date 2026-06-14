# AirPoint — measurement toolkit & methods

This directory turns AirPoint's hand-wave claims into figures. Each measurement
below retires one assertion ("real-time", "modest machines", "jitter is the main
problem", "calibration helps", "lighting-sensitive").

**Important:** the numbers must be *measured on your hardware with a real hand* —
this toolkit produces them, it does not invent them. The only figures you can
quote without running anything are the **analytically derived** ones in §0 (they
follow from the code + first-order filter theory, and are exact for the shipped
defaults).

Dependencies: Python 3.10, the app's own deps, plus `pip install psutil` for the
CPU/RAM columns. The Fitts task uses only the standard-library `tkinter`.

---

## 0. Analytically derived figures (no webcam; exact for the shipped defaults)

These come straight from `map_to_screen` (double EMA, `smoothing_factor`
default 0.65, second pass α₂ = 0.8·α), the 16 ms `QTimer`, and the calibration
mapping. Cite them as *derived*, and use the measured numbers (§1) to confirm.

| Quantity | Value (default) | Basis |
|---|---|---|
| Software frame-rate cap | **62.5 Hz** (16 ms tick) | true FPS is MediaPipe-CPU-bound → measure |
| Smoothing-induced cursor lag | **≈ 2.9 frames ≈ 98–147 ms** (at 30→20 FPS) | EMA mean group delay Σ α/(1−α) over both passes |
| …at rest (velocity-adaptive α→0.85) | ≈ 7.8 frames (≈ 260–390 ms) | the adaptive term raises α when the hand is slow |
| …during fast motion (α→0.50) | ≈ 1.7 frames (≈ 56–83 ms) | adaptive term lowers α when the hand moves |
| Jitter attenuation (white noise) | **≈ 3.9× std** (8× at α=0.85) | per-pass factor √((1−α)/(1+α)), compounded |
| Dead-zone | residual motion < **10 px** fully suppressed | radial hold, default `cursor_dead_zone` |
| Reachable area, uncalibrated | **(x-range)·(y-range)·100 %** of screen | raw `h∈[0,1]` maps directly to screen |
| Reachable area, calibrated | **≈ 100 %** | range remapped to full screen + margin |
| Hand model | MediaPipe Hands, 21 landmarks, **1 hand, CPU** (GPU disabled), det-conf 0.7 | `main.py` Hands config |

The lag↔jitter trade-off above is exactly the motivation for a **1€ filter** as
future work: report the measured jitter (§1.3) as the baseline it must beat
*without* paying the ~100 ms (or worse, at rest) smoothing lag.

---

## 1. Tier 1 — measure yourself (no participants, no ethics)

Produce a trace, then analyse it:

```bash
python main.py --benchmark 60               # 60 s, logs bench/trace_<ts>.csv, then exits
python bench/analyze.py bench/trace_*.csv
```

The trace logs, per frame: per-stage latency, hand-detected flag, raw & smoothed
cursor position, and CPU/RAM.

### 1.1 End-to-end latency (ms) — retires "real-time"
- `analyze.py` reports **total** software latency (mean / median / **p95** / max)
  and the **capture / inference / post** breakdown. Inference (MediaPipe) dominates.
- This is the **software** pipeline (camera-read → cursor-move). For true
  **motion-to-photon** (physical hand move → pixel change), film hand + screen at
  240 FPS (any recent phone) and count frames between hand-start and cursor-start;
  report that alongside the software figure. Reference: < ~100 ms feels responsive.

### 1.2 Frame rate + CPU/RAM — retires "modest machines"
- `analyze.py` reports mean **FPS** and **CPU%/RAM** (needs `psutil`).
- **Name the hardware.** Run twice: on the deployment/centre machine, and on a
  deliberately weak laptop, to show the floor. Report both rows.

### 1.3 Stationary jitter (RMS px) — retires "jitter is the main problem"
**The single most important missing number.** It quantifies the headline
limitation and is the baseline the 1€-filter future work is measured against.

```bash
python main.py --benchmark 30 --profile <yours>   # HOLD HAND AS STILL AS POSSIBLE the whole 30 s
python bench/analyze.py bench/trace_*.csv --jitter
```
- Reports 2-D **RMS deviation** of the delivered (smoothed) cursor *and* the raw
  pre-smoothing position → the empirical **jitter-reduction factor** (validates
  §0's ~3.9×). RMS = √(mean((x−x̄)² + (y−ȳ)²)).
- Then repeat with a **deliberately shaky hand** for the tremor baseline. Two
  numbers (still / shaky) × (raw / smoothed) is a clean 2×2.

### 1.4 Calibration on vs off: reachable area (%) — retires "calibration helps"
The one number that directly validates the per-user-calibration contribution.
Runs on your existing profiles, **no capture needed**:
```bash
python bench/reachable_area.py
```
Prints, per calibrated profile, uncalibrated reach % (your comfortable hand
span) vs ~100 % calibrated, with the fold-improvement. (A small range of motion
typically reaches only 1–6 % of the screen uncalibrated.)

### 1.5 Detection rate vs lighting (%) — retires "lighting-sensitive"
Run a short trace in each condition and read the detection rate:
```bash
# repeat for each cell; use a phone lux-meter app for rough lux
python main.py --benchmark 20            # then note the "detection rate" line
python bench/analyze.py bench/trace_*.csv
```
Fill a 4-row table: {bright, dim} × {plain, cluttered background} (and/or 2–3 lux
levels). Report detection-success % per cell.

---

## 2. Tier 2 — ISO 9241-9 throughput (the benchmark reviewers respect)

Device characterisation (you + a few volunteers; no disabled-user deployment, so
no ethics gate). You control the cursor with AirPoint; the task records timing
and accuracy and computes effective throughput.

```bash
python bench/fitts.py run --device airpoint --participant P01
python bench/fitts.py run --device mouse    --participant P01   # baseline, same task
python bench/fitts.py analyse bench/results/fitts_*.csv
```
- Multidirectional tapping over Amplitude × Width conditions (default A ∈ {250,
  500} px, W ∈ {24, 48, 96} px → ties straight to real UI button sizes).
- Reports **effective throughput (bits/s)** via the effective-measures method
  (Soukoreff & MacKenzie 2004): Wₑ = 4.133·SD, IDₑ = log₂(Aₑ/Wₑ+1), TP = IDₑ/MT —
  directly comparable to mouse / head-mouse / eye-gaze numbers in the literature
  (mouse ≈ 3.7–4.9 bit/s; head/eye pointers ≈ 1.5–3 bit/s).
- Also prints **target-acquisition success rate by button size** (24/48/96 px),
  which shows exactly where small-target failure begins and ties back to §1.3.

---

## 3. Tier 3 — user outcomes (needs ethics approval + consent)

Out of scope for this toolkit, but the same `fitts.py` + a fixed task battery
(open browser → click link → scroll → type a word) gives within-subject
**task-completion rate / time-on-task**, AirPoint vs each participant's current
input method. Add **SUS** (usability) and **NASA-TLX** (workload — you flagged arm
fatigue); use staff/caregiver proxy ratings for this population and say so. Even
n = 3–6 as a per-participant case series beats zero decisively.

---

## Files
- `analyze.py` — trace CSV → latency / FPS / CPU-RAM / detection / jitter
- `reachable_area.py` — profile JSON → calibration reachable-area %
- `fitts.py` — ISO 9241-9 tapping task (`run`) + throughput analysis (`analyse`)
- traces land in `bench/trace_*.csv`; Fitts logs in `bench/results/`
