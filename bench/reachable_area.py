#!/usr/bin/env python3
"""
Reachable-screen-area metric for AirPoint calibration (Tier-1, no webcam needed).

Validates the per-user calibration contribution: with a limited range of motion,
what fraction of the screen can the cursor reach WITHOUT calibration vs WITH it?

How it works (this is exact, from main.py:map_to_screen):
  - Uncalibrated path maps the raw normalized hand coordinate h in [0,1] straight
    to the screen: screen = h * screen_size. So the cursor can only reach the
    fraction of the screen spanned by the user's *comfortable* hand range.
        reachable_uncalibrated = (right-left) * (bottom-top) * 100   [%]
    where (left,right,top,bottom) are the user's recorded comfortable bounds in
    normalized [0,1] camera coordinates (exactly what calibration captures).
  - Calibrated path remaps that same [left,right]x[top,bottom] box onto the full
    screen (plus a small reach margin), so reachable_calibrated ~= 100%.

Usage:
    python bench/reachable_area.py                 # all profiles in ../profiles
    python bench/reachable_area.py profiles/me.json
"""
import json
import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(HERE, "..", "profiles")


def reachable(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cal = cfg.get("calibration")
    if not cal or not all(k in cal for k in ("left", "right", "top", "bottom")):
        return None
    rx = abs(cal["right"] - cal["left"])
    ry = abs(cal["bottom"] - cal["top"])
    margin = cal.get("calibration_margin", 0.05)
    name = os.path.splitext(os.path.basename(profile_path))[0]
    uncal = rx * ry * 100.0
    # Calibrated maps [left,right]x[top,bottom] -> [0,1] with +/- margin padding,
    # then clamps to [0,1], so the full screen is reachable.
    return {
        "profile": name,
        "x_range": rx, "y_range": ry, "margin": margin,
        "reachable_uncalibrated_pct": uncal,
        "reachable_calibrated_pct": 100.0,
        "improvement_x": (100.0 / uncal) if uncal > 0 else float("inf"),
    }


def main(argv):
    paths = argv[1:] or sorted(glob.glob(os.path.join(PROFILES_DIR, "*.json")))
    paths = [p for p in paths if os.path.basename(p) != "default_profile.txt"]
    if not paths:
        print("No profile JSONs found. Pass a path, or run after calibrating a profile.")
        return 1
    rows = []
    for p in paths:
        try:
            r = reachable(p)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {p}: {e}")
            continue
        if r:
            rows.append(r)
    if not rows:
        print("No calibrated profiles found (profiles exist but have no calibration bounds).")
        return 1
    print(f"{'profile':<16} {'x_range':>8} {'y_range':>8} "
          f"{'uncalib %':>10} {'calib %':>8} {'x better':>9}")
    print("-" * 64)
    for r in rows:
        print(f"{r['profile']:<16} {r['x_range']:>8.3f} {r['y_range']:>8.3f} "
              f"{r['reachable_uncalibrated_pct']:>9.1f}% {r['reachable_calibrated_pct']:>7.0f}% "
              f"{r['improvement_x']:>8.1f}x")
    mean_uncal = sum(r["reachable_uncalibrated_pct"] for r in rows) / len(rows)
    print("-" * 64)
    print(f"n={len(rows)} profiles | mean uncalibrated reach = {mean_uncal:.1f}% of screen "
          f"| calibrated = 100%")
    print("\nPaper sentence template:")
    print(f'  "Across n={len(rows)} calibrated profiles, an uncalibrated mapping reached only '
          f'{mean_uncal:.1f}% (range '
          f'{min(r["reachable_uncalibrated_pct"] for r in rows):.1f}-'
          f'{max(r["reachable_uncalibrated_pct"] for r in rows):.1f}%) of screen area, '
          f'versus full-screen coverage after per-user calibration."')
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
