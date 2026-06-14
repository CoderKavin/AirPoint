#!/usr/bin/env python3
"""
ISO 9241-9 multidirectional tapping task + effective-throughput analysis for
AirPoint (Tier-2). Standard pointing-device benchmark; gives a bits/s figure
directly comparable to mouse / head-mouse / eye-gaze numbers in the literature.

You CONTROL THE CURSOR WITH AIRPOINT (or any device) — this app just shows the
targets, records each selection, and computes throughput. So the same task runs
for AirPoint, a normal mouse (baseline), or a user's existing assistive device.

Run the task (records to a CSV):
    python bench/fitts.py run  --device airpoint --participant P01
Then analyse one or more CSVs:
    python bench/fitts.py analyse results/fitts_*.csv

Conditions: every (Amplitude A x Width W) in AMPLITUDES x WIDTHS, N targets each,
clicked in the standard alternating-across-the-circle order. Throughput uses the
effective-measures method (Soukoreff & MacKenzie, 2004): We = 4.133*SD of the
along-axis selection scatter, IDe = log2(Ae/We + 1), TP = IDe / mean(MT).

Tkinter only (no extra deps). Press Esc to abort.
"""
import sys, os, csv, math, time, glob

AMPLITUDES = [250, 500]          # px (ring diameter / movement distance)
WIDTHS = [24, 48, 96]            # px target diameter -> ties to real UI button sizes
N_TARGETS = 11                   # odd -> clean alternating order
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------- the task -----------------------------

def run(device="device", participant="P00"):
    import tkinter as tk
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"fitts_{device}_{participant}_{int(time.time())}.csv")

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="#101014")
    cv = tk.Canvas(root, bg="#101014", highlightthickness=0)
    cv.pack(fill="both", expand=True)
    root.update()
    W, H = cv.winfo_width(), cv.winfo_height()
    cx, cy = W / 2, H / 2

    conditions = [(A, w) for A in AMPLITUDES for w in WIDTHS]
    state = {"ci": 0, "order_idx": 0, "t_prev": None, "prev_xy": (cx, cy), "rows": []}

    def target_positions(A):
        r = A / 2.0
        return [(cx + r * math.cos(2 * math.pi * i / N_TARGETS - math.pi / 2),
                 cy + r * math.sin(2 * math.pi * i / N_TARGETS - math.pi / 2))
                for i in range(N_TARGETS)]

    # ISO alternating order: 0, k, 2k, ... with k=(N//2) so we cross the ring.
    order = [(i * (N_TARGETS // 2)) % N_TARGETS for i in range(N_TARGETS)]

    def draw():
        cv.delete("all")
        if state["ci"] >= len(conditions):
            cv.create_text(cx, cy, text="Done — thank you!", fill="#00dcc8",
                           font=("Segoe UI", 28, "bold"))
            root.update()
            root.after(900, root.destroy)
            return
        A, w = conditions[state["ci"]]
        pts = target_positions(A)
        for (px, py) in pts:
            cv.create_oval(px - w / 2, py - w / 2, px + w / 2, py + w / 2,
                           outline="#333", width=1)
        cur = pts[order[state["order_idx"]]]
        r = w / 2
        cv.create_oval(cur[0] - r, cur[1] - r, cur[0] + r, cur[1] + r,
                       fill="#00dcc8", outline="")
        cv.create_text(cx, 40, fill="#888", font=("Segoe UI", 14),
                       text=f"Condition {state['ci']+1}/{len(conditions)}  "
                            f"A={A}px W={w}px   target {state['order_idx']+1}/{N_TARGETS}   "
                            f"(click the cyan circle; Esc to quit)")
        state["_target"] = cur
        state["_A"], state["_W"] = A, w
        if state["t_prev"] is None:
            state["t_prev"] = time.perf_counter()
        root.update()

    def on_click(ev):
        if state["ci"] >= len(conditions):
            return
        now = time.perf_counter()
        tx, ty = state["_target"]
        A, w = state["_A"], state["_W"]
        mt = now - state["t_prev"]
        hit = math.hypot(ev.x - tx, ev.y - ty) <= w / 2.0
        px, py = state["prev_xy"]
        state["rows"].append({
            "condition": state["ci"], "A": A, "W": w,
            "target_x": tx, "target_y": ty, "prev_x": px, "prev_y": py,
            "click_x": ev.x, "click_y": ev.y, "mt_s": mt, "hit": int(hit),
        })
        state["prev_xy"] = (ev.x, ev.y)
        state["t_prev"] = now
        state["order_idx"] += 1
        if state["order_idx"] >= N_TARGETS:
            state["order_idx"] = 0
            state["ci"] += 1
            state["t_prev"] = None
            state["prev_xy"] = (cx, cy)
        draw()

    cv.bind("<Button-1>", on_click)
    root.bind("<Escape>", lambda e: root.destroy())
    draw()
    root.mainloop()

    if state["rows"]:
        with open(out, "w", newline="", encoding="utf-8") as f:
            wri = csv.DictWriter(f, fieldnames=list(state["rows"][0].keys()))
            wri.writeheader()
            wri.writerows(state["rows"])
        print(f"Saved {len(state['rows'])} selections -> {out}")
        analyse([out])
    else:
        print("No selections recorded.")


# --------------------------- the analysis ---------------------------

def _throughput_for_block(rows):
    """Effective throughput for one (A,W) block (Soukoreff & MacKenzie 2004).
    Drops the first selection (no defined prior position) per ISO practice."""
    rows = rows[1:]
    if len(rows) < 3:
        return None
    dists, proj_err, mts = [], [], []
    for r in rows:
        px, py = r["prev_x"], r["prev_y"]
        tx, ty = r["target_x"], r["target_y"]
        cxp, cyp = r["click_x"], r["click_y"]
        ax, ay = tx - px, ty - py
        amp = math.hypot(ax, ay)
        if amp == 0:
            continue
        ux, uy = ax / amp, ay / amp
        # along-axis position of the click relative to the START point:
        along = (cxp - px) * ux + (cyp - py) * uy
        dists.append(math.hypot(cxp - px, cyp - py))   # actual movement distance
        proj_err.append(along - amp)                    # over/under-shoot along axis
        mts.append(r["mt_s"])
    if len(proj_err) < 3:
        return None
    n = len(proj_err)
    mean_pe = sum(proj_err) / n
    sd = math.sqrt(sum((e - mean_pe) ** 2 for e in proj_err) / (n - 1))
    if sd == 0:
        return None
    Ae = sum(dists) / len(dists)
    We = 4.133 * sd
    IDe = math.log2(Ae / We + 1)
    mt = sum(mts) / len(mts)
    return {"Ae": Ae, "We": We, "IDe": IDe, "MT": mt, "TP": IDe / mt, "n": n}


def analyse(paths):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    if not files:
        print("No CSV files matched.")
        return
    all_rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in ("A", "W", "condition", "hit"):
                    r[k] = float(r[k])
                for k in ("target_x", "target_y", "prev_x", "prev_y",
                          "click_x", "click_y", "mt_s"):
                    r[k] = float(r[k])
                all_rows.append(r)

    # group by (A,W)
    blocks = {}
    for r in all_rows:
        blocks.setdefault((r["A"], r["W"]), []).append(r)

    print(f"\nISO 9241-9 results  ({len(files)} file(s), {len(all_rows)} selections)\n")
    print(f"{'A(px)':>6} {'W(px)':>6} {'IDe(bits)':>9} {'MT(s)':>7} "
          f"{'TP(bit/s)':>10} {'err%':>6} {'n':>4}")
    print("-" * 56)
    tps, id_mt = [], []
    for (A, W), rows in sorted(blocks.items()):
        rows = sorted(rows, key=lambda r: 0)  # keep file order
        res = _throughput_for_block(rows)
        err = 100.0 * (1 - sum(r["hit"] for r in rows) / len(rows))
        if res:
            print(f"{A:>6.0f} {W:>6.0f} {res['IDe']:>9.2f} {res['MT']:>7.3f} "
                  f"{res['TP']:>10.2f} {err:>5.1f}% {res['n']:>4d}")
            tps.append(res["TP"])
            id_mt.append((res["IDe"], res["MT"]))
        else:
            print(f"{A:>6.0f} {W:>6.0f}   (too few clean trials)")

    # success rate by target size (ties to UI button sizes)
    print("\nTarget-acquisition success by button size:")
    by_w = {}
    for r in all_rows:
        by_w.setdefault(r["W"], []).append(r["hit"])
    for W in sorted(by_w):
        hits = by_w[W]
        print(f"  W={W:>3.0f}px : {100*sum(hits)/len(hits):5.1f}% success  (n={len(hits)})")

    if tps:
        mean_tp = sum(tps) / len(tps)
        print(f"\nMean throughput across conditions: {mean_tp:.2f} bits/s")
        print("Reference: desktop mouse ~3.7-4.9 bit/s; head/eye pointers commonly ~1.5-3 bit/s.")
        print("Paper template:")
        print(f'  "AirPoint achieved a mean ISO 9241-9 throughput of {mean_tp:.2f} bits/s '
              f"(effective measures) across {len(blocks)} amplitude x width conditions.\"")


# ------------------------------- cli -------------------------------

def main(argv):
    if len(argv) < 2 or argv[1] not in ("run", "analyse", "analyze"):
        print(__doc__)
        return 1
    if argv[1] == "run":
        device = participant = None
        a = argv[2:]
        for i, t in enumerate(a):
            if t == "--device" and i + 1 < len(a):
                device = a[i + 1]
            if t == "--participant" and i + 1 < len(a):
                participant = a[i + 1]
        run(device or "device", participant or "P00")
    else:
        analyse(argv[2:] or [os.path.join(RESULTS_DIR, "fitts_*.csv")])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
