#!/usr/bin/env python3
"""
Analyse a --benchmark trace CSV (Tier-1): end-to-end software latency, frame
rate, CPU/RAM, hand-detection rate, and stationary cursor jitter (RMS px,
raw vs smoothed).

Produce a trace first, e.g.:
    python main.py --benchmark 60                 # general run (latency/FPS/CPU)
    python main.py --benchmark 30 --profile me    # for jitter: HOLD STILL the whole time
Then:
    python bench/analyze.py bench/trace_*.csv
    python bench/analyze.py bench/trace_xxx.csv --jitter   # report only still-hold jitter

NOTE on jitter: RMS only means "jitter" if the hand was held as still as possible
for the whole trace. Run a dedicated still-hold trace for the jitter number, and a
separate shaky-hand trace for the tremor baseline.
"""
import sys, csv, math, glob, statistics as st


def _load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _f(rows, col):
    out = []
    for r in rows:
        v = r.get(col, "")
        if v != "" and v is not None:
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _rms_2d(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return math.sqrt(sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs))


def analyse(path, jitter_only=False):
    rows = _load(path)
    if not rows:
        print(f"{path}: empty")
        return
    n = len(rows)
    t_rel = _f(rows, "t_rel")
    dur = (max(t_rel) - min(t_rel)) if t_rel else 0.0
    print(f"\n=== {path} ===")
    print(f"frames={n}  duration={dur:.1f}s")

    if not jitter_only:
        total = _f(rows, "total_ms")
        cap = _f(rows, "capture_ms"); inf = _f(rows, "inference_ms"); post = _f(rows, "post_ms")
        fps = n / dur if dur > 0 else float("nan")
        det = _f(rows, "detected")
        det_rate = 100.0 * sum(det) / len(det) if det else float("nan")
        cpu = _f(rows, "cpu_pct"); mem = _f(rows, "mem_mb")

        print("\n-- Latency (software pipeline, ms) --")
        print(f"  total   : mean {st.mean(total):6.1f}  median {st.median(total):6.1f}  "
              f"p95 {_pct(total,95):6.1f}  max {max(total):6.1f}")
        print(f"  capture : mean {st.mean(cap):6.1f}   (camera read + flip)")
        print(f"  inference mean {st.mean(inf):6.1f}   (BGR->RGB + MediaPipe — the dominant cost)")
        print(f"  post    : mean {st.mean(post):6.1f}   (gesture + mapping + cursor move)")

        print("\n-- Frame rate --")
        print(f"  {fps:.1f} FPS (mean over the run)")

        print("\n-- Detection rate --")
        print(f"  {det_rate:.1f}% of frames had a hand detected")

        if cpu:
            print("\n-- Resource use (this machine — record its specs!) --")
            print(f"  CPU: mean {st.mean(cpu):.0f}%  peak {max(cpu):.0f}%   "
                  f"RAM: mean {st.mean(mem):.0f} MB  peak {max(mem):.0f} MB")
        else:
            print("\n-- Resource use: (install psutil to log CPU/RAM: pip install psutil) --")

        print("\nPaper templates:")
        print(f'  "Mean software-pipeline latency was {st.mean(total):.0f} ms '
              f'(95th percentile {_pct(total,95):.0f} ms) at {fps:.0f} FPS."')
        print(f'  "Hand detection succeeded on {det_rate:.0f}% of frames under these conditions."')

    # ---- jitter (use only detected frames) ----
    det_rows = [r for r in rows if r.get("detected") == "1"]
    ox = _f(det_rows, "out_x"); oy = _f(det_rows, "out_y")
    rx = _f(det_rows, "raw_x"); ry = _f(det_rows, "raw_y")
    print("\n-- Stationary cursor jitter (RMS px; valid ONLY if held still) --")
    if len(ox) >= 2 and len(ox) == len(oy):
        out_rms = _rms_2d(ox, oy)
        line = f"  smoothed (delivered) cursor : {out_rms:.2f} px RMS"
        if len(rx) == len(ry) >= 2:
            raw_rms = _rms_2d(rx, ry)
            fac = (raw_rms / out_rms) if out_rms > 0 else float("inf")
            line += f"\n  raw (pre-smoothing)         : {raw_rms:.2f} px RMS  ->  {fac:.1f}x reduction from smoothing"
        print(line)
        print(f'  Paper template: "Stationary cursor jitter was {out_rms:.1f} px RMS (smoothed), '
              f'versus {(_rms_2d(rx,ry) if len(rx)>=2 else float("nan")):.1f} px raw."')
    else:
        print("  (not enough detected frames with cursor output)")


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    jitter = "--jitter" in argv
    files = []
    for a in args:
        files.extend(glob.glob(a))
    if not files:
        print(__doc__)
        return 1
    for fp in files:
        analyse(fp, jitter_only=jitter)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
