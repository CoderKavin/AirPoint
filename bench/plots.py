#!/usr/bin/env python3
"""
Make paper-ready figures from the CSVs you collect. Needs matplotlib
(pip install matplotlib). Saves PNGs next to the inputs.

    python bench/plots.py latency bench/trace_devbox.csv [more.csv ...]
        -> latency histogram (one panel per machine), p95 marked

    python bench/plots.py jitter  bench/trace_still.csv bench/trace_shaky.csv
        -> cursor-position scatter, still vs shaky (smoothed), centred on mean

    python bench/plots.py fitts   bench/results/fitts_*.csv
        -> Fitts MT-vs-IDe regression (the model fit, with R^2) + throughput bars
"""
import sys, os, csv, math, glob


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col(rows, c):
    out = []
    for r in rows:
        v = r.get(c, "")
        if v not in ("", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def latency(paths, plt):
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, p in zip(axes[0], paths):
        tot = _col(_load(p), "total_ms")
        ax.hist(tot, bins=40, color="#00a89a", alpha=0.85)
        mean, p95 = sum(tot) / len(tot), _pct(tot, 95)
        ax.axvline(mean, color="#222", ls="-", lw=1.5, label=f"mean {mean:.0f} ms")
        ax.axvline(p95, color="#d23", ls="--", lw=1.5, label=f"p95 {p95:.0f} ms")
        ax.set_title(os.path.basename(p))
        ax.set_xlabel("software pipeline latency (ms)")
        ax.set_ylabel("frames")
        ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(paths[0]) or ".", "fig_latency.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def jitter(paths, plt):
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#00a89a", "#d23", "#36c", "#e90"]
    for i, p in enumerate(paths):
        rows = [r for r in _load(p) if r.get("detected") == "1"]
        xs, ys = _col(rows, "out_x"), _col(rows, "out_y")
        if len(xs) < 2:
            continue
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        dx = [x - mx for x in xs]; dy = [y - my for y in ys]
        rms = math.sqrt(sum(a * a + b * b for a, b in zip(dx, dy)) / len(dx))
        label = f"{os.path.basename(p)}  (RMS {rms:.1f} px)"
        ax.scatter(dx, dy, s=6, alpha=0.4, color=colors[i % len(colors)], label=label)
    ax.set_aspect("equal")
    ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
    ax.set_xlabel("Δx from mean (px)"); ax.set_ylabel("Δy from mean (px)")
    ax.set_title("Stationary cursor scatter (smoothed)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(paths[0]) or ".", "fig_jitter.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def fitts(paths, plt):
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fitts as F
    rows = []
    for fp in files:
        for r in _load(fp):
            for k in ("A", "W", "target_x", "target_y", "prev_x", "prev_y",
                      "click_x", "click_y", "mt_s", "hit"):
                r[k] = float(r[k])
            rows.append(r)
    blocks = {}
    for r in rows:
        blocks.setdefault((r["A"], r["W"]), []).append(r)
    pts, tps = [], []
    for (A, W), rs in sorted(blocks.items()):
        res = F._throughput_for_block(rs)
        if res:
            pts.append((res["IDe"], res["MT"]))
            tps.append(((A, W), res["TP"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    # MT vs IDe regression (Fitts' law)
    ide = [a for a, _ in pts]; mt = [b for _, b in pts]
    ax1.scatter(ide, mt, color="#00a89a", s=40)
    if len(ide) >= 2:
        n = len(ide); sx, sy = sum(ide), sum(mt)
        sxx = sum(x * x for x in ide); sxy = sum(x * y for x, y in zip(ide, mt))
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a = (sy - b * sx) / n
        ybar = sy / n
        ss_t = sum((y - ybar) ** 2 for y in mt)
        ss_r = sum((y - (a + b * x)) ** 2 for x, y in zip(ide, mt))
        r2 = 1 - ss_r / ss_t if ss_t else float("nan")
        xr = [min(ide), max(ide)]
        ax1.plot(xr, [a + b * x for x in xr], color="#d23",
                 label=f"MT = {a*1000:.0f} + {b*1000:.0f}·IDe ms\nR² = {r2:.3f}")
        ax1.legend(fontsize=9)
    ax1.set_xlabel("effective index of difficulty IDe (bits)")
    ax1.set_ylabel("movement time MT (s)")
    ax1.set_title("Fitts' law fit")
    # throughput per condition
    labels = [f"A{int(a)}/W{int(w)}" for (a, w), _ in tps]
    ax2.bar(labels, [t for _, t in tps], color="#36c")
    ax2.set_ylabel("throughput (bits/s)")
    ax2.set_title("Effective throughput by condition")
    ax2.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(files[0]) or ".", "fig_fitts.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def main(argv):
    if len(argv) < 3 or argv[1] not in ("latency", "jitter", "fitts"):
        print(__doc__)
        return 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Needs matplotlib: pip install matplotlib")
        return 1
    {"latency": latency, "jitter": jitter, "fitts": fitts}[argv[1]](argv[2:], plt)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
