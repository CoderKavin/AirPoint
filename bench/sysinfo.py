#!/usr/bin/env python3
"""
Print a paper-ready hardware/software spec line for the machine you run it on, so
"name the specs" is accurate and automatic. Run this on EACH machine you benchmark
(dev box and weak laptop) and paste the line into your methods/results.

    python bench/sysinfo.py
"""
import platform, sys, os


def main():
    bits = []
    bits.append(("OS", f"{platform.system()} {platform.release()} ({platform.machine()})"))
    proc = platform.processor() or "unknown CPU"
    cores_phys = cores_log = ram_gb = None
    try:
        import psutil
        cores_phys = psutil.cpu_count(logical=False)
        cores_log = psutil.cpu_count(logical=True)
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        freq = psutil.cpu_freq()
        if freq and freq.max:
            proc += f" @ {freq.max/1000:.1f} GHz"
    except Exception:
        cores_log = os.cpu_count()
    bits.append(("CPU", proc))
    if cores_phys or cores_log:
        bits.append(("Cores", f"{cores_phys or '?'} physical / {cores_log or '?'} logical"))
    if ram_gb:
        bits.append(("RAM", f"{ram_gb} GB"))
    bits.append(("Python", platform.python_version()))
    bits.append(("GPU use", "none — MediaPipe runs on CPU (MEDIAPIPE_DISABLE_GPU=1)"))

    print("\nMachine spec (paste into the paper):")
    print("-" * 60)
    for k, v in bits:
        print(f"  {k:<8}: {v}")
    print("-" * 60)
    oneline = "; ".join(f"{k} {v}" for k, v in bits if k in ("OS", "CPU", "Cores", "RAM"))
    print(f'  One-line: "{oneline}"')
    if ram_gb is None:
        print("\n  (install psutil for CPU/RAM/cores: pip install psutil)")


if __name__ == "__main__":
    main()
