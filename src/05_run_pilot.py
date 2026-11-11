"""Tahap 5: pilot pelatihan untuk menguji kelayakan memori dan mengukur kecepatan.

Pilot memakai rank terbesar (32) karena kondisi itulah yang paling membutuhkan memori.
Pilot hanya 20 langkah dan tidak menghitung metrik validasi.

Contoh:
    python 05_run_pilot.py                # 256 token
    python 05_run_pilot.py --max_len 128  # bila terjadi kehabisan memori

Hasil pilot tersimpan di runs/pilot_r32_s42/hasil.json dan dipakai untuk memutuskan
apakah seluruh kondisi memakai 256 atau 128 token. Keputusan itu wajib dicatat di
notes/changelog.md.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, load_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_len", type=int, default=None)
a = ap.parse_args()

cfg = load_cfg()
max_len = a.max_len or cfg["max_len"]
keluar = ROOT / "runs" / f"pilot_r{a.rank}_s{a.seed}"

perintah = [
    sys.executable, str(Path(__file__).resolve().parent / "train.py"),
    "--rank", str(a.rank), "--seed", str(a.seed),
    "--out", str(keluar), "--pilot", "--max_len", str(max_len),
]
print("menjalankan:", " ".join(perintah))
hasil = subprocess.run(perintah)
print(f"exit code: {hasil.returncode}")
print(f"hasil pilot: {keluar / 'hasil.json'}")
