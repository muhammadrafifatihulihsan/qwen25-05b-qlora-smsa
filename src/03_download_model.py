"""Tahap 3: unduh model dan catat revisinya.

Contoh:
    python 03_download_model.py
    HF_ENDPOINT=https://hf-mirror.com python 03_download_model.py   # bila koneksi lambat

Model yang dipakai: Qwen/Qwen2.5-0.5B-Instruct (Apache 2.0). Bobot tidak disimpan
di dalam repositori, hanya di cache Hugging Face di komputer pemakai.
"""

import json

from huggingface_hub import HfApi, snapshot_download

from common import ROOT, load_cfg

cfg = load_cfg()
nama = cfg["model_name"]

info = HfApi().model_info(nama)
print(f"model    : {nama}")
print(f"revisi   : {info.sha}")
print(f"berkas   : {len(info.siblings or [])}")

jalur = snapshot_download(nama, allow_patterns=["*.json", "*.safetensors", "*.txt"])
print(f"tersimpan: {jalur}")

catatan = ROOT / "notes" / "model_info.json"
catatan.parent.mkdir(parents=True, exist_ok=True)
catatan.write_text(
    json.dumps({
        "model_name": nama,
        "revision": info.sha,
        "lisensi": "Apache 2.0",
        "sumber": f"https://huggingface.co/{nama}",
        "lokasi_cache": str(jalur),
    }, indent=2),
    encoding="utf-8",
)
print(f"dicatat  : {catatan}")
