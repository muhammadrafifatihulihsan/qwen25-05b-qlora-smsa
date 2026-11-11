"""Tahap 2a: unduh SmSA (IndoNLU) dari GitHub, simpan hash SHA-256."""
import json, urllib.request
from common import ROOT, load_cfg, sha256_file

cfg = load_cfg()
raw = ROOT / "data" / "raw"; raw.mkdir(parents=True, exist_ok=True)
rec = {}
for split, fn in cfg["data_files"].items():
    url = f"{cfg['data_base_url']}/{fn}"
    dst = raw / fn
    urllib.request.urlretrieve(url, dst)
    rec[fn] = {"url": url, "sha256": sha256_file(dst)}
    print(fn, rec[fn]["sha256"])
for name, url in {"LICENSE_IndoNLU.txt": "https://raw.githubusercontent.com/IndoNLP/indonlu/master/LICENSE"}.items():
    urllib.request.urlretrieve(url, raw / name)
with open(raw / "SHA256.json", "w", encoding="utf8") as f:
    json.dump(rec, f, indent=2)
print("Selesai. Lisensi: lihat notes/lisensi.md dan data/raw/LICENSE_IndoNLU.txt")
