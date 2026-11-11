"""Tahap 8: agregasi validasi dan pemilihan rank (hanya validasi). Data uji tidak disentuh."""
import json, datetime
import numpy as np, pandas as pd
from common import ROOT, load_cfg

cfg = load_cfg(); rows = []
for r in cfg["ranks"]:
    for s in cfg["seeds"]:
        f = ROOT / "runs" / f"r{r}_s{s}" / "hasil.json"
        h = json.load(open(f)) if f.exists() else {"status": "belum"}
        rows.append({"rank": r, "seed": s, **{k: h.get(k) for k in
            ["status", "macro_f1_validasi", "parameter_terlatih", "peak_allocated_mib", "peak_reserved_mib", "waktu_pelatihan_detik", "ukuran_adaptor_mib"]},
            "f1_negatif": h.get("validasi", {}).get("f1_per_kelas", [None]*3)[0],
            "f1_netral": h.get("validasi", {}).get("f1_per_kelas", [None]*3)[1],
            "f1_positif": h.get("validasi", {}).get("f1_per_kelas", [None]*3)[2]})
df = pd.DataFrame(rows); df.to_csv(ROOT / "results" / "semua_run_validasi.csv", index=False)

agg = []
for r, g in df.groupby("rank"):
    ok = (g.status == "ok").all() and len(g) == len(cfg["seeds"])
    d = {"rank": r, "layak": bool(ok)}
    for c in ["macro_f1_validasi", "f1_negatif", "f1_netral", "f1_positif", "peak_allocated_mib", "peak_reserved_mib", "waktu_pelatihan_detik", "parameter_terlatih", "ukuran_adaptor_mib"]:
        v = pd.to_numeric(g[c], errors="coerce")
        d[c + "_mean"] = v.mean(); d[c + "_sd"] = v.std(ddof=1)
    agg.append(d)
agg = pd.DataFrame(agg).sort_values("rank")
agg["selisih_dari_terbaik"] = agg.macro_f1_validasi_mean.max() - agg.macro_f1_validasi_mean
agg["lebih_kecil_dari_sd"] = agg.selisih_dari_terbaik < agg.macro_f1_validasi_sd.mean()   # selisih tak terbedakan dari variasi antar-seed
agg.to_csv(ROOT / "results" / "tabel_rank_validasi.csv", index=False)
print(agg.round(4).to_string(index=False))

feas = agg[agg.layak]
if feas.empty:
    print("\nSimpulan sah: tidak ada rank layak. Rancangan awal belum layak."); raise SystemExit
best = feas.macro_f1_validasi_mean.max()
cand = feas[feas.macro_f1_validasi_mean >= best - cfg["tolerance"] - 1e-12]
pick = int(cand["rank"].min())
rec = {"rank_terpilih": pick, "macro_f1_validasi_terbaik": float(best), "toleransi": cfg["tolerance"],
       "dibekukan_pada": datetime.datetime.now().astimezone().isoformat()}
json.dump(rec, open(ROOT / "results" / "rank_terpilih.json", "w"), indent=2)
print("\nRank terpilih (DIBEKUKAN):", rec)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].errorbar(agg["rank"], agg.macro_f1_validasi_mean, yerr=agg.macro_f1_validasi_sd, marker="o", capsize=3)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("Rank"); ax[0].set_ylabel("Macro-F1 validasi")
ax[1].errorbar(agg["rank"], agg.peak_reserved_mib_mean, yerr=agg.peak_reserved_mib_sd, marker="o", capsize=3)
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("Rank"); ax[1].set_ylabel("Peak reserved (MiB)")
ax[2].errorbar(agg["rank"], agg.waktu_pelatihan_detik_mean, yerr=agg.waktu_pelatihan_detik_sd, marker="o", capsize=3)
ax[2].set_xscale("log", base=2); ax[2].set_xlabel("Rank"); ax[2].set_ylabel("Waktu pelatihan (detik)")
plt.tight_layout(); plt.savefig(ROOT / "results" / "grafik_rank.png", dpi=200)
