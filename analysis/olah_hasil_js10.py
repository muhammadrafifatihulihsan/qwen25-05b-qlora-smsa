"""Mengolah hasil run menjadi tabel, gambar, dan ringkasan angka untuk dokumentasi.

Skrip ini tidak mengubah keputusan penelitian. Pemilihan rank tetap mengikuti
src/07_aggregate.py; skrip ini hanya menyiapkan tabel dan gambar dari runs/*/hasil.json.

Keluaran:
  results/ringkasan.json      semua angka penting, dipakai pembuat README
  results/tabel_per_seed.csv  satu baris per run
  results/tabel_agregat.csv   rata-rata dan simpangan baku per rank
  docs/figures/*.png          gambar kinerja, F1 per kelas, confusion matrix, parameter
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
GAMBAR_DIR = ROOT / "docs" / "figures"
GAMBAR_DIR.mkdir(parents=True, exist_ok=True)

CFG = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))
BARIS = ["status", "macro_f1_validasi", "parameter_terlatih", "peak_allocated_mib",
         "peak_reserved_mib", "waktu_pelatihan_detik", "ukuran_adaptor_mib", "langkah"]

KOSONG = {k: None for k in BARIS} | {
    "f1_negatif": None, "f1_netral": None, "f1_positif": None,
    "confusion_matrix": None, "loss_akhir": None, "loss_rata2_10": None,
    "train_terpotong": None,
}

rekaman = []
for rank in CFG["ranks"]:
    for seed in CFG["seeds"]:
        f = ROOT / "runs" / f"r{rank}_s{seed}" / "hasil.json"
        if not f.exists():
            rekaman.append({"rank": rank, "seed": seed, "status": "belum", **KOSONG})
            continue
        h = json.loads(f.read_text(encoding="utf-8"))
        nilai = h.get("validasi") or {}
        rekaman.append({
            "rank": rank, "seed": seed, **KOSONG,
            **{k: h.get(k) for k in BARIS},
            "f1_negatif": (nilai.get("f1_per_kelas") or [None] * 3)[0],
            "f1_netral": (nilai.get("f1_per_kelas") or [None] * 3)[1],
            "f1_positif": (nilai.get("f1_per_kelas") or [None] * 3)[2],
            "confusion_matrix": nilai.get("confusion_matrix"),
            "loss_akhir": h.get("loss_akhir"),
            "loss_rata2_10": h.get("loss_rata2_10_terakhir"),
            "train_terpotong": h.get("train_terpotong"),
        })

semua = pd.DataFrame(rekaman)
semua.to_csv(OUT / "tabel_per_seed.csv", index=False)

kolom_stat = ["macro_f1_validasi", "f1_negatif", "f1_netral", "f1_positif",
              "parameter_terlatih", "peak_allocated_mib", "peak_reserved_mib",
              "waktu_pelatihan_detik", "ukuran_adaptor_mib"]
agregat = []
for rank, g in semua.groupby("rank"):
    ok = g[g.status == "ok"]
    d = {"rank": int(rank), "jumlah_run_ok": int(len(ok)),
         "jumlah_seed_direncanakan": len(CFG["seeds"])}
    for c in kolom_stat:
        v = pd.to_numeric(ok[c], errors="coerce")
        d[c + "_mean"] = float(v.mean()) if v.notna().any() else None
        d[c + "_sd"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else None
    agregat.append(d)
agregat = pd.DataFrame(agregat).sort_values("rank")
agregat["selisih_f1_dari_terbaik"] = agregat["macro_f1_validasi_mean"].max() - agregat["macro_f1_validasi_mean"]
agregat.to_csv(OUT / "tabel_agregat.csv", index=False)

layak = agregat[(agregat.jumlah_run_ok > 0) & agregat.macro_f1_validasi_mean.notna()]
terbaik = float(layak["macro_f1_validasi_mean"].max()) if len(layak) else None
rank_pilih = None
if terbaik is not None:
    kandidat = layak[layak["macro_f1_validasi_mean"] >= terbaik - CFG["tolerance"] - 1e-12]
    if len(kandidat):
        rank_pilih = int(kandidat["rank"].min())

beku = ROOT / "results" / "rank_terpilih.json"
if beku.exists():
    # pakai rank yang sudah dibekukan oleh 07_aggregate.py agar laporan dan evaluasi uji konsisten
    rank_pilih = int(json.loads(beku.read_text(encoding="utf-8"))["rank_terpilih"])

ringkas = {
    "model": CFG["model_name"],
    "ranks": CFG["ranks"],
    "seeds_direncanakan": CFG["seeds"],
    "seeds_selesai": sorted({int(s) for s, st in zip(semua.seed, semua.status) if st == "ok"}),
    "toleransi": CFG["tolerance"],
    "macro_f1_terbaik_validasi": terbaik,
    "rank_terpilih_sementara": rank_pilih,
    "jumlah_data": {
        "latih": int(pd.read_csv(ROOT / "data/processed/train.csv").shape[0]),
        "validasi": int(pd.read_csv(ROOT / "data/processed/valid.csv").shape[0]),
        "uji": int(pd.read_csv(ROOT / "data/processed/test.csv").shape[0]),
    },
    "distribusi_latih": pd.read_csv(ROOT / "data/processed/train.csv")
                          .label_name.value_counts().to_dict(),
    "per_rank": agregat.replace({np.nan: None}).to_dict(orient="records"),
    "per_seed": semua.replace({np.nan: None}).to_dict(orient="records"),
}

# ----------------------------------- tambahan di luar protokol: rank 64 (bila ada)
extra = []
for f in sorted((ROOT / "runs").glob("r64_s*/hasil.json")):
    try:
        h = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    if h.get("status") != "ok":
        continue
    extra.append({
        "seed": h["seed"],
        "macro_f1_validasi": h.get("macro_f1_validasi"),
        "f1_netral": h.get("validasi", {}).get("f1_per_kelas", [None, None, None])[1],
        "f1_negatif": h.get("validasi", {}).get("f1_per_kelas", [None, None, None])[0],
        "f1_positif": h.get("validasi", {}).get("f1_per_kelas", [None, None, None])[2],
        "parameter_terlatih": h.get("parameter_terlatih"),
        "peak_reserved_mib": h.get("peak_reserved_mib"),
        "ukuran_adaptor_mib": h.get("ukuran_adaptor_mib"),
        "waktu_pelatihan_detik": h.get("waktu_pelatihan_detik"),
    })
if extra:
    m = [e["macro_f1_validasi"] for e in extra if e["macro_f1_validasi"] is not None]
    ringkas["tambahan_r64"] = {
        "jumlah_seed": len(m),
        "seed": [e["seed"] for e in extra],
        "macro_f1_mean": float(np.mean(m)),
        "macro_f1_sd": float(np.std(m, ddof=1)) if len(m) > 1 else None,
        "f1_negatif_mean": float(np.mean([e["f1_negatif"] for e in extra if e["f1_negatif"] is not None])),
        "f1_netral_mean": float(np.mean([e["f1_netral"] for e in extra if e["f1_netral"] is not None])),
        "f1_positif_mean": float(np.mean([e["f1_positif"] for e in extra if e["f1_positif"] is not None])),
        "parameter_terlatih": int(extra[0]["parameter_terlatih"]),
        "peak_reserved_mib_mean": float(np.mean([e["peak_reserved_mib"] for e in extra])),
        "ukuran_adaptor_mib": float(np.mean([e["ukuran_adaptor_mib"] for e in extra])),
        "waktu_menit_mean": float(np.mean([e["waktu_pelatihan_detik"] for e in extra]) / 60),
        "selisih_dari_r32": float(np.mean(m) - agregat.loc[agregat["rank"] == 32, "macro_f1_validasi_mean"].iloc[0]),
    }
else:
    ringkas["tambahan_r64"] = None

(OUT / "ringkasan.json").write_text(json.dumps(ringkas, indent=2, ensure_ascii=False), encoding="utf-8")

# ------------------------------------------------------------------ gambar 1
TINTA = "#111827"
a = agregat.dropna(subset=["macro_f1_validasi_mean"])
if len(a):
    fig, axs = plt.subplots(3, 1, figsize=(6.6, 8.2), dpi=220)
    panel = [
        ("macro_f1_validasi", "Macro-F1 validasi", "#1d4ed8"),
        ("peak_reserved_mib", "Puncak memori reserved (MiB)", "#b45309"),
        ("waktu_pelatihan_detik", "Waktu pelatihan (detik)", "#15803d"),
    ]
    for ax, (kol, label, warna) in zip(axs, panel):
        mean = a[kol + "_mean"]
        sd = a[kol + "_sd"].fillna(0)
        ax.errorbar(a["rank"], mean, yerr=sd, marker="o", color=warna,
                    capsize=3, linewidth=1.6, markersize=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(a["rank"]))
        ax.set_xticklabels([f"r={int(r)}" for r in a["rank"]], fontsize=11)
        ax.set_xlabel("Rank adaptor", fontsize=11.5)
        ax.set_ylabel(label, fontsize=11.5)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10.5)
        if kol == "macro_f1_validasi" and terbaik is not None:
            ax.axhline(terbaik - CFG["tolerance"], color="#9ca3af", linestyle="--", linewidth=1)
            ax.annotate("batas toleransi 0,01", xy=(a["rank"].min(), terbaik - CFG["tolerance"]),
                        xytext=(2, -13), textcoords="offset points", fontsize=9, color="#6b7280")
    fig.suptitle("Kinerja dan biaya komputasi menurut rank adaptor (data validasi)",
                 fontsize=12.5, color=TINTA)
    fig.tight_layout()
    fig.savefig(GAMBAR_DIR / "gambar_tradeoff.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ------------------------------------------------- gambar F1 per kelas per rank
if len(a):
    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=220)
    x = np.arange(len(a))
    lebar = 0.26
    warna = {"F1 negatif": "#1d4ed8", "F1 netral": "#c2410c", "F1 positif": "#15803d"}
    for i, (kol, nama) in enumerate([("f1_negatif", "F1 negatif"),
                                     ("f1_netral", "F1 netral"),
                                     ("f1_positif", "F1 positif")]):
        ax.bar(x + (i - 1) * lebar, a[kol + "_mean"], lebar,
               yerr=a[kol + "_sd"], capsize=2.5, color=warna[nama],
               edgecolor="white", linewidth=0.6, label=nama,
               error_kw={"elinewidth": 0.8, "ecolor": "#374151"})
    ax.set_xticks(x)
    ax.set_xticklabels([f"r={int(r)}" for r in a["rank"]], fontsize=11)
    ax.set_ylabel("F1 per kelas (validasi)", fontsize=9)
    ax.set_xlabel("Rank adaptor", fontsize=10.5)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.36))
    ax.tick_params(labelsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("F1 tiap kelas menurut rank (batang galat adalah simpangan baku antar seed)",
                 fontsize=9.5, color=TINTA)
    # titik per seed agar sebaran tiap ulangan terlihat, bukan hanya simpangan bakunya
    for i, (kol, nama) in enumerate([("f1_negatif", "F1 negatif"),
                                     ("f1_netral", "F1 netral"),
                                     ("f1_positif", "F1 positif")]):
        for j, r in enumerate(a["rank"]):
            nilai = semua[(semua["rank"] == r) & (semua["status"] == "ok")][kol].dropna()
            if len(nilai):
                ax.scatter([j + (i - 1) * lebar] * len(nilai), nilai.values, s=11,
                           color="#111827", zorder=4, linewidths=0)
    fig.tight_layout()
    fig.savefig(GAMBAR_DIR / "gambar_f1_kelas.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ------------------------------------------------------------------ gambar 2
if rank_pilih is not None:
    pilih = semua[(semua["rank"] == rank_pilih) & (semua.status == "ok")]
    if len(pilih):
        cm = np.array(pilih.iloc[0]["confusion_matrix"], dtype=int)
        fig, ax = plt.subplots(figsize=(3.9, 3.4), dpi=220)
        im = ax.imshow(cm, cmap="Blues")
        nama = ["negatif", "netral", "positif"]
        ax.set_xticks(range(3)); ax.set_xticklabels(nama, fontsize=10, rotation=18)
        ax.set_yticks(range(3)); ax.set_yticklabels(nama, fontsize=10)
        ax.set_xlabel("Prediksi", fontsize=10.5); ax.set_ylabel("Label sebenarnya", fontsize=10.5)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() / 2 else TINTA)
        ax.set_title(f"Confusion matrix validasi, rank r={rank_pilih}", fontsize=9.5, color=TINTA)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        fig.tight_layout()
        fig.savefig(GAMBAR_DIR / "gambar_confusion.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

# ------------------------------------------------------------------ gambar 3
if len(a):
    fig, ax = plt.subplots(figsize=(4.3, 2.9), dpi=220)
    ax.bar([f"r={int(r)}" for r in a["rank"]], a["parameter_terlatih_mean"],
           color="#93c5fd", edgecolor=TINTA, linewidth=0.8)
    for x, v in zip(range(len(a)), a["parameter_terlatih_mean"]):
        ax.text(x, v * 1.02, f"{int(v):,}".replace(",", "."), ha="center",
                fontsize=8, color=TINTA)
    ax.set_ylabel("Parameter terlatih", fontsize=9)
    ax.set_xlabel("Rank adaptor", fontsize=10.5)
    ax.set_ylim(0, a["parameter_terlatih_mean"].max() * 1.18)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Jumlah parameter terlatih per rank", fontsize=9.5, color=TINTA)
    fig.tight_layout()
    fig.savefig(GAMBAR_DIR / "gambar_parameter.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

print(agregat.round(4).to_string(index=False))
print("\nrank terpilih sementara:", rank_pilih, "| macro-F1 terbaik:", terbaik)
print("seeds selesai:", ringkas["seeds_selesai"])
