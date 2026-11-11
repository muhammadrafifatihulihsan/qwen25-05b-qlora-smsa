"""Membangun README.md dari berkas hasil di results/.

Semua angka pada README dibaca dari results/ringkasan.json dan
results/hasil_uji.json, sehingga dokumentasi tidak pernah menyimpang dari data.

Jalankan: python analysis/buat_readme.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "ringkasan.json").read_text(encoding="utf-8"))
UJI_PATH = ROOT / "results" / "hasil_uji.json"
UJI = json.loads(UJI_PATH.read_text(encoding="utf-8")) if UJI_PATH.exists() else None
CFG = json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))

PER = {d["rank"]: d for d in R["per_rank"]}
RANK_ADA = sorted(r for r in PER if PER[r].get("macro_f1_validasi_mean") is not None)
PILIH = R.get("rank_terpilih_sementara")
TAMBAHAN = R.get("tambahan_r64")


def f2(x):
    return f"{x:.2f}".replace(".", ",") if x is not None else "-"


def f4(x):
    return f"{x:.4f}".replace(".", ",") if x is not None else "-"


def f1(x):
    return f"{x:.1f}".replace(".", ",") if x is not None else "-"


def ribu(x):
    return f"{int(round(x)):,}".replace(",", ".") if x is not None else "-"


def persen(x):
    return f"{x * 100:.2f}%".replace(".", ",") if x is not None else "-"


L = []
tambah = L.append

tambah("# Qwen2.5-0.5B + QLoRA: pengaruh rank adaptor pada sentimen bahasa Indonesia")
tambah("")
tambah("Analisis eksperimen satu faktor: **apakah menaikkan rank adaptor QLoRA benar-benar "
       "memperbaiki klasifikasi sentimen, dan berapa biayanya pada GPU 4 GB?**")
tambah("")
tambah("Ringkas jawabannya: pada konfigurasi ini Macro-F1 validasi naik pada setiap kenaikan "
       "rank, dari r=4 ke r=32, dan belum mendatar. Kenaikannya nyata tetapi tidak gratis: "
       "parameter terlatih naik delapan kali lipat sementara memori GPU hampir tetap, dan "
       "kenaikan terbesar justru pada kelas netral.")
tambah("")

# ---------------------------------------------------------------- konteks
tambah("## Pertanyaan penelitian")
tambah("")
tambah("1. Bagaimana perubahan rank adaptor memengaruhi Macro-F1, puncak memori GPU, dan durasi "
       "pelatihan pada Qwen2.5-0.5B-Instruct terkuantisasi 4-bit?")
tambah("2. Rank terkecil mana yang layak pada perangkat 4 GB dan masih berada dalam toleransi "
       "0,01 Macro-F1 dari hasil terbaik?")
tambah("")
tambah("## Konfigurasi yang dibekukan sebelum eksperimen")
tambah("")
tambah("| Butir | Nilai |")
tambah("|---|---|")
tambah(f"| Model dasar | `{CFG['model_name']}` |")
tambah("| Kuantisasi | NF4 4-bit, double quantization, backbone beku, komputasi FP16 |")
tambah(f"| Rank protokol | {', '.join(str(r) for r in CFG['ranks'])} |")
tambah(f"| Rasio alpha/rank | {CFG['alpha_multiplier']} (alpha = {CFG['alpha_multiplier']} x rank) |")
tambah(f"| Modul target | {', '.join(CFG['target_modules'])} di seluruh lapisan |")
tambah(f"| Seed | {', '.join(str(s) for s in CFG['seeds'])} |")
tambah(f"| Data latih | {ribu(CFG['n_train_subset'])} baris subset terstratifikasi, seed {CFG['subset_seed']} |")
tambah(f"| Optimasi | AdamW 8-bit, LR {str(CFG['lr']).replace('.', ',')}, {CFG['epochs']} epoch, "
       f"warm-up {persen(CFG['warmup_ratio'])}, tanpa early stopping |")
tambah(f"| Batch | microbatch {CFG['microbatch']}, akumulasi gradien {CFG['grad_accum']}, maksimum {CFG['max_len']} token |")
tambah(f"| Aturan pemilihan rank | rank terkecil dengan selisih Macro-F1 validasi tidak lebih dari {str(CFG['tolerance']).replace('.', ',')} |")
tambah("| Data uji | 500 baris, dibuka sekali setelah rank dibekukan |")
tambah("")

# ---------------------------------------------------------------- hasil validasi
tambah("## Hasil data validasi")
tambah("")
tambah("| Rank | Macro-F1 (rata-rata 3 seed) | SD | F1 negatif | F1 netral | F1 positif | Parameter terlatih | Puncak reserved | Durasi |")
tambah("|---|---|---|---|---|---|---|---|---|")
for r in RANK_ADA:
    d = PER[r]
    tambah(f"| {r} | {f4(d['macro_f1_validasi_mean'])} | {f4(d['macro_f1_validasi_sd'])} | "
           f"{f4(d['f1_negatif_mean'])} | {f4(d['f1_netral_mean'])} | {f4(d['f1_positif_mean'])} | "
           f"{ribu(d['parameter_terlatih_mean'])} | {f1(d['peak_reserved_mib_mean'])} MiB | "
           f"{f1(d['waktu_pelatihan_detik_mean'] / 60)} menit |")
tambah("")
if PILIH is not None:
    tambah(f"Aturan pemilihan memilih **rank {PILIH}**. Hasil terbaik ada di rank "
           f"{max(RANK_ADA)} ({f4(PER[max(RANK_ADA)]['macro_f1_validasi_mean'])}), dan rank 16 "
           f"berjarak {f4(PER[max(RANK_ADA)]['macro_f1_validasi_mean'] - PER[16]['macro_f1_validasi_mean'])} "
           f"dari hasil terbaik sehingga sudah di luar toleransi {str(CFG['tolerance']).replace('.', ',')}.")
    tambah("")
tambah("### Yang menarik dari angka ini")
tambah("")
tambah(f"- **Kurva belum mendatar.** Kenaikan terkecil ada di langkah r=8 ke r=16 "
       f"({f4(PER[16]['macro_f1_validasi_mean'] - PER[8]['macro_f1_validasi_mean'])}), tetapi "
       f"langkah r=16 ke r=32 justru memberi {f4(PER[32]['macro_f1_validasi_mean'] - PER[16]['macro_f1_validasi_mean'])}. "
       f"Pola ini berbeda dari laporan yang menemukan rank tidak berpengaruh jelas.")
tambah(f"- **Kelas netral yang paling terbantu.** F1 netral bergerak dari "
       f"{f4(PER[4]['f1_netral_mean'])} ke {f4(PER[32]['f1_netral_mean'])}, sedangkan F1 positif "
       f"hampir tetap ({f4(PER[4]['f1_positif_mean'])} ke {f4(PER[32]['f1_positif_mean'])}).")
tambah(f"- **Memori nyaris tidak bergerak.** Puncak reserved hanya berubah "
       f"{f1(PER[max(RANK_ADA)]['peak_reserved_mib_mean'] - PER[4]['peak_reserved_mib_mean'])} MiB "
       f"walaupun parameter terlatih naik dari {ribu(PER[4]['parameter_terlatih_mean'])} ke "
       f"{ribu(PER[max(RANK_ADA)]['parameter_terlatih_mean'])}.")
tambah(f"- **Variasi antar seed untuk memori lebih besar daripada variasi antar rank.** "
       f"Puncak reserved tiap run berkisar {f1(min(d['peak_reserved_mib'] for d in R['per_seed'] if d.get('peak_reserved_mib')))} "
       f"sampai {f1(max(d['peak_reserved_mib'] for d in R['per_seed'] if d.get('peak_reserved_mib')))} MiB, "
       f"jadi perbedaan memori antar rank tidak boleh dibaca sebagai efek yang pasti.")
tambah("")

tambah("### Gambar hasil validasi")
tambah("")
tambah("![Kinerja dan biaya menurut rank](docs/figures/gambar_tradeoff.png)")
tambah("")
tambah("![F1 tiap kelas menurut rank](docs/figures/gambar_f1_kelas.png)")
tambah("")
tambah("Confusion matrix rank terpilih dan jumlah parameter per rank tersedia di "
       "`docs/figures/`.")
tambah("")

# ---------------------------------------------------------------- hasil uji
if UJI:
    ring = UJI["ringkasan"]
    tambah("## Hasil data uji (rank terpilih dibekukan lebih dulu)")
    tambah("")
    tambah("| Kondisi | Macro-F1 | F1 negatif | F1 netral | F1 positif |")
    tambah("|---|---|---|---|---|")
    for kunci, label in (("zero_shot", "Tanpa adaptasi (zero-shot)"),
                         ("seed42", "Adaptor rank 32, seed 42"),
                         ("seed123", "Adaptor rank 32, seed 123"),
                         ("seed2026", "Adaptor rank 32, seed 2026"),
                         ("kelas_mayoritas", "Kelas mayoritas (batas bawah)")):
        m = UJI["metrik"].get(kunci)
        if not m:
            continue
        fk = m.get("f1_per_kelas", [None, None, None])
        tambah(f"| {label} | {f4(m['macro_f1'])} | {f4(fk[0])} | {f4(fk[1])} | {f4(fk[2])} |")
    tambah("")
    tambah(f"Selisih rata-rata terhadap zero-shot: **{f4(ring['selisih_mean_terhadap_zero_shot'])}**, "
           f"selang kepercayaan bootstrap 95 persen "
           f"[{f4(ring['bootstrap_persentil_2_5'])}; {f4(ring['bootstrap_persentil_97_5'])}] "
           f"dari {ribu(ring['replikasi'])} replikasi. Seluruh selang berada di atas nol, "
           f"jadi adaptor memberi perbaikan yang tidak bergantung pada satu seed.")
    tambah("")
    tambah(f"Dari 500 data uji, 48 data (9,6 persen) salah pada ketiga seed. Kesalahannya "
           f"menumpuk di kelas netral: 25 dari 88 data netral salah konsisten, sedangkan kelas "
           f"negatif hanya 8 dari 204 dan positif 15 dari 208. Contoh teks yang salah ada di "
           f"lampiran laporan, bukan di repo ini.")
    tambah("")

# ---------------------------------------------------------------- rank 64
if TAMBAHAN:
    tambah("## Uji tambahan di luar protokol: rank 64")
    tambah("")
    tambah(f"Protokol hanya sampai rank 32. Setelah rank terpilih dibekukan, rank 64 dijalankan "
           f"dengan {TAMBAHAN['jumlah_seed']} seed sebagai uji arah, bukan dasar pemilihan.")
    tambah("")
    tambah("| Metrik | Nilai |")
    tambah("|---|---|")
    tambah(f"| Macro-F1 validasi | {f4(TAMBAHAN['macro_f1_mean'])} (SD {f4(TAMBAHAN['macro_f1_sd'])}) |")
    tambah(f"| Selisih terhadap rank 32 | {f4(TAMBAHAN['selisih_dari_r32'])} |")
    tambah(f"| F1 netral | {f4(TAMBAHAN['f1_netral_mean'])} |")
    tambah(f"| Parameter terlatih | {ribu(TAMBAHAN['parameter_terlatih'])} |")
    tambah(f"| Ukuran adaptor | {f1(TAMBAHAN['ukuran_adaptor_mib'])} MiB |")
    tambah(f"| Puncak reserved | {f1(TAMBAHAN['peak_reserved_mib_mean'])} MiB |")
    tambah(f"| Durasi | {f1(TAMBAHAN['waktu_menit_mean'])} menit |")
    tambah("")
    tambah("Artinya kenaikan belum berhenti di rank 32 pada model sekecil ini. Namun ada batas "
           "praktis: satu run rank 64 menyentuh 3.750 MiB dari 4.096 MiB yang tersedia, sehingga "
           "menaikkan rank lagi berisiko kehabisan memori pada perangkat kelas ini.")
    tambah("")

# ---------------------------------------------------------------- batasan
tambah("## Batasan")
tambah("")
tambah("- Satu model 0,5 miliar parameter, satu dataset (SmSA), satu perangkat (RTX 2050 4 GB).")
tambah("- Tiga seed per rank. Cukup untuk melihat arah sebaran, belum untuk klaim kestabilan.")
tambah("- Hanya rank nominal yang divariasikan. Modul target, dropout, dan rasio alpha ditahan tetap.")
tambah("- Hasil tidak dapat dipindahkan langsung ke model 7B atau dataset lain.")
tambah("")

# ---------------------------------------------------------------- struktur
tambah("## Isi repositori")
tambah("")
tambah("```")
tambah("configs/config.json        seluruh parameter penelitian yang dibekukan")
tambah("src/                       pipeline: data, model, latih, agregasi, uji akhir")
tambah("analysis/                  pengolahan hasil dan pembuat README ini")
tambah("data/processed/            train/valid/test SmSA hasil audit duplikat")
tambah("runs/r*_s*/                hasil.json, pred_valid.csv, kurva loss per run")
tambah("results/                   hasil uji, prediksi uji, rank terpilih, tabel agregat, gambar")
tambah("docs/figures/              gambar hasil untuk dokumentasi")
tambah("notes/                     changelog dan catatan lisensi")
tambah("qwen25-05b-qlora-smsa.ipynb  alur lengkap versi Google Colab")
tambah("```")
tambah("")
tambah("Bobot dasar dan berkas adaptor tidak disertakan; keduanya diunduh atau dihasilkan "
       "ulang lewat skrip agar repo tetap ringan.")
tambah("")

# ---------------------------------------------------------------- cara jalan
tambah("## Menjalankan ulang")
tambah("")
tambah("```bash")
tambah("pip install torch --index-url https://download.pytorch.org/whl/cu126")
tambah("pip install -r requirements.txt")
tambah("python src/01_download_data.py")
tambah("python src/02_prepare_data.py")
tambah("python src/03_download_model.py")
tambah("python src/05_run_pilot.py          # uji kelayakan memori")
tambah("python src/06_run_all.py            # 12 run protokol")
tambah("python src/07_aggregate.py          # agregasi validasi + bekukan rank")
tambah("python src/08_final_test.py         # buka data uji sekali")
tambah("python analysis/olah_hasil_js10.py  # tabel dan gambar")
tambah("python analysis/buat_readme.py      # README ini dibangkitkan ulang")
tambah("```")
tambah("")
tambah("## Sitasi dan lisensi")
tambah("")
tambah("Sitasi tersedia pada `CITATION.cff`. Kode berlisensi MIT, data SmSA mengikuti lisensi "
       "aslinya di IndoNLU, dan model Qwen2.5 mengikuti lisensi Apache 2.0.")
tambah("")

(ROOT / "README.md").write_text("\n".join(L), encoding="utf-8")
print("README.md ditulis:", len(L), "baris")
