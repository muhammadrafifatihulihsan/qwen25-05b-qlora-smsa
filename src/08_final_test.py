"""Tahap 9: evaluasi data uji SEKALI, setelah rank dibekukan. Menolak jalan jika rank belum dibekukan."""
import json
import numpy as np, pandas as pd
from common import ROOT, load_cfg, load_split, load_tokenizer, load_base_model, predict, metrics, fast_macro_f1

cfg = load_cfg()
fz = ROOT / "results" / "rank_terpilih.json"
if not fz.exists(): raise SystemExit("Jalankan 07_aggregate.py dulu. Rank harus dibekukan sebelum data uji dibuka.")
r = json.load(open(fz))["rank_terpilih"]
if (ROOT / "results" / "hasil_uji.json").exists():
    raise SystemExit("hasil_uji.json sudah ada. Data uji hanya dievaluasi sekali. Hapus manual hanya jika Anda mencatat alasannya di changelog.")
te = load_split("test"); y = te.label.values; tok = load_tokenizer(cfg)
max_len = json.load(open(ROOT / "runs" / f"r{r}_s{cfg['seeds'][0]}" / "hasil.json"))["max_len"]

from peft import PeftModel
preds = {}
base = load_base_model(cfg)
import torch
preds["zero_shot"] = predict(base, tok, list(te.text), cfg, max_len)
del base; torch.cuda.empty_cache()
for s in cfg["seeds"]:
    m = PeftModel.from_pretrained(load_base_model(cfg), str(ROOT / "runs" / f"r{r}_s{s}" / "adaptor"))
    preds[f"seed{s}"] = predict(m, tok, list(te.text), cfg, max_len)
    del m; torch.cuda.empty_cache()

out = {"rank": r, "max_len": max_len, "metrik": {}}
cols = {"id": te.id, "label": y}
for k, (p, sc) in preds.items():
    out["metrik"][k] = metrics(y, p); cols[f"pred_{k}"] = p
    assert abs(out["metrik"][k]["macro_f1"] - fast_macro_f1(y, p)) < 1e-9
# batas bawah: prediktor kelas mayoritas (kelas terbanyak pada subset latih)
tr = load_split("train"); mayor = int(tr.label.value_counts().idxmax())
out["metrik"]["kelas_mayoritas"] = metrics(y, np.full(len(y), mayor)); out["kelas_mayoritas"] = mayor
pd.DataFrame(cols).to_csv(ROOT / "results" / "pred_uji.csv", index=False)

rng = np.random.default_rng(cfg["bootstrap_seed"])
kidx = [np.where(y == k)[0] for k in (0, 1, 2)]
P = [preds[f"seed{s}"][0] for s in cfg["seeds"]]; Z = preds["zero_shot"][0]
diff = []
for _ in range(cfg["bootstrap_reps"]):
    s = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in kidx])   # indeks sama untuk semua prediksi
    diff.append(np.mean([fast_macro_f1(y[s], p[s]) for p in P]) - fast_macro_f1(y[s], Z[s]))
lo, hi = np.percentile(diff, [2.5, 97.5])
f_seed = [out["metrik"][f"seed{s}"]["macro_f1"] for s in cfg["seeds"]]
out["ringkasan"] = {"macro_f1_uji_per_seed": f_seed, "mean": float(np.mean(f_seed)), "sd": float(np.std(f_seed, ddof=1)),
                    "macro_f1_zero_shot": out["metrik"]["zero_shot"]["macro_f1"], "macro_f1_kelas_mayoritas": out["metrik"]["kelas_mayoritas"]["macro_f1"],
                    "selisih_mean_terhadap_zero_shot": float(np.mean(f_seed) - out["metrik"]["zero_shot"]["macro_f1"]),
                    "bootstrap_persentil_2_5": float(lo), "bootstrap_persentil_97_5": float(hi), "replikasi": cfg["bootstrap_reps"]}
json.dump(out, open(ROOT / "results" / "hasil_uji.json", "w"), indent=2)
print(json.dumps(out["ringkasan"], indent=2))
