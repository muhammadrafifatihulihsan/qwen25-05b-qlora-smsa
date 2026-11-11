"""Tahap 2b-2c: audit duplikat (tanpa label uji untuk keputusan) dan subset latih terstratifikasi."""
import json, hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from common import ROOT, load_cfg, norm, sha256_file

cfg = load_cfg()
raw, out = ROOT / "data" / "raw", ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
lm = cfg["label_map"]

def read(fn):
    d = pd.read_csv(raw / fn, sep="\t", header=None, names=["text", "label_name"],
                    quoting=3, keep_default_na=False)
    d["id"] = range(len(d)); d["label"] = d["label_name"].map(lm)
    assert d["label"].notna().all(), "Ada label tak dikenal"
    d["label"] = d["label"].astype(int); d["key"] = d["text"].map(norm)
    return d

tr, va, te = (read(cfg["data_files"][k]) for k in ("train", "valid", "test"))
rep = {"awal": {"train": len(tr), "valid": len(va), "test": len(te)},
       "label_map": lm,
       "distribusi_awal": {k: d.label_name.value_counts().to_dict() for k, d in (("train", tr), ("valid", va), ("test", te))}}
assert (len(tr), len(va), len(te)) == (11000, 1260, 500), "Pembagian resmi tidak sesuai 11.000/1.260/500"

# duplikat di dalam masing-masing bagian (hanya dilaporkan untuk valid/test)
rep["duplikat_dalam"] = {k: int(d.key.duplicated().sum()) for k, d in (("train", tr), ("valid", va), ("test", te))}
conf = tr.groupby("key").label.nunique(); rep["grup_duplikat_latih_label_berbeda"] = int((conf > 1).sum())

# 1) hapus duplikat di dalam kandidat latih (pertahankan kemunculan pertama)
c = tr.drop_duplicates("key", keep="first"); rep["latih_dihapus_duplikat_dalam"] = len(tr) - len(c)
# 2) keluarkan teks latih yang ada di validasi/test
ke = set(va.key) | set(te.key)
before = len(c); c = c[~c.key.isin(ke)]; rep["latih_dihapus_overlap_valid_test"] = before - len(c)
# 3) duplikat valid-test: dikeluarkan dari validasi, dipertahankan di test
before = len(va); va2 = va[~va.key.isin(set(te.key))]; rep["valid_dihapus_overlap_test"] = before - len(va2)
rep["kandidat_latih_bersih"] = len(c)
rep["akhir_valid"] = len(va2); rep["akhir_test"] = len(te)

n = cfg["n_train_subset"]
if len(c) < n:
    raise SystemExit(f"Kandidat bersih {len(c)} < {n}. Revisi ukuran data (changelog) sebelum lanjut.")
sub_idx, _ = train_test_split(c.id.values, train_size=n, stratify=c.label.values, random_state=cfg["subset_seed"])
sub = c[c.id.isin(set(sub_idx))].sort_values("id")
rep["subset_latih"] = len(sub); rep["distribusi_subset"] = sub.label_name.value_counts().to_dict()

cols = ["id", "text", "label", "label_name"]
sub[cols].to_csv(out / "train.csv", index=False)
va2[cols].to_csv(out / "valid.csv", index=False)
te[cols].to_csv(out / "test.csv", index=False)
idx = sorted(int(i) for i in sub.id)
with open(out / "indeks_subset_2000.json", "w") as f: json.dump(idx, f)
rep["hash_indeks_subset"] = hashlib.sha256(json.dumps(idx).encode()).hexdigest()
rep["hash_berkas_olahan"] = {k: sha256_file(out / f"{k}.csv") for k in ("train", "valid", "test")}
with open(out / "laporan_duplikat.json", "w", encoding="utf8") as f: json.dump(rep, f, indent=2, ensure_ascii=False)
print(json.dumps(rep, indent=2, ensure_ascii=False))
