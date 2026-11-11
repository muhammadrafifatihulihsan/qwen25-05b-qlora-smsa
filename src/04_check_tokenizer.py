"""Tahap 3: cek kode jawaban satu token, contoh tokenisasi, dan proporsi teks terpotong."""
import json
from common import ROOT, load_cfg, load_tokenizer, answer_ids, load_split, fit_prompt, train_example

cfg = load_cfg(); tok = load_tokenizer(cfg)
print("ID token A,B,C:", answer_ids(tok), "(assert satu token lolos)")
tr = load_split("train")
for i in range(3):
    ids, labels, _ = train_example(tok, tr.text[i], int(tr.label[i]), cfg, cfg["max_len"])
    n_prompt = sum(1 for l in labels if l == -100)
    print(f"\n--- contoh {i}: panjang {len(ids)}, label {tr.label_name[i]}")
    print("PROMPT :", repr(tok.decode(ids[:n_prompt])[-160:]))
    print("JAWABAN:", repr(tok.decode(ids[n_prompt:])))
rep = {}
for name in ("train", "valid", "test"):
    d = load_split(name)
    for L in (256, 128):
        flags = [fit_prompt(tok, t, cfg, L)[1] for t in d.text]
        rep[f"{name}_maxlen{L}_terpotong"] = round(sum(flags) / len(flags), 4)
print("\nProporsi terpotong:", json.dumps(rep, indent=2))
with open(ROOT / "notes" / "proporsi_terpotong.json", "w") as f: json.dump(rep, f, indent=2)
