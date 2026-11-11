import json, random, re, unicodedata, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS = ["A", "B", "C"]          # 0 negatif, 1 netral, 2 positif


def load_cfg():
    with open(ROOT / "configs" / "config.json", encoding="utf8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s):
    """Normalisasi HANYA untuk pencocokan duplikat. Teks asli tidak diubah."""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def set_seed(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_split(name):
    import pandas as pd
    return pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")


# ---------- prompt dan tokenisasi ----------
def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def answer_ids(tok):
    ids = {c: tok.encode(c, add_special_tokens=False) for c in LETTERS}
    assert all(len(v) == 1 for v in ids.values()), f"Kode jawaban bukan satu token: {ids}"
    return [ids[c][0] for c in LETTERS]          # urutan A, B, C


def prompt_ids(tok, text, cfg):
    msgs = [{"role": "user", "content": cfg["prompt"].format(teks=text)}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(s, add_special_tokens=False)["input_ids"]


def fit_prompt(tok, text, cfg, max_len):
    """Potong HANYA bagian teks dari kanan agar prompt + jawaban + token akhir <= max_len."""
    reserve = 2                                    # kode jawaban + <|im_end|>
    t_ids = tok(text, add_special_tokens=False)["input_ids"]
    cut, truncated = len(t_ids), False
    ids = prompt_ids(tok, text, cfg)
    while len(ids) + reserve > max_len and cut > 0:
        base = len(prompt_ids(tok, "", cfg))
        cut = min(cut - 1, max(0, max_len - reserve - base))
        truncated = True
        ids = prompt_ids(tok, tok.decode(t_ids[:cut]), cfg)
    return ids, truncated


def train_example(tok, text, label, cfg, max_len):
    p, trunc = fit_prompt(tok, text, cfg, max_len)
    ans = [answer_ids(tok)[label], tok.convert_tokens_to_ids("<|im_end|>")]
    return p + ans, [-100] * len(p) + ans, trunc


# ---------- model ----------
def load_base_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], quantization_config=bnb, device_map={"": 0})
    model.config.use_cache = False
    return model


# ---------- prediksi dan metrik ----------
def predict(model, tok, texts, cfg, max_len):
    """Skor tiga kode (A,B,C) pada posisi jawaban. Prediksi = argmax. Tanpa pembangkitan bebas."""
    import numpy as np, torch
    kode = answer_ids(tok)
    enc = [fit_prompt(tok, t, cfg, max_len)[0] for t in texts]
    model.eval()
    out_scores = []
    bs = cfg["eval_batch"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(enc), bs):
            chunk = enc[i:i + bs]
            L = max(len(x) for x in chunk)
            ids = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in chunk]).cuda()
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in chunk]).cuda()
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            try:
                lg = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                lg = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :]
            out_scores.append(lg[:, kode].float().cpu().numpy())
    scores = np.concatenate(out_scores)
    return scores.argmax(1), scores


def metrics(y, p):
    from sklearn.metrics import f1_score, confusion_matrix
    return {
        "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
        "f1_per_kelas": [float(v) for v in f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def fast_macro_f1(y, p):
    """Macro-F1 cepat (numpy) untuk bootstrap; penyebut 0 menghasilkan F1 = 0."""
    import numpy as np
    cm = np.bincount(3 * y + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return float(f1.mean())
