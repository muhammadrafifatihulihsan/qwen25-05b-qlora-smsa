"""Tahap 5: pelatihan satu kondisi (satu rank, satu seed). Jalankan sebagai proses terpisah.
Contoh : python train.py --rank 8 --seed 42 --out ../runs/r8_s42
Pilot  : python train.py --rank 32 --seed 42 --out ../runs/pilot_r32 --pilot [--max_len 128]
"""
import argparse, json, math, sys, time, traceback
from pathlib import Path
from common import (load_cfg, set_seed, load_split, load_tokenizer, load_base_model,
                    train_example, predict, metrics)

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--max_len", type=int, default=None)
a = ap.parse_args()

import torch, numpy as np
import bitsandbytes as bnb
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

cfg = load_cfg(); max_len = a.max_len or cfg["max_len"]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
res = {"rank": a.rank, "seed": a.seed, "max_len": max_len, "pilot": a.pilot, "status": "started"}

def save(): json.dump(res, open(out / "hasil.json", "w"), indent=2)

try:
    set_seed(a.seed)
    tok = load_tokenizer(cfg)
    tr = load_split("train")
    ex, ntrunc = [], 0
    for t, l in zip(tr.text, tr.label):
        ids, lab, trunc = train_example(tok, t, int(l), cfg, max_len)
        ex.append((ids, lab)); ntrunc += trunc
    res["train_terpotong"] = ntrunc / len(ex)

    model = load_base_model(cfg)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=a.rank, lora_alpha=cfg["alpha_multiplier"] * a.rank, lora_dropout=cfg["lora_dropout"],
                      bias=cfg["bias"], target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    params = [p for p in model.parameters() if p.requires_grad]
    res["parameter_terlatih"] = int(sum(p.numel() for p in params))
    res["alpha"] = lora.lora_alpha

    ga = cfg["grad_accum"]; steps_per_epoch = math.ceil(len(ex) / ga)
    total = cfg["epochs"] * steps_per_epoch
    if a.pilot: total = cfg["pilot_steps"]
    opt = bnb.optim.AdamW8bit(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, round(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda")
    gen = torch.Generator().manual_seed(a.seed)

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    res["memori_setelah_muat_mib"] = torch.cuda.memory_allocated() / 2**20
    t0 = time.perf_counter(); step = 0; losses = []
    model.train()
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(ex), generator=gen).tolist()
        for g in range(0, len(order), ga):
            group = order[g:g + ga]; gl = 0.0
            for i in group:
                ids, lab = ex[i]
                x = torch.tensor([ids]).cuda(); y = torch.tensor([lab]).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=x, labels=y).loss / len(group)
                scaler.scale(loss).backward(); gl += float(loss)
            if not math.isfinite(gl):
                res["status"] = "loss_nan"; res["anomali"] = f"loss tidak finit pada langkah {step}"; save(); sys.exit(3)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(gl); step += 1
            if step % 25 == 0: print(f"langkah {step}/{total} loss {gl:.4f}", flush=True)
            if step >= total: break
        if step >= total: break
    torch.cuda.synchronize(); res["waktu_pelatihan_detik"] = time.perf_counter() - t0
    res["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    res["langkah"] = step; res["loss_akhir"] = losses[-1]; res["loss_rata2_10_terakhir"] = float(np.mean(losses[-10:]))
    res["status"] = "ok"; save()

    if not a.pilot:                                   # di luar timer
        model.save_pretrained(out / "adaptor")
        res["ukuran_adaptor_mib"] = sum(f.stat().st_size for f in (out / "adaptor").rglob("*") if f.is_file()) / 2**20
        va = load_split("valid")
        p, s = predict(model, tok, list(va.text), cfg, max_len)
        res["validasi"] = metrics(va.label.values, p)
        res["macro_f1_validasi"] = res["validasi"]["macro_f1"]
        import pandas as pd
        pd.DataFrame({"id": va.id, "label": va.label, "pred": p, "sA": s[:, 0], "sB": s[:, 1], "sC": s[:, 2]}).to_csv(out / "pred_valid.csv", index=False)
        save()
    print("SELESAI", json.dumps({k: v for k, v in res.items() if k != "validasi"}))
except torch.cuda.OutOfMemoryError:
    res["status"] = "oom"; res["anomali"] = "CUDA out of memory"; save(); print("OOM"); sys.exit(2)
except SystemExit:
    raise
except Exception:
    res["status"] = "error"; res["anomali"] = traceback.format_exc(); save(); print(res["anomali"]); sys.exit(1)
