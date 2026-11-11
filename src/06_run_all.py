"""Tahap 7: 12 pelatihan utama dengan urutan rank diacak per seed. Bisa dilanjutkan jika terhenti.
Jika pilot memaksa 128 token, ubah max_len di configs/config.json DAN catat di notes/changelog.md."""
import json, random, subprocess, sys, time
from common import ROOT, load_cfg

cfg = load_cfg()
sched_f = ROOT / "configs" / "urutan_eksekusi.json"
if sched_f.exists():
    schedule = json.load(open(sched_f))
else:
    rng = random.Random(cfg["schedule_seed"]); schedule = []
    for s in cfg["seeds"]:
        rs = list(cfg["ranks"]); rng.shuffle(rs); schedule += [{"seed": s, "rank": r} for r in rs]
    json.dump(schedule, open(sched_f, "w"), indent=2)
print("Urutan:", schedule)

for item in schedule:
    name = f"r{item['rank']}_s{item['seed']}"; out = ROOT / "runs" / name
    hf = out / "hasil.json"
    if hf.exists() and json.load(open(hf)).get("status") == "ok" and (out / "pred_valid.csv").exists():
        print("lewati (sudah selesai):", name); continue
    (ROOT / "logs").mkdir(exist_ok=True)
    smi = None
    try:
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu",
                                "--format=csv", "-lms", "100", "-f", str(ROOT / "logs" / f"nvidia_smi_{name}.csv")])
    except Exception as e:
        print("nvidia-smi tidak tersedia untuk logging:", e)
    t = time.time()
    rc = subprocess.call([sys.executable, str(ROOT / "src" / "train.py"), "--rank", str(item["rank"]),
                          "--seed", str(item["seed"]), "--out", str(out)])
    if smi: smi.terminate()
    print(f"{name} selesai kode={rc} dalam {time.time()-t:.0f} detik")
    if rc != 0:
        print("PERHATIAN: kondisi gagal. Catat di notes/changelog.md. Jangan dibuang; ulangi dengan setelan sama atau revisi rancangan.")
    time.sleep(cfg["cooldown_seconds"])
