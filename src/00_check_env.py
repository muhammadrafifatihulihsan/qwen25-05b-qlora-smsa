"""Tahap 1: cek perangkat dan versi pustaka. Hasil disimpan ke notes/perangkat.json"""
import json, platform, subprocess, sys
from common import ROOT

info = {"python": sys.version, "os": platform.platform(), "cpu": platform.processor()}
try:
    import torch, transformers, peft, bitsandbytes, sklearn, pandas
    info.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_ok=torch.cuda.is_available(),
                transformers=transformers.__version__, peft=peft.__version__,
                bitsandbytes=bitsandbytes.__version__, sklearn=sklearn.__version__, pandas=pandas.__version__)
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        info.update(gpu=pr.name, vram_mib=pr.total_memory / 2**20, capability=list(torch.cuda.get_device_capability(0)))
except Exception as e:
    info["error"] = repr(e)
try:
    info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
except Exception as e:
    info["nvidia_smi"] = repr(e)
try:
    info["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
except Exception:
    pass
(ROOT / "notes").mkdir(exist_ok=True)
with open(ROOT / "notes" / "perangkat.json", "w", encoding="utf8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print({k: v for k, v in info.items() if k not in ("pip_freeze", "nvidia_smi")})
print("Disimpan ke notes/perangkat.json. Cocokkan dengan spesifikasi di naskah (CPU, RAM, VRAM).")
