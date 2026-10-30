"""Uniform weight average of HF checkpoints: python3 scripts/avg_ckpt.py OUT CKPT1 CKPT2 [...]"""
import glob, os, shutil, sys
from safetensors.torch import load_file, save_file

out, cks = sys.argv[1], sys.argv[2:]
shutil.copytree(cks[-1], out, dirs_exist_ok=True)
for f in glob.glob(f"{cks[-1]}/*.safetensors"):
    n = os.path.basename(f); sd = [load_file(f"{c}/{n}") for c in cks]
    save_file({k: (sum(s[k].float() for s in sd) / len(sd)).to(v.dtype) if v.is_floating_point() else v
               for k, v in sd[-1].items()}, f"{out}/{n}", metadata={"format": "pt"})
