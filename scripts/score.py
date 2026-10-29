"""CER of hypothesis dir (NNNN.txt) vs manifest, both normalised to Simplified, punctuation/space stripped."""
import json, re, sys, unicodedata
from pathlib import Path
import jiwer
from opencc import OpenCC

t2s = OpenCC("t2s")

def norm(s):
    s = t2s.convert(unicodedata.normalize("NFKC", s)).lower()
    return "".join(c for c in s if c.isalnum())

def main(manifest, hypdir):
    refs, hyps = [], []
    for i, line in enumerate(open(manifest)):
        r = json.loads(line)
        h = Path(hypdir, Path(r["audio_filepath"]).stem + ".txt")
        refs.append(norm(r["text"])); hyps.append(norm(h.read_text() if h.exists() else ""))
    print(f"{hypdir}: CER {100*jiwer.cer(refs, hyps):.2f}  (n={len(refs)})")

if __name__ == "__main__":
    main(*sys.argv[1:3])
