"""English WER of hypothesis dir vs manifest: python3 scripts/wer.py <manifest> <hypdir>"""
import json, os, re, sys
import jiwer
n = lambda t: " ".join(re.sub(r"[^a-z' ]", " ", t.lower()).split())
R, H = [], []
for l in open(sys.argv[1]):
    r = json.loads(l)
    R.append(n(r["text"])); H.append(n(open(os.path.join(sys.argv[2], os.path.splitext(os.path.basename(r["audio_filepath"]))[0] + ".txt")).read()))
print(f"{sys.argv[2]}: WER {100 * jiwer.wer(R, H):.2f}  (n={len(R)})")
