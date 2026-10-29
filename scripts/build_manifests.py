"""Filter teacher pseudo-labels into a training manifest.

Keeps a chunk when the teacher output is (a) present, (b) not a translation (enough Han
characters vs. the subtitle), (c) close to the subtitle text: CER(norm(teacher), norm(subs)) <= MAX_CER,
(d) has no long repeated n-gram, (e) has a plausible char rate.
Target text = teacher output (Simplified, punctuated). Writes $STORE/manifests/yodas2_tw.jsonl.
"""
import glob, json, os, re, sys
import jiwer
sys.path.insert(0, os.path.dirname(__file__))
from score import norm

STORE = os.environ["STORE"]
MAX_CER = float(os.environ.get("MAX_CER", 0.35))
HAN = re.compile(r"[一-鿿]")


def repeated(t, n=4, k=4):
    grams = [t[i:i + n] for i in range(len(t) - n)]
    return any(grams.count(g) >= k for g in set(grams))


def keep(r):
    t, h = r["teacher"], r["human"]
    if not t:
        return "empty"
    nt, nh = norm(t), norm(h)
    if len(HAN.findall(t)) < 0.5 * len(HAN.findall(h)):
        return "translated"
    if not nh or jiwer.cer(nh, nt) > MAX_CER:
        return "disagree"
    if repeated(nt):
        return "repeat"
    rate = len(nt) / r["duration"]
    if not 1.0 <= rate <= 9.0:
        return "rate"
    return None


stats, kept, hours = {}, 0, 0.0
with open(f"{STORE}/manifests/yodas2_tw.jsonl", "w") as fo:
    for f in sorted(glob.glob(f"{STORE}/teacher_labels/yodas2/*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            why = keep(r)
            stats[why or "kept"] = stats.get(why or "kept", 0) + 1
            if why is None:
                kept += 1; hours += r["duration"] / 3600
                fo.write(json.dumps(dict(audio_filepath=r["audio_filepath"], duration=r["duration"],
                                         text=r["teacher"], lang="zh-TW"), ensure_ascii=False) + "\n")
print(stats, f"kept {kept} chunks {hours:.1f} h")
