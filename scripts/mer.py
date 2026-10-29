"""Mixed error rate for zh/en code-switching + English-word recall/precision.

Tokens: each Han char is a token, each English word (lowercased) is a token. Text is NFKC +
t2s normalized, punctuation dropped. Usage: python3 scripts/mer.py <manifest> <hypdir>
"""
import json, os, re, sys, unicodedata
from collections import Counter
import jiwer
from opencc import OpenCC

t2s = OpenCC("t2s")
TOK = re.compile(r"[一-鿿]|[a-z0-9']+")


def toks(t):
    return TOK.findall(t2s.convert(unicodedata.normalize("NFKC", t)).lower())


def main(manifest, hypdir):
    R, H, tp, nr, nh = [], [], 0, 0, 0
    for l in open(manifest):
        r = json.loads(l)
        stem = os.path.splitext(os.path.basename(r["audio_filepath"]))[0]
        ref, hyp = toks(r["text"]), toks(open(os.path.join(hypdir, stem + ".txt")).read())
        R.append(" ".join(ref) or "x"); H.append(" ".join(hyp))
        er = Counter(w for w in ref if w.isascii()); eh = Counter(w for w in hyp if w.isascii())
        tp += sum((er & eh).values()); nr += sum(er.values()); nh += sum(eh.values())
    print(f"{hypdir}: MER {100 * jiwer.wer(R, H):.2f}  en-recall {100 * tp / max(nr, 1):.1f}  "
          f"en-precision {100 * tp / max(nh, 1):.1f}  (n={len(R)}, en words={nr})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
