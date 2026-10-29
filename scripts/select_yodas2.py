"""Select likely Taiwan-Mandarin (and code-switched) videos from yodas2 zh000 subtitles.

Per video: share of Traditional-only characters (TW/HK signal), Cantonese markers (drop HK),
English-word density (CS signal). Writes $STORE/yodas2_select.jsonl (one line per kept video).
Usage: python3 scripts/select_yodas2.py <meta_dir>
"""
import glob, json, os, re, sys
from opencc import OpenCC

t2s = OpenCC("t2s")
s2t = OpenCC("s2t")
CANTO = set("嘅咗唔係啲冇嘢佢哋喺嚟揾睇乜嗰")
HAN = re.compile(r"[一-鿿]")
EN = re.compile(r"[A-Za-z]{2,}")


def stats(lines):
    txt = "".join(lines)
    han = HAN.findall(txt)
    if len(han) < 200:
        return None
    h = "".join(han[:5000])
    trad = sum(a != b for a, b in zip(h, t2s.convert(h)))
    simp_only = sum(a != b for a, b in zip(h, s2t.convert(h)))
    return dict(
        han=len(han),
        trad_ratio=trad / len(h),
        simp_ratio=simp_only / len(h),
        canto=sum(c in CANTO for c in han) / len(han),
        en_per_line=sum(len(EN.findall(l)) for l in lines) / len(lines),
    )


def main(meta):
    out = open(os.path.join(os.environ["STORE"], "yodas2_select.jsonl"), "w")
    n = kept = cs = 0
    for f in sorted(glob.glob(f"{meta}/data/zh000/text/*.json")):
        shard = os.path.basename(f)[:-5]
        for vid in json.load(open(f)):
            n += 1
            s = stats(list(vid["text"].values()))
            if not s or s["trad_ratio"] < 0.1 or s["simp_ratio"] > 0.02 or s["canto"] > 0.003:
                continue
            kept += 1
            cs += s["en_per_line"] >= 0.3
            out.write(json.dumps(dict(shard=shard, audio_id=vid["audio_id"], segments=len(vid["text"]), **s)) + "\n")
    print(f"videos {n}  kept TW-like {kept}  of which CS-rich {cs}")


if __name__ == "__main__":
    main(sys.argv[1])
