"""Teacher-verify synthetic speech: keep a TTS clip when CER(teacher transcript, input text) <= MAX_CER.

Step 1 (GPU1, teacher): python3 scripts/filter_tts.py label <kind>
Step 2:                 python3 scripts/filter_tts.py filter <kind>  -> $STORE/manifests/tts_<kind>_ok.jsonl
"""
import json, os, re, subprocess, sys
from pathlib import Path
import jiwer
from opencc import OpenCC

sys.path.insert(0, os.path.dirname(__file__))
from score import norm
from prep_cv22_other import BIN, MODEL, PROMPT

STORE = os.environ["STORE"]
MAX_CER = float(os.environ.get("MAX_CER", 0.15))
key = lambda t: re.sub(r"[^\w]", "", OpenCC("t2s").convert(t))
EVAL_BANNED = set()  # never let eval sentences into training, even if synth ban missed them
for _e in ["cv120", "tt100", "cs200", "cv1k", "libri100"]:
    try:
        EVAL_BANNED |= {key(json.loads(l)["text"]) for l in open(f"{STORE}/eval/{_e}/manifest.jsonl")}
    except FileNotFoundError:
        pass
cmd, kind = sys.argv[1], sys.argv[2]
ADIR, HYP = Path(f"{STORE}/audio/tts/{kind}"), Path(f"{STORE}/teacher_labels/tts_{kind}")

if cmd == "label":
    HYP.mkdir(parents=True, exist_ok=True)
    subprocess.run([BIN, "transcribe", str(ADIR), "--model", MODEL, "--instruction", PROMPT,
                    "--output-dir", str(HYP)], check=False, stdout=subprocess.DEVNULL,  # exits 1 when labels already exist
                   stderr=open(f"{STORE}/logs/pl_tts_{kind}.err", "w"))
else:
    n = k = 0; h = 0.0; seen = set()
    with open(f"{STORE}/manifests/tts_{kind}_ok.jsonl", "w") as fo:
        for l in open(f"{STORE}/manifests/tts_{kind}.jsonl"):
            r = json.loads(l); stem = Path(r["audio_filepath"]).stem; hp = HYP / f"{stem}.txt"
            if stem in seen or not hp.exists():
                continue
            if key(r["text"]) in EVAL_BANNED:
                continue
            seen.add(stem); n += 1
            if jiwer.cer(norm(r["text"]) or "x", norm(hp.read_text())) > MAX_CER:
                continue
            k += 1; h += r["duration"]; fo.write(l)
    print(f"{kind}: checked {n}, kept {k} ({h / 3600:.1f} h)")
