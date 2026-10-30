"""Synthetic zh-TW / code-switched speech with Fun-CosyVoice3-0.5B (Apache-2.0), zero-shot cloned
from CC0 Common Voice zh-TW speakers. Run in the TTS venv ($STORE/venv_tts), GPU1.

Texts:
  - zh: CV22 zh-TW validated_sentences (CC0), minus every sentence of the eval sets
  - gcs: Granary-v2 zh code_switched_text (OpenMDW-1.1)
  - cs: teacher labels of yodas2 train chunks with >= 1 English word (CS200 held-out videos excluded)
Voices: random CV16 zh-TW train clips (CC0) of 3-8 s, with their transcript as prompt text.
Target text = input text (t2s). Output: $STORE/audio/tts/<kind>/*.flac + $STORE/manifests/tts_<kind>.jsonl

Usage: $STORE/venv_tts/bin/python scripts/synth_tts.py <zh|cs> <n_utts> [seed]
"""
import csv, json, os, random, re, sys
from pathlib import Path

STORE = os.environ["STORE"]
sys.path.insert(0, f"{STORE}/CosyVoice"); sys.path.insert(0, f"{STORE}/CosyVoice/third_party/Matcha-TTS")
import soundfile as sf
import torch, torchaudio
from opencc import OpenCC
from cosyvoice.cli.cosyvoice import AutoModel

kind, n = sys.argv[1], int(sys.argv[2]); random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
t2s = OpenCC("t2s"); EN = re.compile(r"[A-Za-z]{2,}"); HAN = re.compile(r"[一-鿿]")
key = lambda t: re.sub(r"[^\w]", "", t2s.convert(t))

banned = set()
for e in ["cv120", "tt100", "cs200", "cv1k", "libri100"]:
    try:
        banned |= {key(json.loads(l)["text"]) for l in open(f"{STORE}/eval/{e}/manifest.jsonl")}
    except FileNotFoundError:
        pass

if kind == "zh":
    texts = [r["sentence"] for r in csv.DictReader(open(f"{STORE}/raw/cv22_zhtw/validated_sentences.tsv"),
                                                    delimiter="\t", quoting=csv.QUOTE_NONE)]
    texts = [t for t in texts if 4 <= len(HAN.findall(t)) and key(t) not in banned]
elif kind == "gcs":  # Granary-v2 zh code_switched_text (OpenMDW-1.1)
    import gzip
    texts = [r.get("code_switched_text") or "" for r in map(json.loads, gzip.open(
        f"{STORE}/raw/granary_v2_zh/zh_asr.jsonl.gz", "rt"))]
    texts = [t for t in texts if EN.search(t) and key(t) not in banned]
else:
    texts = [json.loads(l)["text"] for l in open(f"{STORE}/manifests/yodas2_tw_train.jsonl")]
    texts = [t for t in texts if EN.search(t) and 8 <= len(t) <= 80 and key(t) not in banned]
texts = sorted(set(texts))
print(kind, len(texts), "candidate texts", flush=True)

voices = [json.loads(l) for l in open(f"{STORE}/manifests/cv_zhtw.jsonl")]
voices = [v for v in voices if 3.0 <= v["duration"] <= 8.0]

tts = AutoModel(model_dir=f"{STORE}/tts_models/CosyVoice3")
OUT = Path(f"{STORE}/audio/tts/{kind}"); OUT.mkdir(parents=True, exist_ok=True)
man = open(f"{STORE}/manifests/tts_{kind}.jsonl", "a")
done = {p.stem for p in OUT.glob("*.flac")}
k = 0
for i in range(n):
    stem = f"{kind}{sys.argv[3] if len(sys.argv) > 3 else 0}_{i:06d}"
    text, v = random.choice(texts), random.choice(voices)
    if stem in done:
        continue
    try:
        out = [j["tts_speech"] for j in tts.inference_zero_shot(
            text, "You are a helpful assistant.<|endofprompt|>" + v["text"], v["audio_filepath"], stream=False)]
    except Exception as ex:  # noqa: BLE001 - skip odd texts
        print("skip", text, ex, flush=True); continue
    wav = torchaudio.functional.resample(torch.cat(out, 1), tts.sample_rate, 16000)[0].numpy()
    d = len(wav) / 16000
    if not 1.0 <= d <= 20.0 or d > 0.6 * len(text) + 3:  # runaway / truncated generations
        continue
    sf.write(OUT / f"{stem}.flac", wav, 16000)
    man.write(json.dumps(dict(audio_filepath=str(OUT / f"{stem}.flac"), duration=d, text=t2s.convert(text),
                              lang="zh-TW", tts_voice=v["audio_filepath"]), ensure_ascii=False) + "\n")
    man.flush(); k += 1
    if k % 200 == 0:
        print(k, "synthesized", flush=True)
print("done", k)
