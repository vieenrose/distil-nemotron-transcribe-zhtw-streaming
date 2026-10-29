"""Build small fixed eval sets (16 kHz mono WAV + JSONL manifest) under $STORE/eval."""
import io, json, os, subprocess, sys, tarfile
import pyarrow.parquet as pq

STORE = os.environ["STORE"]
RAW = f"{STORE}/eval/raw"

def to_wav(data: bytes, out: str):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", out],
                   input=data, check=True)

def dur(path):
    import soundfile as sf
    return sf.info(path).duration

def write(name, items):
    d = f"{STORE}/eval/{name}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/manifest.jsonl", "w") as m:
        for i, (audio, text) in enumerate(items):
            wav = f"{d}/{i:04d}.wav"; to_wav(audio, wav)
            m.write(json.dumps({"audio_filepath": wav, "duration": dur(wav), "text": text}, ensure_ascii=False) + "\n")
    print(name, len(items))

n_cv, n_tt = int(sys.argv[1]), int(sys.argv[2])
t = pq.read_table(f"{RAW}/cv/data/test-00000-of-00003.parquet", columns=["audio", "sentence"]).slice(0, n_cv).to_pylist()
write(f"cv{n_cv}", [(r["audio"]["bytes"], r["sentence"]) for r in t])

items, cur = [], {}
with tarfile.open(f"{RAW}/tt/test/test-000000.tar") as tf:
    for m in tf:
        key, ext = m.name.rsplit(".", 1)
        cur.setdefault(key, {})[ext] = tf.extractfile(m).read()
        if {"mp3", "txt"} <= cur[key].keys():
            items.append((cur[key]["mp3"], cur.pop(key)["txt"].decode().strip()))
            if len(items) == n_tt: break
write(f"tt{n_tt}", items)
