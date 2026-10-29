"""CV16.1 zh-TW clean (CC0) train+validation -> 16 kHz FLAC + manifest with Simplified targets.
Test split excluded (speakers shared with the CV120 eval). Writes $STORE/manifests/cv_zhtw.jsonl."""
import glob, io, json, os
import pyarrow.parquet as pq, soundfile as sf, librosa
from opencc import OpenCC

STORE = os.environ["STORE"]; t2s = OpenCC("t2s")
adir = f"{STORE}/audio/cv_zhtw"; os.makedirs(adir, exist_ok=True); os.makedirs(f"{STORE}/manifests", exist_ok=True)
with open(f"{STORE}/manifests/cv_zhtw.jsonl", "w") as fo:
    for f in sorted(glob.glob(f"{STORE}/raw/cv_zhtw/*.parquet")):
        for r in pq.read_table(f, columns=["audio", "sentence", "path"]).to_pylist():
            out = f"{adir}/{os.path.splitext(os.path.basename(r["path"]))[0]}.flac"
            if not os.path.exists(out):
                wav, _ = librosa.load(io.BytesIO(r["audio"]["bytes"]), sr=16000)
                sf.write(out, wav, 16000)
            fo.write(json.dumps(dict(audio_filepath=out, duration=sf.info(out).duration,
                                     text=t2s.convert(r["sentence"]), lang="zh-TW"), ensure_ascii=False) + "\n")
