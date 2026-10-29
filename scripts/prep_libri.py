"""LibriSpeech train-clean-100 (CC BY 4.0) -> manifest for English replay ($STORE/manifests/libri100.jsonl).
FLAC is already 16 kHz; text lowercased to sentence case is left as-is (uppercase in source)."""
import glob, json, os
import soundfile as sf

STORE = os.environ["STORE"]
root = f"{STORE}/raw/librispeech/LibriSpeech/train-clean-100"
with open(f"{STORE}/manifests/libri100.jsonl", "w") as fo:
    for tr in glob.glob(f"{root}/*/*/*.trans.txt"):
        d = os.path.dirname(tr)
        for l in open(tr):
            uid, text = l.strip().split(" ", 1)
            f = f"{d}/{uid}.flac"
            fo.write(json.dumps(dict(audio_filepath=f, duration=sf.info(f).duration,
                                     text=text.lower().capitalize(), lang="en-US")) + "\n")
