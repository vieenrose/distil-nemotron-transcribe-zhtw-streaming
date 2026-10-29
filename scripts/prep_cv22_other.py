"""CV22 zh-TW 'other' (unvalidated, CC0) -> 16 kHz FLAC, then teacher-verify against the prompt sentence.

Drops speakers of the CV16 test shard used for CV120, and clips already in the CV16 train manifest.
Step 1 (decode): python3 scripts/prep_cv22_other.py decode
Step 2 (teacher, GPU1): python3 scripts/prep_cv22_other.py label
Step 3 (filter): python3 scripts/prep_cv22_other.py filter   -> $STORE/manifests/cv22_other.jsonl
Kept clip target = human sentence (t2s) when CER(teacher, sentence) <= 0.1.
"""
import csv, json, os, subprocess, sys, tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pyarrow.parquet as pq
import jiwer
from opencc import OpenCC

sys.path.insert(0, os.path.dirname(__file__))
from score import norm

STORE = os.environ["STORE"]
RAW = Path(f"{STORE}/raw/cv22_zhtw"); ADIR = Path(f"{STORE}/audio/cv22_other")
LAB = Path(f"{STORE}/teacher_labels/cv22_other"); HYP = LAB / "hyp"
BIN = f"{STORE}/NeMo-Speech.cpp/build/cuda-server/bin/nemo-speech"
MODEL = f"{STORE}/teacher_gguf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-NVFP4.gguf"
PROMPT = ("请将这段语音逐字转写成中文文字。说话的语言是台湾的普通话，可能夹杂英文单词；"
          "英文单词保留原文。不要翻译。只输出转写内容。")


def rows():
    test_spk = set(pq.read_table(f"{STORE}/eval/raw/cv/data/test-00000-of-00003.parquet",
                                 columns=["client_id"]).column(0).to_pylist())
    seen = {Path(json.loads(l)["audio_filepath"]).stem for l in open(f"{STORE}/manifests/cv_zhtw.jsonl")}
    out = []
    for r in csv.DictReader(open(RAW / "other.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r["client_id"] in test_spk or Path(r["path"]).stem in seen:
            continue
        out.append(r)
    return out


def to_flac(args):
    src, dst = args
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst], check=True)
    os.remove(src)


def decode():
    keep = {r["path"] for r in rows()}
    tmp = Path(f"{STORE}/tmp/cv22"); tmp.mkdir(parents=True, exist_ok=True); ADIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in sorted(RAW.glob("*.tar")):
        with tarfile.open(t) as tf:
            for m in tf:
                name = os.path.basename(m.name)
                if name in keep and not (ADIR / (Path(name).stem + ".flac")).exists():
                    tf.extract(m, tmp, filter="data")
                    jobs.append((str(tmp / m.name), str(ADIR / (Path(name).stem + ".flac"))))
    with ProcessPoolExecutor(24) as ex:
        list(ex.map(to_flac, jobs, chunksize=64))
    print(len(keep), "kept clips;", len(jobs), "decoded")


def label():
    HYP.mkdir(parents=True, exist_ok=True)
    subprocess.run([BIN, "transcribe", str(ADIR), "--model", MODEL, "--instruction", PROMPT,
                    "--output-dir", str(HYP)], check=True, stdout=subprocess.DEVNULL,
                   stderr=open(f"{STORE}/logs/pl_cv22.err", "w"))


def filt():
    import soundfile as sf
    t2s = OpenCC("t2s"); n = k = 0; h = 0.0
    with open(f"{STORE}/manifests/cv22_other.jsonl", "w") as fo:
        for r in rows():
            stem = Path(r["path"]).stem; hp = HYP / f"{stem}.txt"; ap = ADIR / f"{stem}.flac"
            if not hp.exists() or not ap.exists():
                continue
            n += 1
            ref = norm(r["sentence"])
            if not ref or jiwer.cer(ref, norm(hp.read_text()) or "x") > 0.1:
                continue
            d = sf.info(str(ap)).duration; k += 1; h += d
            fo.write(json.dumps(dict(audio_filepath=str(ap), duration=d, text=t2s.convert(r["sentence"]),
                                     lang="zh-TW"), ensure_ascii=False) + "\n")
    print(f"labeled {n}, kept {k} ({h / 3600:.1f} h)")


if __name__ == "__main__":
    {"decode": decode, "label": label, "filter": filt}[sys.argv[1]]()
