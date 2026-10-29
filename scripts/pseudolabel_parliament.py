"""Pseudo-label OpenFormosa/parliament shards with the teacher (NeMo-Speech.cpp, GPU1).

Per shard: decode audio -> 16 kHz mono FLAC under $STORE/audio/parliament/<shard>/,
run teacher on the directory, write $STORE/teacher_labels/parliament/<shard>.jsonl
with teacher text + human text + metadata. Skips shards already done.
"""
import glob, io, json, os, subprocess, sys, time
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf

STORE = os.environ["STORE"]
SRC = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--OpenFormosa--parliament/snapshots/*/data/*.parquet")))
BIN = f"{STORE}/NeMo-Speech.cpp/build/cuda-server/bin/nemo-speech"
MODEL = f"{STORE}/teacher_gguf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-NVFP4.gguf"
PROMPT = ("请将这段语音逐字转写成中文文字。说话的语言是台湾的普通话，可能夹杂英文单词；"
          "英文单词保留原文。不要翻译。只输出转写内容。")
COLS = ["audio", "text", "duration", "segment_id", "session_date", "cer", "snr", "dnsmos_ovr"]

def decode(data, out):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", out],
                   input=data, check=True)

def run_shard(path):
    name = Path(path).stem
    out_jsonl = Path(f"{STORE}/teacher_labels/parliament/{name}.jsonl")
    if out_jsonl.exists():
        return
    adir = Path(f"{STORE}/audio/parliament/{name}"); adir.mkdir(parents=True, exist_ok=True)
    hdir = Path(f"{STORE}/tmp/hyp_{name}"); hdir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(path, columns=COLS).to_pylist()
    t0 = time.time()
    for r in rows:
        f = adir / f"{r['segment_id']}.flac"
        if not f.exists():
            decode(r["audio"]["bytes"], str(f))
    t1 = time.time()
    subprocess.run([BIN, "transcribe", str(adir), "--model", MODEL, "--instruction", PROMPT,
                    "--output-dir", str(hdir), "--force"],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=open(f"{STORE}/logs/pl_{name}.err", "w"))
    t2 = time.time()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".part")
    with open(tmp, "w") as fo:
        for r in rows:
            f = adir / f"{r['segment_id']}.flac"
            h = hdir / f"{r['segment_id']}.txt"
            fo.write(json.dumps({
                "audio_filepath": str(f), "duration": r["duration"],
                "teacher": h.read_text().strip() if h.exists() else None,
                "human": r["text"], "segment_id": r["segment_id"], "session_date": r["session_date"],
                "upstream_cer": r["cer"], "snr": r["snr"], "dnsmos_ovr": r["dnsmos_ovr"],
            }, ensure_ascii=False) + "\n")
    tmp.rename(out_jsonl)
    subprocess.run(["rm", "-rf", str(hdir)])
    audio_h = sum(r["duration"] for r in rows) / 3600
    print(f"{name}: {len(rows)} clips {audio_h:.2f} h | decode {t1-t0:.0f}s teacher {t2-t1:.0f}s "
          f"| RTFx {audio_h*3600/(t2-t1):.0f}", flush=True)

if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    for p in SRC[start:end]:
        run_shard(p)
