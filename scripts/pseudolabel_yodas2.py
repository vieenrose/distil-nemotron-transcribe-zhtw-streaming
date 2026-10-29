"""Pseudo-label TW-selected yodas2 zh000 videos with the teacher (NeMo-Speech.cpp, GPU1).

Per audio shard: extract selected videos ($STORE/yodas2_select.jsonl), merge consecutive
subtitle segments into 4-20 s chunks (gap < 1 s), write 16 kHz FLAC under
$STORE/audio/yodas2/<shard>/, run the teacher, write $STORE/teacher_labels/yodas2/<shard>.jsonl
(teacher text + subtitle text). Skips shards already done or not yet downloaded.
Usage: python3 scripts/pseudolabel_yodas2.py
"""
import io, json, os, subprocess, tarfile, time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

STORE = os.environ["STORE"]
RAW = f"{STORE}/raw/yodas2_zh000/audio"
META = f"{STORE}/raw/yodas2_meta/data/zh000/text"
BIN = f"{STORE}/NeMo-Speech.cpp/build/cuda-server/bin/nemo-speech"
MODEL = f"{STORE}/teacher_gguf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-NVFP4.gguf"
PROMPT = ("请将这段语音逐字转写成中文文字。说话的语言是台湾的普通话，可能夹杂英文单词；"
          "英文单词保留原文。不要翻译。只输出转写内容。")
MIN_S, MAX_S, GAP_S = 4.0, 20.0, 1.0


def chunks(segs):
    """segs: {key: text}, key = <vid>-<idx>-<start_cs>-<end_cs> (10 ms units)."""
    items = []
    for k, t in segs.items():
        p = k.rsplit("-", 3)
        items.append((int(p[2]) / 100, int(p[3]) / 100, t))
    items.sort()
    out, cur = [], None
    for s, e, t in items:
        if cur and s - cur[1] < GAP_S and e - cur[0] <= MAX_S:
            cur = [cur[0], e, cur[2] + " " + t]
        else:
            if cur and cur[1] - cur[0] >= MIN_S:
                out.append(cur)
            cur = [s, e, t]
    if cur and cur[1] - cur[0] >= MIN_S:
        out.append(cur)
    return out


def run_shard(shard, vids):
    out_jsonl = Path(f"{STORE}/teacher_labels/yodas2/{shard}.jsonl")
    tar_path = Path(f"{RAW}/{shard}.tar.gz")
    nxt = Path(f"{RAW}/{int(shard) + 1:08d}.tar.gz")
    if out_jsonl.exists() or not tar_path.exists() or (int(shard) < 41 and not nxt.exists()):
        return  # done, or still downloading
    subs = {v["audio_id"]: v["text"] for v in json.load(open(f"{META}/{shard}.json"))}
    adir = Path(f"{STORE}/audio/yodas2/{shard}"); adir.mkdir(parents=True, exist_ok=True)
    hdir = Path(f"{STORE}/tmp/hyp_y{shard}"); hdir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with tarfile.open(tar_path) as tf:
        for m in tf:
            vid = Path(m.name).stem
            if vid not in vids or vid not in subs:
                continue
            wav, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            if sr != 16000:
                g = np.gcd(sr, 16000)
                wav = resample_poly(wav, 16000 // g, sr // g).astype("float32")
            for i, (s, e, t) in enumerate(chunks(subs[vid])):
                cid = f"{vid}_{i:04d}"
                seg = wav[int(s * 16000):int(e * 16000)]
                if len(seg) < MIN_S * 16000:
                    continue
                sf.write(adir / f"{cid}.flac", seg, 16000)
                rows.append(dict(id=cid, duration=len(seg) / 16000, human=t))
    t1 = time.time()
    if rows:
        subprocess.run([BIN, "transcribe", str(adir), "--model", MODEL, "--instruction", PROMPT,
                        "--output-dir", str(hdir), "--force"], check=True, stdout=subprocess.DEVNULL,
                       stderr=open(f"{STORE}/logs/ply_{shard}.err", "w"))
    t2 = time.time()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".part")
    with open(tmp, "w") as fo:
        for r in rows:
            h = hdir / f"{r['id']}.txt"
            fo.write(json.dumps(dict(audio_filepath=str(adir / f"{r['id']}.flac"), duration=r["duration"],
                                     teacher=h.read_text().strip() if h.exists() else None,
                                     human=r["human"], segment_id=r["id"]), ensure_ascii=False) + "\n")
    tmp.rename(out_jsonl)
    subprocess.run(["rm", "-rf", str(hdir)])
    h = sum(r["duration"] for r in rows) / 3600
    print(f"{shard}: {len(rows)} chunks {h:.2f} h | prep {t1-t0:.0f}s teacher {t2-t1:.0f}s "
          f"| RTFx {h*3600/max(t2-t1,1):.0f}", flush=True)


if __name__ == "__main__":
    sel = [json.loads(l) for l in open(f"{STORE}/yodas2_select.jsonl")]
    by_shard = {}
    for s in sel:
        by_shard.setdefault(s["shard"], set()).add(s["audio_id"])
    while True:
        pending = [s for s in sorted(by_shard) if not Path(f"{STORE}/teacher_labels/yodas2/{s}.jsonl").exists()]
        if not pending:
            break
        before = len(pending)
        for s in pending:
            run_shard(s, by_shard[s])
        if len([s for s in by_shard if not Path(f"{STORE}/teacher_labels/yodas2/{s}.jsonl").exists()]) == before:
            time.sleep(120)  # waiting for downloads
