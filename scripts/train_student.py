"""Sequence-level distillation of the streaming 0.6B RNNT on teacher pseudo-labels (HF, single GPU).

- Starts from the NVIDIA base (commercial-clean). The zh-TW prompt slot is untrained in the base,
  so its prompt_projector column is initialized from zh-CN.
- Streaming-consistent: each step samples a right context from the model's supported lookaheads,
  the encoder uses the chunked-limited attention mask.
- Joint + RNNT loss are checkpointed per utterance to bound the B x T x U x V memory.
- Mix: manifests with weights, duration-bucketed batches.

Usage: python3 scripts/train_student.py config.json   (see configs/)
"""
import json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForRNNT, AutoProcessor

sys.path.insert(0, os.path.dirname(__file__))
from score import norm
import jiwer

cfg = json.load(open(sys.argv[1]))
STORE = os.environ["STORE"]
OUT = Path(cfg["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.get("seed", 0)); random.seed(cfg.get("seed", 0))
dev = cfg.get("device", "cuda")

proc = AutoProcessor.from_pretrained(cfg["init"])
model = AutoModelForRNNT.from_pretrained(cfg["init"], dtype=torch.float32).to(dev)
PROMPTS = proc.prompt_dictionary
HID = model.config.encoder_config.hidden_size
BLANK = model.config.blank_token_id
if cfg.get("init_prompt_from"):
    with torch.no_grad():
        w = model.prompt_projector.linear_1.weight
        for dst, src in cfg["init_prompt_from"].items():
            w[:, HID + PROMPTS[dst]] = w[:, HID + PROMPTS[src]]
model.train()
if cfg.get("freeze_encoder_steps"):
    for p in model.encoder.parameters():
        p.requires_grad = False

# ---------------- data
def load_manifest(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if cfg.get("min_dur", 1.0) <= r["duration"] <= cfg.get("max_dur", 20.0)]

sources = [(load_manifest(m["path"]), m["weight"]) for m in cfg["train"]]
for (rows, w), m in zip(sources, cfg["train"]):
    print(f"{m['path']}: {len(rows)} utts {sum(r['duration'] for r in rows)/3600:.1f} h weight {w}", flush=True)
tok = proc.tokenizer
LMAP = lambda l: cfg.get("lang_map", {}).get(l, cfg.get("force_lang") or l)
UNK = tok.unk_token_id


def encode(text):
    ids = tok(text, add_special_tokens=False).input_ids
    return ids if ids and (UNK is None or ids.count(UNK) <= 0.05 * len(ids)) else None


def sample_batch():
    """Draw utterances by source weight until max_batch_sec of audio."""
    weights = [w for _, w in sources]
    batch, total = [], 0.0
    while True:
        rows, _ = random.choices(sources, weights=weights)[0]
        r = random.choice(rows)
        if total + r["duration"] > cfg["max_batch_sec"] and batch:
            return batch
        ids = encode(r["text"])
        if ids is None:
            continue
        wav, sr = sf.read(r["audio_filepath"], dtype="float32")
        if cfg.get("speed_perturb") and random.random() < 0.5:
            f = random.choice([0.9, 1.1])
            wav = torchaudio.functional.resample(torch.from_numpy(wav), int(16000 * f), 16000).numpy()
        batch.append((wav, ids, LMAP(r.get("lang", "zh-TW"))))
        total += len(wav) / 16000


def to_inputs(batch):
    wavs, ids, langs = zip(*batch)
    x = proc(list(wavs), sampling_rate=16000, language=list(langs))
    U = max(len(i) for i in ids)
    labels = torch.full((len(ids), U), 0, dtype=torch.long)
    for k, i in enumerate(ids):
        labels[k, :len(i)] = torch.tensor(i)
    dec_in = torch.cat([torch.full((len(ids), 1), BLANK), labels], 1)
    return x, labels, torch.tensor([len(i) for i in ids]), dec_in


def utt_loss(enc, dec, lab, tl, el):
    with torch.autocast(dev, dtype=torch.bfloat16):
        logits = model.joint(encoder_hidden_states=enc[None, :, None, :], decoder_hidden_states=dec[None, None, :, :])
    return torchaudio.functional.rnnt_loss(logits.float(), lab[None].int(), el[None].int(), tl[None].int(),
                                           blank=BLANK, reduction="sum")


def step_loss(batch, lookahead):
    x, labels, tlen, dec_in = to_inputs(batch)
    with torch.autocast(dev, dtype=torch.bfloat16):
        enc_out = model.get_audio_features(input_features=x["input_features"].to(dev),
                                           attention_mask=x["attention_mask"].to(dev),
                                           prompt_ids=x["prompt_ids"].to(dev), num_lookahead_tokens=lookahead)
        enc = enc_out.pooler_output
        dec = model.decoder(dec_in.to(dev))
    elen = model.encoder._get_subsampling_output_length(x["attention_mask"].sum(-1)).to(dev)
    labels, tlen = labels.to(dev), tlen.to(dev)
    total, ntok = 0.0, 0
    for k in range(enc.shape[0]):
        T, U = int(elen[k]), int(tlen[k])
        l = checkpoint(utt_loss, enc[k, :T], dec[k, :U + 1], labels[k, :U], tlen[k], elen[k], use_reentrant=False)
        if not torch.isfinite(l) or l.item() / U > cfg.get("max_tok_loss", 15.0):
            continue  # misaligned / bad pseudo-label: skip
        total, ntok = total + l, ntok + U
    return total / max(ntok, 1) if ntok else None


# ---------------- eval
EVALS = {name: [json.loads(l) for l in open(p)] for name, p in cfg["eval"].items()}


@torch.no_grad()
def evaluate(lang="zh-TW", lookahead=None):
    model.eval(); res = {}
    for name, rows in EVALS.items():
        refs, hyps = [], []
        for r in rows:
            wav, _ = sf.read(r["audio_filepath"], dtype="float32")
            x = proc(wav, sampling_rate=16000, language=LMAP(r.get("lang", lang))).to(dev)
            if lookahead is not None:
                x["num_lookahead_tokens"] = lookahead
            with torch.autocast(dev, dtype=torch.bfloat16):
                out = model.generate(**x, return_dict_in_generate=True)
            h = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            refs.append(norm(r["text"]) or "x"); hyps.append(norm(h))
        res[name] = round(100 * jiwer.cer(refs, hyps), 2)
    model.train(); return res


# ---------------- train
params = [p for p in model.parameters()]
opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-3), betas=(0.9, 0.98))
steps, warm, accum = cfg["steps"], cfg["warmup"], cfg.get("accum", 1)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * max(cfg.get("min_lr_ratio", 0.05),
                                                    0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
LOOKAHEADS = model.config.encoder_config.supported_num_lookahead_tokens
log = open(OUT / "log.jsonl", "a")
best = None
if cfg.get("eval_at_start", True):
    r = evaluate(); print("step 0", r, flush=True); log.write(json.dumps(dict(step=0, **r)) + "\n"); log.flush()
t0, run = time.time(), 0.0
for step in range(1, steps + 1):
    if cfg.get("freeze_encoder_steps") and step == cfg["freeze_encoder_steps"] + 1:
        for p in model.encoder.parameters():
            p.requires_grad = True
    for _ in range(accum):
        loss = step_loss(sample_batch(), random.choice(LOOKAHEADS))
        if loss is None:
            continue
        (loss / accum).backward(); run += loss.item() / accum
    torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % cfg.get("log_every", 50) == 0:
        print(f"step {step} loss {run / cfg.get('log_every', 50):.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"{(time.time() - t0) / step:.2f}s/step", flush=True); run = 0.0
    if step % cfg["eval_every"] == 0 or step == steps:
        r = evaluate(); score = sum(r.values())
        print(f"step {step} eval {r}", flush=True)
        log.write(json.dumps(dict(step=step, **r)) + "\n"); log.flush()
        if best is None or score < best:
            best = score; model.save_pretrained(OUT / "best"); proc.save_pretrained(OUT / "best")
model.save_pretrained(OUT / "last"); proc.save_pretrained(OUT / "last")
