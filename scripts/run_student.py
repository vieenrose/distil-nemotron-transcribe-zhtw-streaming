"""Transcribe eval manifest with a HF RNNT checkpoint; writes NNNN.txt into outdir."""
import json, os, sys
from pathlib import Path
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.audio_utils import load_audio

model_id, manifest, outdir, lang = sys.argv[1:5]
proc = AutoProcessor.from_pretrained(model_id)
model = AutoModelForRNNT.from_pretrained(model_id, device_map="cuda").eval()
os.makedirs(outdir, exist_ok=True)
sr = proc.feature_extractor.sampling_rate
for line in open(manifest):
    r = json.loads(line)
    audio = load_audio(r["audio_filepath"], sampling_rate=sr)
    inputs = proc(audio, sampling_rate=sr, language=lang).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)
    text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
    Path(outdir, Path(r["audio_filepath"]).stem + ".txt").write_text(text)
