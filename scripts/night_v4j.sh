#!/bin/bash
# More TTS zh (seed 2), teacher-filter, then train v4j from avg(v4g,v4h).
source "$(dirname "$0")/../env.sh"; cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=1 $STORE/venv_tts/bin/python scripts/synth_tts.py zh 12000 2 > $STORE/logs/synth_zh2.log 2>&1
python3 scripts/filter_tts.py label zh; python3 scripts/filter_tts.py filter zh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_student.py configs/v4j.json > $STORE/logs/train_v4j.log 2>&1
grep eval $STORE/logs/train_v4j.log
