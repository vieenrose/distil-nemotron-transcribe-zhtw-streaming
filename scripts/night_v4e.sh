#!/bin/bash
# Wait for TTS jobs (PIDs given as args), teacher-filter synthetic data, then train v4e.
source "$(dirname "$0")/../env.sh"; cd "$(dirname "$0")/.."
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
for k in zh cs gcs; do python3 scripts/filter_tts.py label $k && python3 scripts/filter_tts.py filter $k; done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_student.py configs/v4e.json > $STORE/logs/train_v4e.log 2>&1
grep eval $STORE/logs/train_v4e.log
