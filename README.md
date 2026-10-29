# Distilling Nemotron-3.5-Transcribe-30B-A4B into a streaming 0.6B ASR for zh-TW and zh-TW/en code-switching

Starting point: the previous iteration, `Luigi/nemotron-3.5-asr-streaming-0.6b-zhtw` (v3, CV zh-TW CER 9.81).
Goal: close the gap to X-ASR (6.66 CV / 6.13 Taiwan-Tongues) and make zh/en code-switching (CS) robust in streaming.

## Verified context (2026-10-29)
| Item | Status |
|---|---|
| Teacher | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`: 67.5 GB BF16, OpenMDW-1.1 license. Mandarin supported, CS explicitly supported. Offline model (SALM). |
| Teacher on a 5090 | The official card lists **NVFP4 only** on RTX 5090 (vLLM 0.32 / HF / GGML). The `nvidia/...-FP8` and `-NVFP4` repos return "not found" with our token (not public yet?). |
| Teacher actually used | `AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF`: AD-NVFP4 decoder (19.7 GB) + F16 encoder (1.7 GB), run with [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) (CUDA, sm_120). The `nativ-community/*` repos are MLX (Apple only) and unusable here. |
| Student | v3 above. 13,087-token tokenizer: 6.3% of Traditional characters fail a round-trip. Targets are **Simplified**, then OpenCC. |
| Hardware | **GPU1 only** (RTX 5090, 32 GB), 249 GB RAM, 32 cores. Storage: `/media/big-disk-2/distil-zhtw`. The previous iteration used 2 GPUs, so expect ~2× wall-clock time. |

## Lessons from v3 that shape this plan
1. **The "two teachers agree" filter discarded 41% of IVOD.** A single, much stronger teacher with native CS support lets us keep far more data, but a guard against LLM hallucination is still required.
2. **English and French drift**: keep multilingual replay and the repair phase, and add French to the replay.
3. **The tokenizer is the ceiling for Traditional output.** This is the key decision, handled in Phase 3.
4. **No CS metric exists yet.** The CS eval set must be built first.

## Phase 0: environment and baselines
- Teacher served by NeMo-Speech.cpp (GGUF AD-NVFP4). Fallback: quantize BF16 ourselves with ModelOpt NVFP4 (~70 GB RAM, feasible but slow).
- Validate the teacher on CV zh-TW and Taiwan-Tongues with the same OpenCC normalization as v3.
  - **Go/no-go**: the teacher should beat X-ASR (< 6.5 CER). Otherwise distillation adds nothing on pure zh; refocus it on CS and use X-ASR as a co-teacher for zh.
- **Build a zh-TW/en CS eval set**: 3–5 h of real Taiwanese content (tech podcasts, lectures, meetings), pre-transcribed by the teacher then **hand-corrected**. Candidate: `DynamicSuperb/Code-switchSpeechRecognition_NTUML2021`.
- Metrics:
  - MER (CER on zh characters, WER on en words);
  - English-word recall and precision;
  - zh-only CER;
  - per-chunk eval at 80, 160, 560 and 1120 ms.

## Phase 1: teacher pseudo-labeling (~1–2 weeks on GPU1)
- **Sources**:
  - IVOD 立法院 (`OpenFormosa/parliament`, see below);
  - CS-rich Taiwanese lectures (NTU ML 2021, see data plan), respecting licenses and ToS;
  - the rest of CV zh-TW;
  - Taiwan-Tongues train (zh-TW and en).

  Target: 1,500–2,500 h, at least 30% CS.
- Segments of 5–20 s. Teacher prompt: **P4** (Chinese instruction, see results), temperature 0, reasoning off.
- **Filters**:
  - abnormal characters/second, repeated n-grams, empty or translated output (English-only on zh audio);
  - mean teacher log-prob below a threshold;
  - **light** cross-check with v3 (not strict two-teacher): reject only if CER(teacher, v3) > 0.5, which signals a likely hallucination. Moderate disagreements are exactly the useful examples;
  - where human text exists (IVOD), agreement with it as an extra signal.
- Storage: teacher text as produced (Simplified) plus a `s2twp` Traditional copy for when the tokenizer changes.
- Synthetic CS (≤ 15%): LLM-generated zh-TW sentences with English insertions (tech terms, proper nouns), then multi-speaker TTS. Optional, only if real CS is insufficient.

## Phase 2: sequence-level distillation on top of v3 (~1 week on GPU1)
- Same NeMo recipe as v3: `speech_to_text_finetune.py`, streaming prompt config, AdamW 1e-5, bf16. On one GPU, double gradient accumulation to keep the effective batch.
- Mix:

  | Share | Data |
  |---|---|
  | 40% | zh-TW pseudo-labels |
  | 30% | CS |
  | 15% | English (LibriSpeech/GigaSpeech, TW-accented English) |
  | 10% | multilingual replay (fr/es/hi first, given the regressions) |
  | 5% | human CV zh-TW (anchor) |
- Language prompt: compare `zh-TW` vs `auto` on CS. Keep `unified` prompt mode.
- Weight the loss by teacher confidence.
- Repair phase (lr 4e-6) if English or French drifts, as in v3.

## Phase 3: Traditional-character tokenizer (decided after Phase 2)
- **Option A** (default, cheap): keep Simplified + OpenCC, but switch the conversion to **`s2twp`** (phrase-level, Taiwan vocabulary) instead of `s2t`, and measure ambiguity errors (发→發/髮, 后→後/后, 台→臺/台) on the eval.
- **Option B**: extend the vocabulary with the missing Traditional characters. The RNNT joint/decoder grows; new rows are initialized from the corresponding Simplified character, then fine-tuned on Traditional teacher targets. Removes the OpenCC step and its ambiguities, but every other language must be re-checked.
- Criterion: move to B only if OpenCC-induced errors exceed 1 absolute CER on the zh-TW eval.

## Phase 4 (optional): representation distillation
- Add an L2/cosine loss between the student encoder and the teacher's ASR encoder (1,280-dim Transformer) through a linear projection, λ≈0.1. Downsampled fp16 features precomputed on ~300 h (~100 GB).
- Caveat: the student is causal/streaming while the teacher encoder sees full context. Target top layers only and measure the gain before generalizing.

## Phase 5: iterations
- Hard mining: where v4 and the teacher diverge most, add more data of that kind and relabel.
- Context biasing (NeMo GPU-PB) for English tech vocabulary at deployment.

## Target scoreboard (v4)
| Eval | v3 | v4 target |
|---|---|---|
| CV zh-TW CER | 9.81 | **< 7.5** |
| Taiwan-Tongues CER | 14.85 | **< 10** |
| CS zh-TW/en MER (new set) | to measure | −30% relative vs v3 |
| LibriSpeech WER | 2.79 | ≤ 2.8 |
| FLEURS fr WER | 16.15 | ≤ 15.5 |

## Environment
```bash
source env.sh   # CUDA_VISIBLE_DEVICES=1, STORE, HF_HOME, TMPDIR
```
Scripts: `scripts/prep_eval.py` (fixed eval subsets), `scripts/score.py` (CER, t2s-normalized, punctuation stripped), `scripts/run_student.py` (HF RNNT inference), `scripts/pseudolabel_parliament.py` (teacher pseudo-labeling per shard).

## Phase 0 results (2026-10-29): teacher GGUF AD-NVFP4 via NeMo-Speech.cpp, GPU1
Same clips (CER, t2s normalization, punctuation removed):

| System | CV zh-TW (120) | Taiwan-Tongues (100) |
|---|---|---|
| v3 (current student) | **14.25** | 14.85 |
| Teacher, `-l zh-TW` | 26.84–28.62 | **11.90** |
| Teacher, "Mandarin TW" English prompt (P2) | 20.78 | 14.11 |
| Teacher, ASCEND CS prompt (P1) | 40.26 | 20.37 |

(v3's 14.85 on Taiwan-Tongues matches its model card exactly; the CV gap vs the published 9.81 is a different 120-clip subset.)

- The teacher **always outputs Simplified**, which matches the student tokenizer.
- On short read CV clips the teacher **hallucinates or switches language**: Cantonese/Hokkien romanization, English, even Bengali.
- With an agreement filter CER(v3, teacher) ≤ 0.5, 82–91% of clips are kept and the teacher is **ahead** of v3: 5.67 vs 6.66 on CV, 8.28 vs 12.35 on TT.
- **Decision**: GO for distillation, with a mandatory v3/teacher agreement filter at 0.5. Gains are largest on natural speech (TT/IVOD/lectures) and small on short read speech (CV).
- These scores predate the P4 prompt; re-evaluation with P4 is pending.

## IVOD pseudo-labeling (running since 2026-10-29)
- Source: `OpenFormosa/parliament`, 308,521 clips, **1,299 h**, 10–20 s, sessions 2020-09 → 2026-04. **License CC BY-NC 4.0** (non-commercial).
- Human text is edited and loosely aligned at clip boundaries (median CER 0.22 vs upstream ASR). It is used as a **filtering signal**, not as a target.
- **Unintended translation**: with `-l zh-TW` or the official prompts, the teacher translates 20–44% of IVOD clips into English. Test on 200 clips:

  | Setting | English outputs | Median CER vs human | Share ≤ 0.4 |
  |---|---|---|---|
  | none | 88/200 | 0.47 | 44% |
  | `-l zh` | 54 | 0.36 | 57% |
  | P2 (English, "Mandarin TW") | 39 | 0.35 | 59% |
  | P3 (official verbatim) | 74 | 0.42 | 47% |
  | **P4 (Chinese prompt)** | **0** | **0.29** | **81%** |

  P4: `请将这段语音逐字转写成中文文字。说话的语言是台湾的普通话，可能夹杂英文单词；英文单词保留原文。不要翻译。只输出转写内容。`
- Throughput on GPU1: RTFx ≈ 120–190, i.e. ~10 h for 1,299 h (audio decoding included).
- Next: re-evaluate CV/TT with P4, then apply the combined filter (agreement with human text and with v3).

## Data plan (inventory 2026-10-29)
Downloading now (~25 GB): `ky552/ML2021_ASR_ST`, `andybi7676/ntuml2021_long`, `voidful/hung-yi_lee` (NTU ML lectures, Taiwan Mandarin with dense English terms), `DynamicSuperb/Code-switchSpeechRecognition_NTUML2021` (CS eval), `adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en` (TW-accented English replay).

Next candidates: `espnet/yodas2` zh000 (60 GB, CC-BY-3.0, mostly zh-CN, needs TW filtering), `georgechang8/ASCEND_CLEAN` (CS eval, HK accent), GigaSpeech M + LibriSpeech-360 (English replay), `formospeech/matbn` (TW broadcast, gated).

Skipped: WorldSpeech zh_tw and `lianghsun/ly-ivod-asr-zhtw` (likely duplicates of the parliament audio), Emilia ZH (size, zh-CN accent).

## Licensing policy (target: v4 free for commercial use)
Only data whose license allows commercial training is used for training. NC/unclear sources may be used for **evaluation only**.

| Source | License | Use |
|---|---|---|
| Common Voice zh-TW | CC0 | train + eval |
| LibriSpeech | CC BY 4.0 | train (attribution) |
| yodas2 | CC BY 3.0 | train (attribution) |
| ASCEND_CLEAN | CC BY-SA 4.0 | eval only (share-alike risk on weights) |
| `OpenFormosa/parliament` (IVOD) | **CC BY-NC 4.0** | **excluded from training**; labeling stopped after 4 shards |
| NTU ML 2021 lectures (`ky552/ML2021_ASR_ST` MIT tag, `voidful/hung-yi_lee` other) | audio rights unclear (YouTube lectures) | eval only until the lecturer's permission is confirmed |
| Taiwan-Tongues (`adi-gov-tw/*`) | "other", follows upstream project terms | to verify before training |
| GigaSpeech | metadata Apache-2.0, audio terms non-commercial | excluded |
| MATBN | gated, research license | excluded |
| Teacher outputs (Nemotron-3.5-Transcribe) | OpenMDW-1.1 | pseudo-labels usable |

Open points:
- **v3 was itself fine-tuned on IVOD** (NC). For a clean commercial v4, restart from the NVIDIA base streaming 0.6B (check its license) instead of v3, or get legal clearance.
- Main data gap: commercially usable zh-TW natural speech and CS audio. Candidates: yodas2 (CC BY, filter TW accent / CS), Common Voice, synthetic CS via commercially licensed TTS.
