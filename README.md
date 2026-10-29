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

## Phase 4b: sequence-level RL / risk minimization (after Phase 2–3 converge)
Cross-entropy on pseudo-labels optimizes token likelihood. RL instead optimizes what we measure (CER/MER, English-term accuracy, latency) on the student's **own** hypotheses, so exposure bias and CS errors are trained directly.

**Method: MWER / minimum Bayes risk for RNNT** (GRPO-style group sampling):
- For each utterance, sample K=4–8 hypotheses with streaming beam search or sampling (T≈1.0) from the student.
- Reward per hypothesis, computed against the teacher pseudo-label (or human text where commercially usable):
  - `r = -MER` (zh CER + en WER);
  - `+ α·` English-word F1 (CS, rare tech terms);
  - `- β·` script/language errors (wrong script, translation, hallucinated repetitions);
  - `- γ·` emission latency (mean token delay vs forced alignment, FastEmit-style), so streaming does not get slower.
- Advantage = r − group mean (no critic). Loss = −Σ A·log P_RNNT(hyp), with the RNNT log-likelihood of each hypothesis computed exactly.
- Keep the RNNT loss on the reference (λ≈0.5–1) plus a KL-to-reference-model penalty to prevent drift. Low lr (1e-6 – 3e-6), short (~1 epoch on 100–300 h).

**Data**: focus on hard cases: CS clips, clips where the v4 student and teacher disagree (CER 0.1–0.5), and noisy/long natural speech. Commercial-license data only (see policy).

**Teacher as a reward model (optional)**: rescore the student's N-best with the teacher LLM's log-likelihood (the teacher's decoder conditioned on audio) as a soft reward, instead of only exact match with one pseudo-label. This avoids over-fitting the teacher's single hypothesis. It is costly (one teacher forward per hypothesis), so use it only on a subset.

**Guards**: evaluate every N steps on CV/TT/CS/Libri/FLEURS-fr at 160 and 560 ms chunks; stop on any regression > 0.3 abs. Watch for reward hacking: empty or short outputs (length-normalized reward, deletion penalty).

**Expected gain**: MWER-type training typically gives a few percent relative WER on top of CE. The larger value here is CS/English-term recall and robustness to teacher errors. Go/no-go: ≥ 3% relative MER on the CS eval with no regression elsewhere.

**Implementation**: NeMo has no ready RNNT-MWER recipe, so we write a custom Lightning step. Sampling uses NeMo greedy/beam batch decoding (`return_hypotheses`); the hypothesis log-prob is `-rnnt_loss(hyp_as_target)`. Cost is ~K× decoding plus K RNNT losses per batch, so plan 2–4 days on GPU1.

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

## IVOD pseudo-labeling (stopped 2026-10-29: CC BY-NC, excluded; kept as a record of the prompt study)
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
Superseded by the licensing policy below: NTU ML lectures and Taiwan-Tongues were downloaded then excluded.

Next candidates: `espnet/yodas2` zh000 (60 GB, CC-BY-3.0, mostly zh-CN, needs TW filtering), `georgechang8/ASCEND_CLEAN` (CS eval, HK accent), GigaSpeech M + LibriSpeech-360 (English replay), `formospeech/matbn` (TW broadcast, gated).

Skipped: WorldSpeech zh_tw and `lianghsun/ly-ivod-asr-zhtw` (likely duplicates of the parliament audio), Emilia ZH (size, zh-CN accent).

## Licensing policy (target: v4 free for commercial use)
**Rule: only data whose license explicitly allows commercial use is used for training or pseudo-labeling.** Anything NC, gated research-only, "other" without clear terms, or with unclear audio rights is excluded. Already-downloaded excluded data was moved to `$STORE/excluded/` and is not used by any script.

| Source | License | Status |
|---|---|---|
| Common Voice zh-TW / en | CC0 | ✅ train |
| LibriSpeech | CC BY 4.0 | ✅ train (attribution) |
| yodas2 | CC BY 3.0 | ✅ train (attribution) |
| Teacher outputs (Nemotron-3.5-Transcribe) | OpenMDW-1.1 | ✅ pseudo-labels |
| ASCEND_CLEAN | CC BY-SA 4.0 | ⚠️ eval only (not in training, to avoid share-alike on weights) |
| Taiwan-Tongues (`adi-gov-tw/*`) | "other", unclear | ❌ excluded from training; TT100 kept only as a local benchmark for comparison with v3 |
| `OpenFormosa/parliament` (IVOD) | CC BY-NC 4.0 | ❌ excluded |
| NTU ML 2021 lectures (`ky552/ML2021_ASR_ST`, `andybi7676/ntuml2021_long`, `voidful/hung-yi_lee`, `DynamicSuperb/...NTUML2021`) | unclear audio rights | ❌ excluded |
| GigaSpeech | non-commercial audio terms | ❌ excluded |
| MATBN | research license | ❌ excluded |

Consequences:
- **v3 was fine-tuned on IVOD (NC)**, so v4 restarts from the NVIDIA base streaming 0.6B (license to confirm), not from v3. v3 stays as a comparison baseline and filter signal only.
- New data sources: Common Voice zh-TW/en, yodas2 zh/en (filtered for TW accent and CS, pseudo-labeled by the teacher), LibriSpeech, and synthetic CS from a commercially licensed TTS.
- A commercially clean CS eval set must be built (e.g. from yodas2 CS clips, hand-corrected).

## Commercially usable data: status (2026-10-29)
- **yodas2 zh000** (CC BY 3.0): 418 h in total. `scripts/select_yodas2.py` keeps videos with mostly Traditional subtitles, no Simplified-only characters and no Cantonese markers: **193 h of likely TW Mandarin** (630 of 2,196 videos). Only ~2 h are CS-rich by subtitle English density. YouTube subtitles often drop English, so teacher relabeling (P4) should surface more CS. Audio download is running (60 GB).
- **Hub search** for permissive zh/en CS speech found nothing usable: SEAME, ASCEND, CS-Dialogue and NTU ML are all NC, SA or unclear.
- **CS therefore comes mainly from synthesis**:
  - text: zh-TW sentences with English terms, written by an LLM whose terms allow it, plus CS lines from teacher-labeled yodas2;
  - audio: an Apache-2.0 TTS (e.g. CosyVoice2; check that the weight license allows commercial use) with reference voices from CC0 Common Voice zh-TW speakers;
  - cap at 15–30% of the mix, and validate on a real CS eval set (hand-corrected yodas2 CS clips).
- English: LibriSpeech (CC BY 4.0), Common Voice en (CC0), yodas2 en000 (CC BY 3.0).
