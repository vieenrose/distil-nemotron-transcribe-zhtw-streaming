# Plan v2 : distiller Nemotron-3.5-Transcribe-30B-A4B dans le modèle streaming 0.6B, pour le zh-TW et le code-switching zh-TW/en

Point de départ : le modèle de l'itération précédente, `Luigi/nemotron-3.5-asr-streaming-0.6b-zhtw` (v3, CV zh-TW CER 9.81).
Objectif : combler l'écart avec X-ASR (6.66 CV / 6.13 Taiwan-Tongues) et rendre le code-switching zh/en (CS) robuste en streaming.

## Contexte vérifié (2026-10-29)
| Élément | État |
|---|---|
| Teacher | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` : BF16 de 67,5 Go, licence OpenMDW-1.1. Mandarin supporté, CS explicitement supporté. Modèle offline (SALM). |
| Teacher sur 5090 | La carte officielle indique **NVFP4 uniquement** sur RTX 5090 (vLLM 0.32 / HF / GGML). Les repos `nvidia/...-FP8` et `-NVFP4` répondent « not found » avec notre token (pas encore publics ?). |
| Alternatives teacher | `AtomicChat/...-GGUF` : variante AD-NVFP4 de 19,7 Go plus l'encodeur F16 de 1,7 Go, à servir via `~/llama.cpp` (mtmd). Les repos `nativ-community/*` sont en MLX, **inutilisables** (Apple). |
| Élève | Le v3 ci-dessus. Tokenizer de 13 087 tokens : 6,3 % des caractères traditionnels ne passent pas l'aller-retour. Cibles en **simplifié**, puis OpenCC `s2t`. |
| Matériel | **GPU1 seul** (5090, 32 Go), 249 Go de RAM, 32 cœurs. Stockage : `/media/big-disk-2/distil-zhtw` (422 Go libres). L'itération précédente utilisait 2 GPU, il faut donc prévoir environ 2× plus de temps mur. |
| Logiciels | torch 2.10/cu128 dans le Python système. `~/NeMo` (fb20f06, 2026-07) est présent mais pas installé. vLLM absent. Prévoir un venv `uv` dédié sous `$STORE`. |

## Leçons de v3 qui orientent ce plan
1. **Le filtre « deux teachers d'accord » a jeté 41 % de l'IVOD.** Un teacher unique bien plus fort, qui gère nativement le CS, permet de garder beaucoup plus de données. Il faut tout de même un garde-fou contre les hallucinations du LLM.
2. **Dérive de l'anglais et du français** : on conserve le replay multilingue et la phase de réparation, en ajoutant le français au replay.
3. **Le tokenizer est le plafond pour le traditionnel.** C'est la décision clé, traitée en Phase 3.
4. **Aucune métrique CS n'existe encore.** Il faut construire le jeu d'éval en premier.

## Phase 0 : environnement et baselines (1 à 2 j)
- venv `uv` : `nemo_toolkit[asr]` depuis `~/NeMo` (à mettre à jour) et vLLM 0.32. Tester d'abord le teacher via vLLM NVFP4. Si les poids NVFP4 restent inaccessibles, passer au GGUF AD-NVFP4 avec llama.cpp. Dernier recours : quantifier nous-mêmes le BF16 avec ModelOpt NVFP4. Ça demande environ 70 Go de RAM et c'est faisable, mais lent.
- Valider le teacher sur CV zh-TW et Taiwan-Tongues. Les scripts sont à reprendre de `~/zhtw-bench`, avec la même normalisation OpenCC que v3.
  - **Critère go/no-go** : le teacher doit faire mieux que X-ASR (moins de 6,5 CER). Sinon, la distillation n'apporte rien sur le zh pur. On la recentre alors sur le CS, et X-ASR devient co-teacher pour le zh.
- **Construire le jeu d'éval CS zh-TW/en** : 3 à 5 h de vrai contenu taïwanais (podcasts tech, cours, réunions), pré-transcrites par le teacher puis **corrigées à la main**. Sans ce jeu, rien n'est mesurable.
- Métriques :
  - MER (CER sur les caractères zh, WER sur les mots en) ;
  - rappel et précision des mots anglais ;
  - CER zh seul ;
  - éval par chunk 80, 160, 560 et 1120 ms.

## Phase 1 : pseudo-labeling par le teacher (~1 à 2 sem. GPU1)
- **Sources** :
  - IVOD 立法院 complet, sans le filtre 2-teachers (CC-BY-4.0) ;
  - podcasts et YouTube taïwanais riches en CS, dans le respect des licences et des ToS ;
  - le reste de CV zh-TW ;
  - Taiwan-Tongues train.

  Cible : 1 500 à 2 500 h, dont au moins 30 % de CS.
- Découpage VAD en 5 à 20 s. Prompt teacher :
  `Audio transcription. Language: as spoken. Wording: verbatim... Formatting: punctuation and capitalization. Output: transcript only.`, à température 0, reasoning off.
- **Filtres** :
  - densité caractères/seconde hors norme, n-grammes répétés, sortie vide ou traduite (anglais seul sur de l'audio zh) ;
  - log-prob moyenne du teacher sous un seuil ;
  - contrôle croisé **léger** avec v3 (et non le 2-teachers strict) : on ne rejette que si CER(teacher, v3) dépasse 50 %, ce qui signale une hallucination probable. Les désaccords modérés sont justement les exemples utiles.
- Stockage : texte du teacher en **traditionnel**, et texte converti `t2s` pour l'entraînement tant que le tokenizer reste inchangé.
- Synthétique CS (au plus 15 %) : phrases zh-TW avec insertion d'anglais générées par LLM (termes tech, noms propres), puis TTS multi-locuteurs. Le TTS est optionnel, à ajouter seulement si le CS réel ne suffit pas.

## Phase 2 : distillation en séquence sur v3 (~1 sem. GPU1)
- Même recette NeMo que v3 : `speech_to_text_finetune.py`, la config streaming prompt, AdamW à 1e-5, bf16. Sur un seul GPU, doubler l'accumulation de gradients pour garder le batch effectif.
- Mélange :

  | Part | Données |
  |---|---|
  | 40 % | zh-TW pseudo-labels |
  | 30 % | CS |
  | 15 % | anglais LibriSpeech/GigaSpeech |
  | 10 % | replay multilingue (fr/es/hi en priorité, vu les régressions) |
  | 5 % | données humaines CV zh-TW (ancrage) |
- Prompt de langue : en CS, comparer `zh-TW` et `auto`. Garder le mode `unified`.
- Pondérer la perte par la confiance du teacher.
- Phase de réparation (lr 4e-6) si l'anglais ou le français dérive, comme en v3.

## Phase 3 : tokenizer traditionnel (décision après la Phase 2)
- **Option A** (défaut, peu coûteuse) : garder le simplifié plus OpenCC. Faire passer la conversion en **`s2twp`** (niveau phrase, vocabulaire TW) au lieu de `s2t`, et mesurer les erreurs d'ambiguïté (发→發/髮, 后→後/后, 台→臺/台) sur l'éval.
- **Option B** : étendre le vocabulaire avec les caractères traditionnels manquants. Le joint/décodeur RNNT grandit ; les nouvelles lignes sont initialisées depuis le caractère simplifié correspondant, puis on fine-tune directement sur les cibles traditionnelles du teacher. Cela supprime l'étape OpenCC et ses ambiguïtés, mais demande de revérifier toutes les autres langues.
- Critère : passer à B seulement si le taux d'erreur dû à la conversion OpenCC dépasse 1 CER absolu sur l'éval zh-TW.

## Phase 4 (option) : distillation des représentations
- Ajouter une perte L2/cosinus entre l'encodeur élève et l'encodeur ASR du teacher (Transformer de 1 280 dimensions) via une projection linéaire, avec λ≈0.1. Les features fp16 sous-échantillonnées sont pré-calculées sur environ 300 h (~100 Go).
- Point délicat : l'élève est causal/streaming alors que l'encodeur teacher voit tout le contexte. Ne viser que les couches hautes et mesurer le gain avant de généraliser.

## Phase 5 : itérations
- Hard mining : là où v4 et le teacher divergent le plus, on ajoute de la donnée de ce type et on relabellise.
- Context biasing (NeMo GPU-PB) pour le vocabulaire tech anglais au déploiement.

## Tableau de bord cible (v4)
| Éval | v3 | Cible v4 |
|---|---|---|
| CV zh-TW CER | 9.81 | **< 7.5** |
| Taiwan-Tongues CER | 14.85 | **< 10** |
| CS zh-TW/en MER (nouveau jeu) | à mesurer | −30 % relatif vs v3 |
| LibriSpeech WER | 2.79 | ≤ 2.8 |
| FLEURS fr WER | 16.15 | ≤ 15.5 |

## Environnement
```bash
source env.sh   # CUDA_VISIBLE_DEVICES=1, STORE, HF_HOME, TMPDIR
```

## Résultats Phase 0 (2026-10-29) : teacher GGUF AD-NVFP4 via NeMo-Speech.cpp, GPU1
Mêmes extraits (CER, normalisation t2s, ponctuation retirée) :

| Système | CV zh-TW (120) | Taiwan-Tongues (100) |
|---|---|---|
| v3 (élève actuel) | **14.25** | 14.85 |
| Teacher, `-l zh-TW` | 26.84–28.62 | **11.90** |
| Teacher, prompt « Mandarin TW » (P2) | 20.78 | 14.11 |
| Teacher, prompt CS ASCEND (P1) | 40.26 | 20.37 |

- Le teacher **sort toujours du simplifié**, ce qui est compatible avec le tokenizer de l'élève.
- Sur les clips CV courts et lus, le teacher **hallucine ou change de langue** : romanisation cantonaise/hokkien, anglais, bengali.
- Avec le filtre d'accord CER(v3, teacher) ≤ 0.5, on garde 82–91 % des clips, et le teacher passe **devant** v3 : 5.67 vs 6.66 sur CV, 8.28 vs 12.35 sur TT.
- **Décision** : GO pour la distillation, avec un filtre d'accord v3/teacher à 0.5 obligatoire. La plus-value est maximale sur la parole naturelle (type TT/IVOD/podcasts) et faible sur la lecture courte (CV).
- Débit : environ 35 s par lot de 100 clips courts, chargement compris (MTP draft 7, 16 slots).

## Pseudo-labeling IVOD (en cours depuis 2026-10-29)
- Source : `OpenFormosa/parliament`, 308 521 clips, **1 299 h**, 10 à 20 s, sessions 2020-09 → 2026-04. **Licence CC BY-NC 4.0** (non commerciale).
- Le texte humain est édité et mal aligné aux bornes (CER médian de 0,22 vs l'ASR amont). Il sert de **signal de filtrage** et non de cible.
- **Traduction involontaire** : avec `-l zh-TW` ou le prompt officiel, le teacher traduit en anglais 20 à 44 % des clips IVOD. Test sur 200 clips :

  | Réglage | Sorties en anglais | CER médian vs humain | Part ≤ 0.4 |
  |---|---|---|---|
  | aucun | 88/200 | 0.47 | 44 % |
  | `-l zh` | 54 | 0.36 | 57 % |
  | P2 (anglais, « Mandarin TW ») | 39 | 0.35 | 59 % |
  | P3 (officiel verbatim) | 74 | 0.42 | 47 % |
  | **P4 (prompt en chinois)** | **0** | **0.29** | **81 %** |

  P4 : `请将这段语音逐字转写成中文文字。说话的语言是台湾的普通话，可能夹杂英文单词；英文单词保留原文。不要翻译。只输出转写内容。`
- Débit sur GPU1 : RTFx ≈ 166, soit environ 10 h pour 1 299 h (décodage audio compris).
- Ensuite : réévaluer CV/TT avec P4, puis appliquer le filtre combiné (accord avec le texte humain et accord avec v3).
