# Model card: tashkeel-linear v0.1

## Model details

* **Task:** Arabic diacritization (restoring harakat / tashkeel) — one label per Arabic letter out of 15 classes.
* **Architecture:** multi-class linear classifier (one-vs-rest, hinge loss, averaged SGD, scikit-learn 1.9) over 2^21 hashed binary features (character window +-5, character 2-5-grams, word identity x letter position, neighbouring words). A rule forbids tanween except on the last two letters of a word.
* **Size:** 15 x 2,097,152 weights stored as int8 + per-class scale, 7.5 MB compressed (`weights/tashkeel-linear.npz` + `.json`).
* **Training:** 15 epochs, mini-batches of 5,000 lines, L2 alpha = 1e-7, seed 0; ~10 minutes on 2 CPU cores. Reproduce with `python train.py`.
* **Version / date:** 0.1.0, 2026-11-17. Author: Abdulkarim Algilani (repository owner); code generated with AI assistance.

## Intended use

* Adding a first pass of diacritics to classical / formal Arabic text for reading aids, text-to-speech preprocessing, search normalisation, or as a fast, transparent research baseline.
* Output should be **reviewed by a person** whenever correctness matters.

**Out of scope:** Quran text or other sacred text without expert review; legal, medical or exam material without review; dialectal Arabic and social-media text; any use to identify or profile people.

## Training data

* [AliOsm/arabic-text-diacritization](https://github.com/AliOsm/arabic-text-diacritization) (MIT), 50,000 training lines (8.36 M letters), 2,500 validation and 2,500 test lines, as published by the dataset authors (no re-splitting).
* The lines come from the [Tashkeela corpus](https://sourceforge.net/projects/tashkeela/) (GPLv2), mostly classical Islamic jurisprudence and literature books.
* Licence note: the weights are a derivative of this text. Redistribution is permitted by both licences; cautious users should treat the weights as GPLv2.

## Evaluation (test split, 2,500 lines, 426,469 letters, 107,291 words; measured 2026-11-17)

| System | DER | DER (no case ending) | WER | WER (no case ending) |
|---|---|---|---|---|
| Most frequent mark per letter | 48.02% | 46.99% | 84.79% | 73.96% |
| Word lookup (memorised training words + letter fallback) | 9.05% | 5.59% | 24.15% | 9.88% |
| **tashkeel-linear** | **4.71%** | **3.19%** | **14.81%** | **7.33%** |

DER counts every Arabic letter, including letters whose correct label is "no mark". Hash size (2^20 vs 2^21) and number of epochs were chosen by validation DER over three training runs (final val DER 4.63%). Those exploratory runs also printed test scores, but selection used validation only. Per-label error rates are in `results/metrics.md`.

## Limitations and biases

* **Domain bias:** trained almost entirely on classical religious-legal and literary prose, so vocabulary and style of modern news, technology, names of modern places/brands and dialects are under-represented; accuracy there is lower (not measured, since no openly licensed modern diacritized test set was used).
* **Grammar:** case endings (i'rab) are the main error source; the model sees only nearby words, not full syntax.
* **Rare labels:** dammatan (27% error), shadda + tanween (16-29%) and shadda alone (90%) are poorly predicted.
* **No spelling correction:** hamza/alef spelling mistakes in the input are kept and may lead to wrong marks.
* Results come from a single seed; no confidence intervals.

## Ethical considerations

The model processes text only; it stores no personal data and is not designed to identify, profile or track anyone. Diacritization errors can change meaning, so it must not be presented as authoritative for religious or legal texts.
