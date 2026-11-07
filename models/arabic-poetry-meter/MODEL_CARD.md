# Model card: bahr (Arabic poetry meter classifier) v0.1.0

## Summary

`bahr` predicts which of **14 classical Arabic meters (البحور)** a verse (بيت) is written in.
It is a linear SVM over three groups of features: character n-grams, n-grams of a
"rhythm skeleton" (consonant / long vowel / short vowel / sukun / shadda / tanween,
with the start and end of each hemistich marked), and the letter count of each hemistich.
It is small (5.7 MB on disk), runs on a CPU in milliseconds per verse, and is stored as
JSON + a NumPy archive with **no pickle**, so loading it never executes code.

| | |
|---|---|
| Task | 14-way single-label text classification |
| Input | one verse, hemistichs separated by `#` (diacritics optional) |
| Output | meter label plus raw SVM scores for the top-k meters |
| Architecture | TF-IDF char 1-5-grams (100k) + TF-IDF rhythm-skeleton 2-8-grams (150k) + one-hot hemistich lengths -> `LinearSVC(C=1.0)` |
| Weights | `weights/` (linear weights stored in float16; identical test predictions to float32, checked) |
| Training time | about 8 minutes total for `train.py` (including all baselines) on 2 CPU cores |
| License | MIT (same as this repository); training data MIT (see below) |

## Intended use

- Helping students, teachers and poetry readers get a **first guess** of a verse's meter,
  with the top 3 candidates to check by hand (تقطيع).
- A baseline for researchers working on Arabic prosody, to compare neural models against.
- Sorting or tagging large verse collections where a human checks the uncertain cases.

**Not intended** for grading students, for judging whether a verse is "correct" poetry,
or for any decision about people. It does not detect broken meter (كسر): it always
picks one of 14 meters, even for prose.

## Training data

**MetRec** (Al-shaibani, Alyafeai & Ahmad, 2020), downloaded by `train.py` from
<https://github.com/zaidalyafeai/MetRec> (`baits.zip`, MIT License, SHA-256 checked).
55,440 verses labelled with 14 meters; official split 47,124 train / 8,316 test.
The data is not committed here. The two rarest of the 16 classical meters (al-muqtadab
and al-mudari') are not in the dataset, so the model cannot predict them.

Validation: a stratified 10% of the official train split (4,713 verses, seed 0) was
held out for every choice (features, `C`, augmentation). The test split was used once,
for the final model.

Augmentation: each training verse is also added with all diacritics removed, so the
model works on verses typed without tashkeel.

## Metrics (measured in this run, official MetRec test split, n = 8,316)

| Model | Test accuracy |
|---|---|
| Majority class | 8.6% |
| Hemistich length only | 24.8% |
| Character n-grams only | 60.1% |
| **bahr (full, augmented)** | **84.5%** (95% bootstrap CI 83.8-85.3%) |

- Macro-F1: 0.837
- Test with the 23 verses that also appear in train removed: 84.5%
- Same test verses with **all diacritics removed**: 77.2%. On the validation set with
  diacritics removed, the same model trained without the augmentation scored 59.3% and
  the augmented one 78.2%.
- By diacritization of the input: none 75.9% (n=1,288), partial 80.9% (n=2,401),
  heavy 88.7% (n=4,627).
- Best meters: al-taweel F1 0.97, al-baseet 0.93, al-mujtath 0.92. Weakest: al-madeed 0.66
  (often taken for al-ramal), al-rajaz 0.69 and al-saree' 0.71 (confused with each other).

Reference point, **not re-run here**: the dataset authors report 94.3% with a 5-layer
bidirectional GRU on the same split. `bahr` is about 10 points less accurate but needs no
GPU or deep-learning framework and is fully inspectable.

## Limitations and biases

- Accuracy depends strongly on diacritics; bare text is harder because the short vowels
  that make up the rhythm are invisible.
- Verses come from online poetry collections; the meter labels were not all checked by
  prosody experts, so some labels may be wrong. The test score inherits that noise.
- Verses from the same poem can fall in both train and test (MetRec has no poem ids),
  which may make the test score somewhat optimistic for brand-new poems.
- Classical meters only; Nabati / colloquial poetry (الشعر النبطي) and free verse are out
  of scope, and the model will still output a classical meter for them.
- Classes with fewer examples (al-madeed, al-hazaj, al-mutadarak) are recognised less
  reliably.
- Scores are SVM margins, not calibrated probabilities.

## How to reproduce

```bash
pip install -r requirements.txt
python train.py      # downloads MetRec, trains, writes results/ and weights/
python plots.py      # charts in results/
pytest -q tests
```

## Citation of the data

Al-shaibani, M. S., Alyafeai, Z., & Ahmad, I. (2020). MetRec: A dataset for meter
classification of Arabic poetry. *Data in Brief*, 33, 106497.
