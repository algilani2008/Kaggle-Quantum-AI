# ARC-AGI-2 — DSL Program-Search Solver

`pipeline.py` fits a library of 105 candidate DSL programs to each task's train pairs, keeps only the programs that reproduce **every** train output exactly, ranks them by simplicity, and uses the top two as `attempt_1` / `attempt_2`.

It depends only on `numpy` (see `requirements.txt`), so it runs offline inside a Kaggle code-competition notebook.

## DSL families

| Family | What it does |
|---|---|
| Geometry | identity, rotations, flips, transposes |
| Crop / Scale | crop to non-background box; integer up/down-scaling |
| **Pattern Replication** | tile the input into the output with a learned transform per tile; repair a masked periodic pattern using the smallest consistent 2-D period; repair a mirror/rotation-symmetric pattern |
| **Grid Reconstruction** | split on separator lines into equal cells, then combine cells pixel-wise with a learned truth table (covers AND/OR/XOR/NOR variants) or select the odd-one-out / most / fewest-coloured cell |
| **Dynamic Shape Matching** | connected objects (4/8-connectivity, single/multi-colour): extract the largest, smallest, unique-colour or unique-shape object; recolour objects by a learned rule on size, size rank, shape signature or bounding box |
| Post-processing | learned colour mapping after any base family; geometry composed after other families (depth-2 search) |

## Measured performance (public ARC-AGI-2 data, pass@2)

| Split | Tasks | Solved |
|---|---|---|
| Training | 1000 | **6.9%** |
| Evaluation | 120 | **0.0%** |

Full breakdowns: `report_training.md`, `report_evaluation.md`.

The training split includes many easier ARC-AGI-1-style tasks. The evaluation split is representative of the hidden test set, and a hand-written DSL of this size does not solve any of it. **This version will not reach the top of the leaderboard.** It is a correct, offline-safe base to build on.

## Run

```bash
pip install -r requirements.txt
# public repo layout (one JSON per task), scored when test outputs exist
python pipeline.py --tasks-dir path/to/ARC-AGI-2/data/evaluation --out-dir output
# Kaggle layout
python pipeline.py --challenges arc-agi_test_challenges.json --out-dir /kaggle/working
python pipeline.py --challenges arc-agi_evaluation_challenges.json \
                   --solutions arc-agi_evaluation_solutions.json --out-dir output
```

Writes `submission.json` (Kaggle format), `report.json` and `report.md`.

## Strategy to improve the score

Ordered by expected gain on the evaluation split:

1. **Neural test-time training (largest gain).** Strong ARC results come from fine-tuning a small language model on each test task's own train pairs at inference time, with heavy augmentation (the 8 geometric transforms, colour permutations, pair reordering), then sampling and voting over candidates. This needs a GPU notebook and a pre-trained checkpoint attached as a Kaggle dataset, which makes it a separate project from this file.
2. **Use the DSL as a verifier, not only a generator.** Keep this search, but also score neural candidates against the train pairs: any candidate a fitted DSL program agrees with gets ranked first. This is the cheapest way to combine the two.
3. **Grow the DSL from evaluation failures.** `report.json` lists, per task, which programs fit. Hand-inspect evaluation tasks with zero fits, group them by the missing operation (object movement, line drawing, flood fill, counting-based output size, conditional rules) and add one family per group, re-measuring after each.
4. **Deeper search with pruning.** Allow 2–3 step compositions of all families, pruned by output-shape consistency with the train outputs before running a full check.
5. **Better attempt_2.** Today attempt_2 is the second-simplest fitted program or the input. Use augmentation agreement instead: run the solver on rotated/flipped versions of the task and pick the answer that appears most often after transforming back.

Measure every change on the evaluation split, never on the training split alone, since gains on the training split often do not transfer.
