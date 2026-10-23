# Competitions

One folder per Kaggle competition, named by its slug and created automatically by the daily brief scheduled task from `templates/competition/`:

```
competitions/<slug>/
  README.md     competition summary (name, link, dates, metric, task type, data)
  config.json   settings for run.py (tabular flag, target, id column, folds)
  run.py        runs the shared ../../pipeline.py for this competition
```

Data, outputs and `state.json` are not committed (see `.gitignore`); they live in the matching Google Drive folder under `Kaggle_Projects`.

To run one locally after placing its CSVs in `competitions/<slug>/data/`:

```bash
python competitions/<slug>/run.py
```
