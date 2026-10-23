# Kaggle Auto-ML Competition Engine

Advanced stacking pipeline that runs daily as a scheduled task.

## What it does
1. Loads `train*.csv`, `test*.csv` and `sample_submission*.csv` from the data folder and infers target, ID column and task type.
2. Engineers features scaled to the data's dimensions: categorical encoding, row statistics, and product/ratio/difference cross features among the top numeric columns by mutual information.
3. Stacks XGBoost, LightGBM and CatBoost (K-fold out-of-fold predictions + linear meta-model).
4. Runs error analysis on the worst 10% of predictions and retrains with adjusted hyperparameters (and hard-example weighting when warranted); keeps the better round.
5. Writes `submission.csv` only if the CV score beats the baseline (best previous run in `state.json`, or a default LightGBM on the first run).
6. Writes `report.md` and `report.json`.

## Run
```bash
pip install -r requirements.txt
python pipeline.py --data-dir data --out-dir output --state-file state.json
```
Options: `--target`, `--id-col`, `--folds` (default 5), `--max-train-rows`.

## Data
Competition CSVs live in the Google Drive folder `Kaggle_Quantum_Data`. `state.json` (the baseline) is kept in that folder too so it survives between runs.

## Submitting
Upload `output/submission.csv` to the competition page on Kaggle. Do not commit API keys to this repo.
