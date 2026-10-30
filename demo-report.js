window.DEMO_REPORT = {
 "version": "1.0",
 "run_at": "2026-10-30T11:58:35.458980+00:00",
 "task": "binary",
 "metric": "roc_auc",
 "higher_is_better": true,
 "target": "target",
 "id_col": "id",
 "features": {
  "input_shape": [
   6000,
   16
  ],
  "categorical_encoded": [
   "region",
   "plan"
  ],
  "row_stats": [
   "row__mean",
   "row__std",
   "row__nan"
  ],
  "top_numeric_by_mutual_info": [
   "f8",
   "f5",
   "f12",
   "f6",
   "f9",
   "f1",
   "f2",
   "f10"
  ],
  "cross_features": [
   "f8__x__f5",
   "f8__div__f5",
   "f8__minus__f5",
   "f8__x__f12",
   "f8__div__f12",
   "f8__minus__f12",
   "f8__x__f6",
   "f8__div__f6",
   "f8__minus__f6",
   "f8__x__f9",
   "f8__div__f9",
   "f8__minus__f9",
   "f8__x__f1",
   "f8__div__f1",
   "f8__minus__f1",
   "f8__x__f2",
   "f8__div__f2",
   "f8__minus__f2",
   "f8__x__f10",
   "f8__div__f10",
   "f8__minus__f10",
   "f5__x__f12",
   "f5__div__f12",
   "f5__minus__f12",
   "f5__x__f6",
   "f5__div__f6",
   "f5__minus__f6",
   "f5__x__f9",
   "f5__div__f9",
   "f5__minus__f9",
   "f5__x__f1",
   "f5__div__f1",
   "f5__minus__f1",
   "f5__x__f2",
   "f5__div__f2",
   "f5__minus__f2",
   "f5__x__f10",
   "f5__div__f10",
   "f5__minus__f10",
   "f12__x__f6",
   "f12__div__f6",
   "f12__minus__f6",
   "f12__x__f9",
   "f12__div__f9",
   "f12__minus__f9",
   "f12__x__f1",
   "f12__div__f1",
   "f12__minus__f1",
   "f12__x__f2",
   "f12__div__f2",
   "f12__minus__f2",
   "f12__x__f10",
   "f12__div__f10",
   "f12__minus__f10",
   "f6__x__f9",
   "f6__div__f9",
   "f6__minus__f9",
   "f6__x__f1",
   "f6__div__f1",
   "f6__minus__f1",
   "f6__x__f2",
   "f6__div__f2",
   "f6__minus__f2",
   "f6__x__f10",
   "f6__div__f10",
   "f6__minus__f10",
   "f9__x__f1",
   "f9__div__f1",
   "f9__minus__f1",
   "f9__x__f2",
   "f9__div__f2",
   "f9__minus__f2",
   "f9__x__f10",
   "f9__div__f10",
   "f9__minus__f10",
   "f1__x__f2",
   "f1__div__f2",
   "f1__minus__f2",
   "f1__x__f10",
   "f1__div__f10",
   "f1__minus__f10",
   "f2__x__f10",
   "f2__div__f10",
   "f2__minus__f10"
  ],
  "cross_feature_budget_k": 8,
  "dropped_constant": [
   "row__nan"
  ],
  "final_shape": [
   6000,
   104
  ]
 },
 "round1": {
  "params": {
   "n_estimators": 400,
   "learning_rate": 0.05,
   "max_depth": 6,
   "num_leaves": 31,
   "subsample": 0.8,
   "colsample": 0.8,
   "min_child_weight": 1,
   "min_child_samples": 20,
   "reg_lambda": 1.0
  },
  "base_scores": {
   "xgboost": 0.9839423207802416,
   "lightgbm": 0.9842453839123758,
   "catboost": 0.9837411490842827
  },
  "stack_score": 0.9839073754621643
 },
 "error_analysis": {
  "worst10_error_share": 0.8389,
  "top_shifted_features": [
   [
    "f6__minus__f2",
    0.517
   ],
   [
    "f8__minus__f6",
    0.479
   ],
   [
    "f6",
    0.434
   ],
   [
    "f6__minus__f10",
    0.427
   ],
   [
    "f6__minus__f9",
    0.39
   ]
  ],
  "actions": [
   "Errors concentrated (>50% of loss in worst 10%): increased regularisation",
   "Worst rows form a distinct region (max shift 0.52 SD): upweighting hard examples x2",
   "Lowered learning rate x0.7 and raised trees x1.4"
  ]
 },
 "round2": {
  "params": {
   "n_estimators": 560,
   "learning_rate": 0.034999999999999996,
   "max_depth": 6,
   "num_leaves": 31,
   "subsample": 0.7000000000000001,
   "colsample": 0.8,
   "min_child_weight": 3,
   "min_child_samples": 40,
   "reg_lambda": 3.0
  },
  "base_scores": {
   "xgboost": 0.983433808067424,
   "lightgbm": 0.9834361414590921,
   "catboost": 0.9834981430091307
  },
  "stack_score": 0.9837023703370363
 },
 "selected_round": 1,
 "final_cv_score": 0.9839073754621643,
 "baseline": 0.9843436085902146,
 "baseline_source": "default single LightGBM (first run)",
 "beat_baseline": false,
 "submission_written": false,
 "runtime_sec": 80.5,
 "log": [
  "[    0.0s] Loaded train (6000, 18), test (2000, 17); target='target', id='id'",
  "[    0.0s] Task: binary (2 classes)",
  "[    0.3s] Features: 16 -> 104 (84 cross features)",
  "[    0.3s] No previous baseline; computing default LightGBM baseline",
  "[    3.0s] Baseline roc_auc: 0.984344 [default single LightGBM (first run)]",
  "[    3.0s] Round 1: stacking XGBoost + LightGBM + CatBoost",
  "[    9.8s]   fold 1/5 done",
  "[   16.4s]   fold 2/5 done",
  "[   22.9s]   fold 3/5 done",
  "[   30.0s]   fold 4/5 done",
  "[   37.2s]   fold 5/5 done",
  "[   37.2s] Round 1 stack roc_auc: 0.983907  base: xgboost=0.98394, lightgbm=0.98425, catboost=0.98374",
  "[   37.2s] Error analysis: Errors concentrated (>50% of loss in worst 10%): increased regularisation",
  "[   37.2s] Error analysis: Worst rows form a distinct region (max shift 0.52 SD): upweighting hard examples x2",
  "[   37.2s] Error analysis: Lowered learning rate x0.7 and raised trees x1.4",
  "[   37.2s] Round 2: re-tuned stack",
  "[   46.0s]   fold 1/5 done",
  "[   54.6s]   fold 2/5 done",
  "[   63.2s]   fold 3/5 done",
  "[   71.8s]   fold 4/5 done",
  "[   80.5s]   fold 5/5 done",
  "[   80.5s] Round 2 stack roc_auc: 0.983702",
  "[   80.5s] Selected round 1: roc_auc=0.983907; beats baseline: False",
  "[   80.5s] Did not beat baseline: no submission.csv written"
 ],
 "demo_dataset": "synthetic: sklearn make_classification, 6,000 train / 2,000 test rows, 14 numeric + 2 categorical columns"
};
