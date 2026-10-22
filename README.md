# Pre-release box office revenue prediction with tree-based ensembles (TMDB 5000)

Code, results and figures for the article *Pre-Release Prediction of Movie Box Office Revenue with
Tree-Based Ensembles: A Chronological Evaluation on TMDB Data* (A. Amirbekov, BDA-2402, Astana IT University).

Research questions

* **RQ1** Using only information available before release, which tuned tree-based ensemble
  (Random Forest, XGBoost, LightGBM, CatBoost) gives the most accurate out-of-sample worldwide revenue
  predictions for later releases, and does it improve on the naive and linear baselines?
* **RQ2** Which pre-release characteristics contribute most strongly to the predictions of the
  best-performing tree-based model (SHAP, non-causal)?

## Repository layout

```
run_all.sh               one command that reproduces everything
requirements.txt         exact package versions of the reported run
src/config.py            paths, seeds (42; 42-46 for seed runs), thresholds, search budget
src/data.py              join, cleaning rules, exclusion log, chronological split
src/features.py          pre-release features, time-aware past box-office features, A2 encoding
src/models.py            baselines, model factories, search spaces, CatBoost CLI wrapper
src/metrics.py           log/dollar metrics, Duan smearing, bootstrap CIs, paired tests, Holm
src/run_experiment.py    tuning, test evaluation and all additional analyses
src/make_figures.py      figures/*.pdf (+ .png previews)
src/make_tables.py       outputs/tables/*.tex
outputs/                 every result file (JSON/CSV) and run_log.txt with wall-clock times
figures/                 vector figures used in the article
```

## Data

Download the *TMDB 5000 Movie Dataset* from Kaggle
(<https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>) and put `tmdb_5000_movies.csv` and
`tmdb_5000_credits.csv` into `data/` (or set `TMDB_DATA_DIR`). The CSV files are not modified.

## Running

```bash
pip install -r requirements.txt        # see notes below for xgboost / catboost
export CATBOOST_CLI=/path/to/catboost   # CatBoost 1.2.8 command-line binary
./run_all.sh
```

`QUICK=1 ./run_all.sh` runs a smoke test with two tuning candidates per model (results go to
`outputs_quick/`, not used in the article).

### XGBoost and CatBoost

The reported run was made in a container without PyPI access. XGBoost 3.0.2 was compiled from the
official source tag (`cmake` build of `libxgboost.so`; the `python-package` folder is put on
`sys.path`, set `XGB_PYTHON_PACKAGE` to its location). With normal internet access
`pip install xgboost==3.0.2` is equivalent. CatBoost 1.2.8 was used through its official
command-line binary (release asset `catboost-linux-x86_64-1.2.8`); `src/models.py` contains a
scikit-learn compatible wrapper (`CatBoostCLIRegressor`) that writes TSV pools and a column
description file (`Categ` for original language), calls `catboost fit` / `calc`, and obtains exact
TreeSHAP values with `catboost fstr --fstr-type ShapValues`.

## Environment of the reported run

See `outputs/environment.json` and `outputs/run_log.txt`.

Reported run: Intel Xeon @ 2.80 GHz (2 vCPU), 7 GB RAM, no GPU; Python 3.13.16, numpy 2.5.3, pandas 3.0.5,
scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.0.2 (source build), lightgbm 4.6.0, CatBoost CLI 1.2.8,
matplotlib 3.11.2. `./run_all.sh` took 968 s wall-clock (tuning: RF 190 s, XGBoost 219 s, LightGBM 97 s,
CatBoost 262 s). Two complete runs gave result files that agree to within 1e-12.

## Main results (671 test films, released 2011-11-25 to 2016-09-09)

| Model | CV log RMSE | Test log MAE | Test log RMSE [95% CI] | Test log R2 | USD MAE (smeared), M |
|---|---|---|---|---|---|
| Naive median | 1.890 ± 0.259 | 1.492 | 2.026 [1.851, 2.208] | -0.003 | 148.7 |
| Budget-only OLS | 1.482 ± 0.166 | 1.069 | 1.509 [1.353, 1.681] | 0.444 | 91.6 |
| Linear Regression | 1.589 ± 0.184 | 1.010 | 1.413 [1.270, 1.567] | 0.512 | 165.2 |
| Random Forest | 1.361 ± 0.148 | 0.999 | 1.437 [1.292, 1.583] | 0.495 | 108.8 |
| XGBoost | 1.360 ± 0.147 | 1.023 | 1.478 [1.331, 1.629] | 0.466 | 116.4 |
| LightGBM | 1.361 ± 0.147 | 1.000 | 1.426 [1.284, 1.572] | 0.503 | 111.4 |
| CatBoost | 1.343 ± 0.144 | 0.975 | 1.407 [1.265, 1.554] | 0.516 | 108.7 |

Full tables (with CIs, naive back-transform, tests, ablation, IPW, 70/30 split) are in
`outputs/tables/` and `outputs/*.csv`; `outputs/all_numbers.json` lists every number.

## Output files

| File | Content |
|---|---|
| `data_summary.json`, `excluded_records.csv`, `selection_comparison.json` | cleaning log, removed records with reason, excluded vs retained films |
| `hyperparameters.json`, `cv_results.json` | selected hyperparameters, CV fold errors, tuning time |
| `metrics_test.csv`, `predictions_test.csv`, `smearing.json` | test metrics with bootstrap CIs, per-film predictions, smearing factors |
| `stat_tests.csv`, `stat_tests_baselines.csv`, `seed_runs.csv` | paired bootstrap / Wilcoxon / Holm, seeds 42-46 |
| `ablation.csv`, `ipw_metrics.csv`, `ipw_info.json`, `split70_*.csv`, `group_errors.csv` | additional analyses |
| `errors_best_model.csv`, `worst_*.csv`, `error_analysis.json`, `calibration_deciles.csv` | error analysis |
| `shap_global.csv`, `shap_groups.csv`, `shap_info.json`, `shap_values_test.npy` | SHAP (CatBoost, test set) |
| `environment.json`, `timings.json`, `run_log.txt` | versions, hardware, running times |
