#!/usr/bin/env bash
# Reproduces every number, table and figure of the article.
#   TMDB_DATA_DIR  folder with tmdb_5000_movies.csv and tmdb_5000_credits.csv (default: ./data)
#   CATBOOST_CLI   path to the CatBoost 1.2.8 command-line binary
#   XGB_PYTHON_PACKAGE  path to an XGBoost python-package folder if xgboost is not pip-installed
set -euo pipefail
cd "$(dirname "$0")"
start=$(date +%s)
python -m src.run_experiment      # data, tuning, evaluation, analyses -> outputs/
python -m src.make_figures        # figures/*.pdf
python -m src.make_tables         # outputs/tables/*.tex
python -m src.export_numbers      # outputs/all_numbers.json (every reported number)
end=$(date +%s)
echo "total wall-clock: $((end-start)) s" | tee -a outputs/run_log.txt
