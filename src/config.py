"""Global configuration for the box office experiment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TMDB_DATA_DIR", ROOT / "data"))
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
MODEL_DIR = OUT_DIR / "models"
TABLE_DIR = OUT_DIR / "tables"

# Optional locations of non-PyPI builds used in the original run (see README).
XGB_PYTHON_PACKAGE = os.environ.get("XGB_PYTHON_PACKAGE", "/home/claude/wheels/xgb-src/python-package")
CATBOOST_CLI = os.environ.get("CATBOOST_CLI", "/home/claude/a3/catboost_cli")

SEED = 42
SEEDS = [42, 43, 44, 45, 46]
N_JOBS = 2

TEST_FRACTION = 0.20          # latest 20 % of films = final test set
N_CV_FOLDS = 5                # expanding-window folds inside development
MIN_VALID_USD = 1_000         # revenue/budget below this are treated as entry errors
CAT_MIN_COUNT = 10            # A2 frequency threshold (development films only)
HISTORY_GAP_DAYS = 180        # a past film counts as history only if released >= 180 days earlier
BOOTSTRAP_B = 2000
ALPHA = 0.05

N_ITER = {"Random Forest": 30, "XGBoost": 50, "LightGBM": 50, "CatBoost": 30}

if os.environ.get("QUICK"):  # smoke test only
    N_ITER = {k: 2 for k in N_ITER}
    BOOTSTRAP_B = 200
    OUT_DIR = ROOT / "outputs_quick"
    MODEL_DIR = OUT_DIR / "models"
    TABLE_DIR = OUT_DIR / "tables"

for d in (OUT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)
