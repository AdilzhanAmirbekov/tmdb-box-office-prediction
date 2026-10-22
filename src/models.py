"""Model definitions, search spaces and a scikit-learn style wrapper for the CatBoost CLI."""
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

if C.XGB_PYTHON_PACKAGE and os.path.isdir(C.XGB_PYTHON_PACKAGE):
    sys.path.insert(0, C.XGB_PYTHON_PACKAGE)
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402


# --------------------------------------------------------------------------- simple baselines
class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predicts the development median of the log target for every film."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class BudgetOnlyOLS(BaseEstimator, RegressorMixin):
    """log(1+revenue) ~ log(budget) + budget_missing (median-imputed log budget)."""

    def fit(self, X, y):
        lb = X["log_budget"].to_numpy(dtype=float)
        self.med_ = float(np.nanmedian(lb))
        self.lr_ = LinearRegression().fit(self._design(X), y)
        return self

    def _design(self, X):
        lb = X["log_budget"].to_numpy(dtype=float)
        miss = np.isnan(lb).astype(float)
        return np.column_stack([np.where(np.isnan(lb), self.med_, lb), miss])

    def predict(self, X):
        return self.lr_.predict(self._design(X))


def make_linear(X):
    """Linear regression with development-median imputation and standardisation of the
    non-binary numeric columns. Binary indicators (one-hot / multi-hot) pass through."""
    binary = [c for c in X.columns if set(pd.unique(X[c].dropna())) <= {0.0, 1.0}]
    cont = [c for c in X.columns if c not in binary]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cont),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ])
    return Pipeline([("pre", pre), ("lr", LinearRegression())])


# --------------------------------------------------------------------------- CatBoost CLI wrapper
class CatBoostCLIRegressor(BaseEstimator, RegressorMixin):
    """Minimal scikit-learn compatible wrapper around the CatBoost 1.2.8 command-line binary.
    Data are exchanged as TSV files; categorical columns are declared as `Categ` in the
    column-description file and handled natively by CatBoost."""

    def __init__(self, iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                 subsample=0.8, rsm=1.0, random_strength=1.0, random_seed=42,
                 cat_features=None, thread_count=C.N_JOBS, binary=C.CATBOOST_CLI):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.rsm = rsm
        self.random_strength = random_strength
        self.random_seed = random_seed
        self.cat_features = cat_features
        self.thread_count = thread_count
        self.binary = binary

    # -- helpers
    def _write_pool(self, X, y, path):
        df = X.copy()
        for c in df.columns:
            if c in (self.cat_features or []):
                df[c] = df[c].astype(str)
        df.insert(0, "__target__", 0.0 if y is None else np.asarray(y, dtype=float))
        df.to_csv(path, sep="\t", header=False, index=False, na_rep="nan", float_format="%.10g")

    def _write_cd(self, X, path):
        with open(path, "w") as f:
            f.write("0\tLabel\n")
            for i, c in enumerate(X.columns, start=1):
                kind = "Categ" if c in (self.cat_features or []) else "Num"
                f.write(f"{i}\t{kind}\t{c}\n")

    def fit(self, X, y):
        self.workdir_ = tempfile.mkdtemp(prefix="cbcli_")
        self.columns_ = list(X.columns)
        learn, cd = os.path.join(self.workdir_, "learn.tsv"), os.path.join(self.workdir_, "pool.cd")
        self.model_path_ = os.path.join(self.workdir_, "model.bin")
        self._write_pool(X, y, learn)
        self._write_cd(X, cd)
        cmd = [self.binary, "fit", "--learn-set", learn, "--column-description", cd,
               "--loss-function", "RMSE", "--iterations", str(int(self.iterations)),
               "--learning-rate", f"{self.learning_rate:.6g}", "--depth", str(int(self.depth)),
               "--l2-leaf-reg", f"{self.l2_leaf_reg:.6g}", "--bootstrap-type", "Bernoulli",
               "--subsample", f"{self.subsample:.6g}", "--rsm", f"{self.rsm:.6g}",
               "--random-strength", f"{self.random_strength:.6g}",
               "--random-seed", str(int(self.random_seed)), "--thread-count", str(self.thread_count),
               "--model-file", self.model_path_, "--train-dir", self.workdir_,
               "--allow-writing-files", "false", "--logging-level", "Silent"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return self

    def _pool_for(self, X):
        path = os.path.join(self.workdir_, f"in_{uuid.uuid4().hex}.tsv")
        self._write_pool(X[self.columns_], None, path)
        return path

    def predict(self, X):
        inp = self._pool_for(X)
        out = inp.replace(".tsv", "_pred.tsv")
        subprocess.run([self.binary, "calc", "--model-file", self.model_path_, "--input-path", inp,
                        "--column-description", os.path.join(self.workdir_, "pool.cd"),
                        "--output-path", out, "--prediction-type", "RawFormulaVal",
                        "--thread-count", str(self.thread_count)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        pred = pd.read_csv(out, sep="\t")
        os.remove(inp), os.remove(out)
        return pred.iloc[:, -1].to_numpy(dtype=float)

    def shap_values(self, X):
        """Exact TreeSHAP values from `catboost fstr --fstr-type ShapValues`.
        Returns array (n, n_features + 1); last column = expected value."""
        inp = self._pool_for(X)
        out = inp.replace(".tsv", "_shap.tsv")
        subprocess.run([self.binary, "fstr", "--model-file", self.model_path_, "--input-path", inp,
                        "--column-description", os.path.join(self.workdir_, "pool.cd"),
                        "--output-path", out, "--fstr-type", "ShapValues",
                        "--thread-count", str(self.thread_count)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        sv = pd.read_csv(out, sep="\t", header=None).to_numpy(dtype=float)
        os.remove(inp), os.remove(out)
        return sv

    def save(self, path):
        shutil.copy(self.model_path_, path)

    def __del__(self):
        try:
            shutil.rmtree(self.workdir_, ignore_errors=True)
        except Exception:
            pass


# --------------------------------------------------------------------------- tuned models
def tree_models(seed=C.SEED, cat_features=("language",)):
    """Estimator + randomized-search space for the four ensembles (A2 hyperparameters)."""
    return {
        "Random Forest": (
            RandomForestRegressor(random_state=seed, n_jobs=C.N_JOBS),
            {"n_estimators": [200, 300, 500],
             "max_depth": [None, 8, 12, 16, 20],
             "min_samples_split": [2, 5, 10, 20],
             "min_samples_leaf": [1, 2, 4, 8],
             "max_features": [0.2, 0.33, 0.5, 0.7, 1.0]}),
        "XGBoost": (
            xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed,
                             n_jobs=C.N_JOBS, verbosity=0),
            {"n_estimators": randint(200, 1001),
             "learning_rate": loguniform(0.01, 0.2),
             "max_depth": randint(2, 9),
             "min_child_weight": [1, 3, 5, 10],
             "subsample": uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.5, 0.5),
             "reg_lambda": loguniform(0.1, 10),
             "reg_alpha": loguniform(1e-3, 1)}),
        "LightGBM": (
            lgb.LGBMRegressor(random_state=seed, n_jobs=C.N_JOBS, subsample_freq=1, verbose=-1),
            {"n_estimators": randint(200, 1001),
             "learning_rate": loguniform(0.01, 0.2),
             "num_leaves": [7, 15, 31, 63],
             "max_depth": [-1, 4, 6, 8],
             "min_child_samples": [5, 10, 20, 40],
             "subsample": uniform(0.6, 0.4),
             "colsample_bytree": uniform(0.5, 0.5),
             "reg_lambda": loguniform(1e-3, 10),
             "reg_alpha": loguniform(1e-3, 1)}),
        "CatBoost": (
            CatBoostCLIRegressor(random_seed=seed, cat_features=list(cat_features)),
            {"iterations": [300, 500, 800, 1000],
             "learning_rate": loguniform(0.02, 0.2),
             "depth": randint(4, 9),
             "l2_leaf_reg": loguniform(1, 10),
             "subsample": uniform(0.6, 0.4),
             "rsm": uniform(0.5, 0.5),
             "random_strength": loguniform(0.1, 10)}),
    }


SEED_PARAM = {"Random Forest": "random_state", "XGBoost": "random_state",
              "LightGBM": "random_state", "CatBoost": "random_seed"}

USES_CB_MATRIX = {"CatBoost"}

VERSIONS = {"xgboost": xgb.__version__, "lightgbm": lgb.__version__}
