"""End-to-end experiment: data -> features -> tuning -> test evaluation -> analyses.

Run from the repository root:  python -m src.run_experiment
All results are written to outputs/ (JSON / CSV) and outputs/state.pkl (for figures/tables).
"""
import json
import pickle
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from . import config as C
from . import data as D
from . import features as F
from . import metrics as Mx
from . import models as M

warnings.filterwarnings("ignore", category=UserWarning)
T0 = time.time()
TIMES = {}
OUT = C.OUT_DIR


def log(msg):
    line = f"[{time.time() - T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(OUT / "run_log.txt", "a") as f:
        f.write(line + "\n")


def dump(obj, name):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


ALL_MODELS = ["Naive median", "Budget-only OLS", "Linear Regression",
              "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
TREES = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]


def make_model(name, params=None, cat_cols=("language",), seed=C.SEED):
    if name == "Naive median":
        return M.MedianBaseline()
    if name == "Budget-only OLS":
        return M.BudgetOnlyOLS()
    if name == "Linear Regression":
        return None  # built from the training matrix (needs column types)
    est = clone(M.tree_models(seed, cat_cols)[name][0])
    if params:
        est.set_params(**params)
    est.set_params(**{M.SEED_PARAM[name]: seed})
    if name == "CatBoost":
        est.set_params(cat_features=list(cat_cols))
    return est


def fit_predict(name, params, mats, tr, te, seed=C.SEED):
    """mats = (X_tree, X_cb, cat_cols); tr/te boolean masks or index arrays; returns predictions."""
    Xt, Xc, cc = mats
    X = Xc if name in M.USES_CB_MATRIX else Xt
    y = mats_y[0]
    if name == "Linear Regression":
        est = M.make_linear(X.iloc[tr])
    else:
        est = make_model(name, params, cc, seed)
    est.fit(X.iloc[tr], y[tr])
    return est.predict(X.iloc[te]), est


def env_info():
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cpu = [l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")]
    mem = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()[1]
    cb = subprocess.run([C.CATBOOST_CLI, "--version"], capture_output=True, text=True).stdout
    cbtag = [l.split(":", 1)[1].strip() for l in cb.splitlines() if "Branch" in l]
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_model": cpu[0] if cpu else "", "n_cpu": int(subprocess.run(["nproc"], capture_output=True, text=True).stdout),
        "ram_gb_total": mem, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "xgboost": M.VERSIONS["xgboost"], "lightgbm": M.VERSIONS["lightgbm"],
        "catboost_cli": cbtag[0] if cbtag else "", "matplotlib": __import__("matplotlib").__version__,
    }


# =========================================================================== main
(OUT / "run_log.txt").write_text("")
ENV = env_info()
dump(ENV, "environment.json")
log(f"environment: {ENV}")

# ---------------------------------------------------------------- 1. data
t = time.time()
ALL, DLOG, EXC = D.load_joined()
ALL = F.add_history_features(ALL)
POP, DLOG, EXCL = D.modelling_population(ALL, DLOG, EXC)
EXCL.to_csv(OUT / "excluded_records.csv", index=False)
IS_DEV = D.chronological_split(POP)
DEV, TEST = POP[IS_DEV], POP[~IS_DEV]
VOCAB = F.fit_vocab(DEV)
Xt, Xc, CC = F.transform(POP, VOCAB, "history")
y = POP["log_revenue"].to_numpy()
mats_y = [y]
MATS = (Xt, Xc, CC)
dev_idx, test_idx = np.where(IS_DEV)[0], np.where(~IS_DEV)[0]
y_test = y[test_idx]


def tdist(s):
    usd = np.expm1(s) / 1e6
    return {"n": int(len(s)), "usd_min": float(usd.min()), "usd_q25": float(np.percentile(usd, 25)),
            "usd_median": float(np.median(usd)), "usd_mean": float(usd.mean()),
            "usd_q75": float(np.percentile(usd, 75)), "usd_max": float(usd.max()),
            "log_mean": float(s.mean()), "log_sd": float(s.std(ddof=1)),
            "share_below_1M": float(np.mean(usd < 1)), "skew_usd": float(pd.Series(usd).skew())}


tss = TimeSeriesSplit(n_splits=C.N_CV_FOLDS)
FOLDS = list(tss.split(dev_idx))
DLOG.update({
    "n_dev": int(IS_DEV.sum()), "n_test": int((~IS_DEV).sum()),
    "dev_date_min": str(DEV.release_date.min().date()), "dev_date_max": str(DEV.release_date.max().date()),
    "test_date_min": str(TEST.release_date.min().date()), "test_date_max": str(TEST.release_date.max().date()),
    "target_dev": tdist(DEV.log_revenue.to_numpy()), "target_test": tdist(TEST.log_revenue.to_numpy()),
    "target_all": tdist(POP.log_revenue.to_numpy()),
    "budget_missing_dev": int(DEV.budget_clean.isna().sum()), "budget_missing_test": int(TEST.budget_clean.isna().sum()),
    "n_features_tree": int(Xt.shape[1]), "n_features_catboost": int(Xc.shape[1]),
    "vocab_sizes": {k: len(v) for k, v in VOCAB.items()}, "languages_kept": VOCAB["languages"],
    "countries_kept": VOCAB["countries"],
    "cv_folds": [{"train_n": int(len(a)), "valid_n": int(len(b)),
                  "train_end": str(DEV.release_date.iloc[a[-1]].date()),
                  "valid_start": str(DEV.release_date.iloc[b[0]].date()),
                  "valid_end": str(DEV.release_date.iloc[b[-1]].date())} for a, b in FOLDS],
    "old_encoding": F.old_encoding_other_shares(DEV, VOCAB),
    "history_missing_share_dev": {c: float(DEV[c].mean()) for c in ["dir_hist_missing", "cast_hist_missing", "comp_hist_missing"]},
    "history_missing_share_test": {c: float(TEST[c].mean()) for c in ["dir_hist_missing", "cast_hist_missing", "comp_hist_missing"]},
})
DLOG["excluded_by_reason"] = EXCL.reason.value_counts().to_dict()


# selection comparison: revenue==0 films vs retained (all periods and development period)
def traits(d):
    top = ["Drama", "Comedy", "Thriller", "Action", "Romance", "Adventure", "Horror", "Documentary"]
    return {"n": int(len(d)), "share_budget_known": float(d.budget_clean.notna().mean()),
            "median_budget_usd_m": float(d.budget_clean.median() / 1e6),
            "median_release_year": float(d.release_date.dt.year.median()),
            "share_released_2000_or_later": float((d.release_date.dt.year >= 2000).mean()),
            "share_english": float((d.original_language == "en").mean()),
            "share_us_production": float(d.countries_l.apply(lambda l: "US" in l).mean()),
            "median_runtime": float(d.runtime_clean.median()),
            "share_director_history": float(1 - d.dir_hist_missing.mean()),
            "genre_shares": {g: float(d.genres_l.apply(lambda l, g=g: g in l).mean()) for g in top}}


SEL = {"retained": traits(POP), "excluded_revenue_zero": traits(ALL[ALL.revenue == 0])}
dump(SEL, "selection_comparison.json")
dump(DLOG, "data_summary.json")
TIMES["data_and_features_s"] = time.time() - t
log(f"data: population {len(POP)}, dev {IS_DEV.sum()}, test {(~IS_DEV).sum()}, features {Xt.shape[1]}")

# ---------------------------------------------------------------- 2. tuning
CV = {}
PARAMS = {}
Xdev_t, Xdev_c, ydev = Xt.iloc[dev_idx], Xc.iloc[dev_idx], y[dev_idx]
for name, (est, space) in M.tree_models(C.SEED, CC).items():
    t = time.time()
    X = Xdev_c if name in M.USES_CB_MATRIX else Xdev_t
    rs = RandomizedSearchCV(est, space, n_iter=C.N_ITER[name], cv=FOLDS, scoring="neg_root_mean_squared_error",
                            random_state=C.SEED, n_jobs=1, refit=False, error_score="raise")
    rs.fit(X, ydev)
    b = rs.best_index_
    folds = [-rs.cv_results_[f"split{k}_test_score"][b] for k in range(C.N_CV_FOLDS)]
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in rs.best_params_.items()}
    PARAMS[name] = best
    CV[name] = {"fold_log_rmse": folds, "mean": float(np.mean(folds)), "std": float(np.std(folds, ddof=1)),
                "n_iter": C.N_ITER[name], "tuning_time_s": time.time() - t,
                "all_candidates_mean_rmse": [-float(s) for s in rs.cv_results_["mean_test_score"]]}
    log(f"tuned {name}: CV log RMSE {CV[name]['mean']:.3f} +- {CV[name]['std']:.3f} ({time.time()-t:.0f}s) {best}")
    pd.to_pickle(rs.cv_results_, C.MODEL_DIR / f"cv_results_{name.replace(' ', '_')}.pkl")


# ---------------------------------------------------------------- 3. OOF (smearing) + final fit
def oof_predictions(name, params, mats, dev_ids, folds, seed=C.SEED):
    oof = np.full(len(dev_ids), np.nan)
    fold_rmse = []
    for a, b in folds:
        p, _ = fit_predict(name, params, mats, dev_ids[a], dev_ids[b], seed)
        oof[b] = p
        fold_rmse.append(float(np.sqrt(np.mean((y[dev_ids[b]] - p) ** 2))))
    return oof, fold_rmse


PRED, SMEAR, FIT_TIME, OOF = {}, {}, {}, {}
for name in ALL_MODELS:
    params = PARAMS.get(name)
    oof, fr = oof_predictions(name, params, MATS, dev_idx, FOLDS)
    ok = ~np.isnan(oof)
    resid = y[dev_idx][ok] - oof[ok]
    SMEAR[name] = Mx.smearing_factor(resid)
    OOF[name] = oof
    if name not in CV:
        CV[name] = {"fold_log_rmse": fr, "mean": float(np.mean(fr)), "std": float(np.std(fr, ddof=1))}
    t = time.time()
    p, est = fit_predict(name, params, MATS, dev_idx, test_idx)
    FIT_TIME[name] = time.time() - t
    PRED[name] = p
    if name == "CatBoost":
        est.save(C.MODEL_DIR / "catboost_final.bin")
    else:
        pd.to_pickle(est, C.MODEL_DIR / f"final_{name.replace(' ', '_')}.pkl")
    log(f"{name}: smearing {SMEAR[name]:.4f}, test log RMSE {np.sqrt(np.mean((y_test-p)**2)):.4f}, fit {FIT_TIME[name]:.1f}s")
dump(PARAMS, "hyperparameters.json")
dump({k: {kk: vv for kk, vv in v.items() if kk != "all_candidates_mean_rmse"} for k, v in CV.items()}, "cv_results.json")
dump({"smearing_factor": SMEAR, "residuals_used": "out-of-fold log residuals from the 5 expanding-window "
      "validation folds of the development period (films not in the first training block)",
      "n_oof_residuals": int((~np.isnan(OOF['Naive median'])).sum())}, "smearing.json")
dump(FIT_TIME, "final_fit_time.json")

# ---------------------------------------------------------------- 4. test metrics + bootstrap
BIDX = Mx.bootstrap_indices(len(test_idx))
MET = {}
rows = []
for name in ALL_MODELS:
    m = Mx.all_metrics(y_test, PRED[name], SMEAR[name])
    ci = Mx.bootstrap_ci(y_test, PRED[name], SMEAR[name], BIDX)
    MET[name] = {"point": m, "ci": ci}
    rows.append({"model": name, **m, **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()},
                 "cv_mean": CV[name]["mean"], "cv_std": CV[name]["std"], "smear": SMEAR[name],
                 "tuning_time_s": CV[name].get("tuning_time_s", 0.0), "fit_time_s": FIT_TIME[name]})
METDF = pd.DataFrame(rows)
METDF.to_csv(OUT / "metrics_test.csv", index=False)
log("test metrics:\n" + METDF[["model", "log_MAE", "log_RMSE", "log_R2", "usd_MAE_smear", "usd_RMSE_smear",
                               "usd_R2_smear", "usd_MedAE_smear", "cv_mean", "cv_std"]].round(3).to_string())

pd.DataFrame({"id": TEST.id.values, "title": TEST.title.values, "release_date": TEST.release_date.dt.date.values,
              "revenue_usd": TEST.revenue.values, "budget_usd": TEST.budget_clean.values, "y_log": y_test,
              **{f"pred_log_{k}": v for k, v in PRED.items()}}).to_csv(OUT / "predictions_test.csv", index=False)

# ---------------------------------------------------------------- 5. best model + paired tests
BEST = min(TREES, key=lambda n: MET[n]["point"]["log_RMSE"])
BEST_CV = min(TREES, key=lambda n: CV[n]["mean"])
log(f"best tree model on test: {BEST}; best by CV: {BEST_CV}")
cmp_rows = []
for name in ALL_MODELS:
    if name == BEST:
        continue
    r = Mx.paired_comparison(y_test, PRED[BEST], PRED[name], BIDX)
    cmp_rows.append({"best": BEST, "other": name, **r})
CMP = pd.DataFrame(cmp_rows)
CMP["holm_p"], CMP["holm_reject"] = Mx.holm(CMP.wilcoxon_p.values)
CMP.to_csv(OUT / "stat_tests.csv", index=False)
log("paired tests:\n" + CMP.round(4).to_string())

# also LR vs naive, LR vs budget-only (context for "does anything beat the baselines")
extra = []
for a, b in [("Linear Regression", "Naive median"), ("Linear Regression", "Budget-only OLS")]:
    extra.append({"a": a, "b": b, **Mx.paired_comparison(y_test, PRED[a], PRED[b], BIDX)})
pd.DataFrame(extra).to_csv(OUT / "stat_tests_baselines.csv", index=False)

# ---------------------------------------------------------------- 6. seed variability
t = time.time()
seed_rows = []
for name in TREES:
    for s in C.SEEDS:
        p, _ = fit_predict(name, PARAMS[name], MATS, dev_idx, test_idx, seed=s)
        m = Mx.all_metrics(y_test, p, SMEAR[name])
        seed_rows.append({"model": name, "seed": s, **m})
SEEDDF = pd.DataFrame(seed_rows)
SEEDDF.to_csv(OUT / "seed_runs.csv", index=False)
SEEDSUM = SEEDDF.groupby("model")[["log_RMSE", "log_MAE", "log_R2", "usd_MAE_smear"]].agg(["mean", "std"])
SEEDSUM.to_csv(OUT / "seed_variability.csv")
TIMES["seed_runs_s"] = time.time() - t
log("seed variability:\n" + SEEDSUM.round(4).to_string())

# ---------------------------------------------------------------- 7. ablation (best model, fixed hyperparameters)
t = time.time()
variants = {
    "Full model": dict(encoding="history"),
    "Without budget": dict(encoding="history", drop=F.BUDGET_COLS),
    "Without past box-office features": dict(encoding="none"),
    "A2 frequency encoding instead of past box office": dict(encoding="old"),
    "Without release timing": dict(encoding="history", drop=F.TIMING_COLS),
    "With post-release variables (leakage demo)": dict(encoding="history", leak=True),
}
ABL, ABL_PRED = [], {}
for vname, kw in variants.items():
    mats = F.transform(POP, VOCAB, **kw)
    for name in [BEST, "Linear Regression"]:
        p, _ = fit_predict(name, PARAMS.get(name), mats, dev_idx, test_idx)
        oof, _ = oof_predictions(name, PARAMS.get(name), mats, dev_idx, FOLDS)
        ok = ~np.isnan(oof)
        s = Mx.smearing_factor(y[dev_idx][ok] - oof[ok])
        m = Mx.all_metrics(y_test, p, s)
        ci = Mx.bootstrap_ci(y_test, p, s, BIDX)
        row = {"variant": vname, "model": name, "n_features": mats[0].shape[1] if name != "CatBoost" else mats[1].shape[1],
               "smear": s, **m, "log_RMSE_lo": ci["log_RMSE"][0], "log_RMSE_hi": ci["log_RMSE"][1],
               "log_MAE_lo": ci["log_MAE"][0], "log_MAE_hi": ci["log_MAE"][1]}
        if vname != "Full model":
            ref = PRED[name]
            r = Mx.paired_comparison(y_test, p, ref, BIDX)
            row.update({"d_log_RMSE_vs_full": r["d_log_RMSE"], "d_lo": r["d_log_RMSE_lo"], "d_hi": r["d_log_RMSE_hi"],
                        "wilcoxon_p_vs_full": r["wilcoxon_p"]})
        ABL.append(row)
        ABL_PRED[(vname, name)] = p
        log(f"ablation {vname} / {name}: log RMSE {m['log_RMSE']:.4f}")
ABLDF = pd.DataFrame(ABL)
mask = (ABLDF.variant != "Full model") & (ABLDF.model == BEST) & ~ABLDF.variant.str.contains("leakage")
ABLDF.loc[mask, "holm_p"] = Mx.holm(ABLDF.loc[mask, "wilcoxon_p_vs_full"].values)[0]
ABLDF.to_csv(OUT / "ablation.csv", index=False)
TIMES["ablation_s"] = time.time() - t

# ---------------------------------------------------------------- 8. IPW selection sensitivity
t = time.time()
last_dev = DEV.release_date.max()
SELPOP = ALL[ALL.release_date <= last_dev].copy()
SELPOP["recorded"] = SELPOP.revenue_ok.astype(int)
Xs, _, _ = F.transform(SELPOP, VOCAB, "history")
Xs_test, _, _ = F.transform(TEST, VOCAB, "history")
selm = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                 ("lr", LogisticRegression(C=1.0, max_iter=5000))])
selm.fit(Xs, SELPOP.recorded)
auc_in = roc_auc_score(SELPOP.recorded, selm.predict_proba(Xs)[:, 1])
# out-of-time check of the selection model on test-period films (recorded or not)
TESTPER = ALL[ALL.release_date > last_dev]
Xtp, _, _ = F.transform(TESTPER, VOCAB, "history")
auc_out = roc_auc_score(TESTPER.revenue_ok.astype(int), selm.predict_proba(Xtp)[:, 1])
p_rec = np.clip(selm.predict_proba(Xs_test)[:, 1], 0.05, 1.0)
W = 1.0 / p_rec
W = W / W.mean()
ess = W.sum() ** 2 / np.sum(W ** 2)
IPW = []
for name in ALL_MODELS:
    mu = Mx.all_metrics(y_test, PRED[name], SMEAR[name])
    mw = Mx.all_metrics(y_test, PRED[name], SMEAR[name], w=W)
    IPW.append({"model": name, **{f"unw_{k}": v for k, v in mu.items()}, **{f"ipw_{k}": v for k, v in mw.items()}})
IPWDF = pd.DataFrame(IPW)
IPWDF.to_csv(OUT / "ipw_metrics.csv", index=False)
IPWINFO = {"n_selection_sample": int(len(SELPOP)), "n_recorded": int(SELPOP.recorded.sum()),
           "share_recorded_dev_period": float(SELPOP.recorded.mean()),
           "n_test_period_all": int(len(TESTPER)), "share_recorded_test_period": float(TESTPER.revenue_ok.mean()),
           "auc_in_sample": float(auc_in), "auc_test_period": float(auc_out),
           "p_recorded_test_min": float(p_rec.min()), "p_recorded_test_median": float(np.median(p_rec)),
           "weight_min": float(W.min()), "weight_max": float(W.max()), "effective_sample_size": float(ess),
           "n_clipped_at_0.05": int((selm.predict_proba(Xs_test)[:, 1] < 0.05).sum())}
dump(IPWINFO, "ipw_info.json")
TIMES["ipw_s"] = time.time() - t
log(f"IPW: {IPWINFO}")

# ---------------------------------------------------------------- 9. per-group errors + per-year
G = TEST.copy()
G["budget_group"] = np.where(G.budget_clean.isna(), "Budget missing", "")
known = G.budget_clean.notna()
G.loc[known, "budget_group"] = pd.qcut(G.loc[known, "budget_clean"], 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
G["language_group"] = np.where(G.original_language == "en", "English", "Non-English")
G["year"] = G.release_date.dt.year
G["director_history"] = np.where(G.dir_hist_missing == 1, "No director history", "Director history")
grp_rows = []
budget_q_edges = pd.qcut(G.loc[known, "budget_clean"], 5, retbins=True)[1].tolist()
for gcol in ["budget_group", "language_group", "year", "director_history"]:
    for gval, sub in G.groupby(gcol):
        ii = np.searchsorted(test_idx, sub.index.to_numpy())  # positions within test arrays
        for name in ALL_MODELS:
            e = y_test[ii] - PRED[name][ii]
            grp_rows.append({"grouping": gcol, "group": str(gval), "n": int(len(ii)), "model": name,
                             "log_MAE": float(np.mean(np.abs(e))), "log_RMSE": float(np.sqrt(np.mean(e ** 2))),
                             "mean_signed_error": float(np.mean(e)),
                             "median_actual_usd_m": float(np.median(np.expm1(y_test[ii])) / 1e6)})
GRP = pd.DataFrame(grp_rows)
GRP.to_csv(OUT / "group_errors.csv", index=False)
dump({"budget_quintile_edges_usd": budget_q_edges}, "group_info.json")

# ---------------------------------------------------------------- 10. alternative 70/30 split
t = time.time()
IS_DEV70 = D.split_at_fraction(POP, 0.70)
V70 = F.fit_vocab(POP[IS_DEV70])
mats70 = F.transform(POP, V70, "history")
d70, t70 = np.where(IS_DEV70)[0], np.where(~IS_DEV70)[0]
folds70 = list(TimeSeriesSplit(C.N_CV_FOLDS).split(d70))
R70 = []
B70 = Mx.bootstrap_indices(len(t70))
P70 = {}
for name in ALL_MODELS:
    p, _ = fit_predict(name, PARAMS.get(name), mats70, d70, t70)
    oof, _ = oof_predictions(name, PARAMS.get(name), mats70, d70, folds70)
    ok = ~np.isnan(oof)
    s = Mx.smearing_factor(y[d70][ok] - oof[ok])
    m = Mx.all_metrics(y[t70], p, s)
    ci = Mx.bootstrap_ci(y[t70], p, s, B70)
    P70[name] = p
    R70.append({"model": name, "smear": s, **m, "log_RMSE_lo": ci["log_RMSE"][0], "log_RMSE_hi": ci["log_RMSE"][1]})
R70 = pd.DataFrame(R70)
R70.to_csv(OUT / "split70_metrics.csv", index=False)
best70 = min(TREES, key=lambda n: R70.set_index("model").loc[n, "log_RMSE"])
c70 = []
for name in ALL_MODELS:
    if name != BEST:
        c70.append({"other": name, **Mx.paired_comparison(y[t70], P70[BEST], P70[name], B70)})
C70 = pd.DataFrame(c70)
C70["holm_p"], C70["holm_reject"] = Mx.holm(C70.wilcoxon_p.values)
C70.to_csv(OUT / "split70_tests.csv", index=False)
dump({"n_dev": int(len(d70)), "n_test": int(len(t70)), "dev_end": str(POP.release_date.iloc[d70[-1]].date()),
      "test_start": str(POP.release_date.iloc[t70[0]].date()), "best_tree_70": best70}, "split70_info.json")
TIMES["split70_s"] = time.time() - t
log("70/30 split:\n" + R70[["model", "log_MAE", "log_RMSE", "log_R2", "usd_MAE_smear"]].round(4).to_string())

# ---------------------------------------------------------------- 11. error analysis
E = pd.DataFrame({"id": TEST.id.values, "title": TEST.title.values, "date": TEST.release_date.dt.date.values,
                  "actual_usd_m": TEST.revenue.values / 1e6, "budget_usd_m": TEST.budget_clean.values / 1e6,
                  "y_log": y_test, "pred_log": PRED[BEST]})
E["pred_usd_m"] = Mx.to_dollars(E.pred_log.values, SMEAR[BEST]) / 1e6
E["log_error"] = E.y_log - E.pred_log  # positive = under-prediction
E.to_csv(OUT / "errors_best_model.csv", index=False)
under = E.sort_values("log_error", ascending=False).head(10)
over = E.sort_values("log_error").head(10)
under.to_csv(OUT / "worst_under.csv", index=False)
over.to_csv(OUT / "worst_over.csv", index=False)
lr_e = y_test - PRED["Linear Regression"]
low = E.actual_usd_m < 1
bands = pd.cut(E.actual_usd_m, [0, 1, 10, 50, 100, 300, 1e6],
               labels=["<1", "1-10", "10-50", "50-100", "100-300", ">300"], right=False)
band_tab = E.groupby(bands, observed=False).agg(n=("log_error", "size"), mean_signed=("log_error", "mean"),
                                                mae=("log_error", lambda s: np.mean(np.abs(s))))
band_tab.to_csv(OUT / "error_by_actual_band.csv")
E["pred_decile"] = pd.qcut(E.pred_log, 10, labels=False) + 1
dec = E.groupby("pred_decile").agg(n=("y_log", "size"), mean_pred=("pred_log", "mean"), mean_actual=("y_log", "mean"),
                                   median_actual=("y_log", "median"))
dec.to_csv(OUT / "calibration_deciles.csv")
ERR = {
    "best_model": BEST,
    "share_abs_log_error_gt1": float(np.mean(np.abs(E.log_error) > 1)),
    "share_abs_log_error_gt1_LR": float(np.mean(np.abs(lr_e) > 1)),
    "share_abs_log_error_gt2": float(np.mean(np.abs(E.log_error) > 2)),
    "share_abs_log_error_lt05": float(np.mean(np.abs(E.log_error) < 0.5)),
    "n_actual_below_1M": int(low.sum()),
    "mean_signed_error_below_1M": float(E.log_error[low].mean()),
    "mean_abs_error_below_1M": float(E.log_error[low].abs().mean()),
    "mean_abs_error_at_least_1M": float(E.log_error[~low].abs().mean()),
    "share_of_squared_error_from_below_1M": float((E.log_error[low] ** 2).sum() / (E.log_error ** 2).sum()),
    "log_RMSE_excluding_below_1M": float(np.sqrt(np.mean(E.log_error[~low] ** 2))),
    "log_RMSE_excluding_below_1M_LR": float(np.sqrt(np.mean(lr_e[~low.values] ** 2))),
    "mean_signed_error_all": float(E.log_error.mean()),
    "median_ratio_pred_to_actual": float(np.median(E.pred_usd_m / E.actual_usd_m)),
    "corr_error_actual": float(np.corrcoef(E.log_error, E.y_log)[0, 1]),
    "top10_actual_share_of_usd_sq_error": float(
        np.sort((E.actual_usd_m - E.pred_usd_m) ** 2)[::-1][:10].sum() / ((E.actual_usd_m - E.pred_usd_m) ** 2).sum()),
}
dump(ERR, "error_analysis.json")
log(f"error analysis: {ERR}")


# ---------------------------------------------------------------- 12. SHAP (best tree model)
def shap_values(name, est, X):
    if name == "LightGBM":
        return est.booster_.predict(X, pred_contrib=True)
    if name == "XGBoost":
        return est.get_booster().predict(M.xgb.DMatrix(X), pred_contribs=True)
    if name == "CatBoost":
        return est.shap_values(X)
    raise ValueError("Random Forest SHAP not implemented natively")


t = time.time()
Xbest = Xc if BEST in M.USES_CB_MATRIX else Xt
_, est_best = fit_predict(BEST, PARAMS[BEST], MATS, dev_idx, test_idx)
SV = shap_values(BEST, est_best, Xbest.iloc[test_idx])
phi, base = SV[:, :-1], SV[:, -1]
recon = phi.sum(1) + base
pbest = est_best.predict(Xbest.iloc[test_idx])
glob = pd.DataFrame({"feature": Xbest.columns, "mean_abs_shap": np.abs(phi).mean(0)}).sort_values(
    "mean_abs_shap", ascending=False)
glob["share"] = glob.mean_abs_shap / glob.mean_abs_shap.sum()
glob.to_csv(OUT / "shap_global.csv", index=False)
np.save(OUT / "shap_values_test.npy", phi)
loc = {}
for tag, row in (("over", over.iloc[0]), ("under", under.iloc[0])):
    i = int(np.where(TEST.id.values == row.id)[0][0])
    contrib = pd.Series(phi[i], index=Xbest.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    loc[tag] = {"title": row.title, "date": str(row.date), "actual_usd_m": float(row.actual_usd_m),
                "pred_usd_m": float(row.pred_usd_m), "budget_usd_m": None if np.isnan(row.budget_usd_m) else float(row.budget_usd_m),
                "y_log": float(row.y_log), "pred_log": float(row.pred_log), "base_value": float(base[i]),
                "top_contributions": [{"feature": f, "shap": float(v), "value": (None if pd.isna(Xbest.iloc[test_idx[i]][f]) else
                                       (Xbest.iloc[test_idx[i]][f] if isinstance(Xbest.iloc[test_idx[i]][f], str) else float(Xbest.iloc[test_idx[i]][f])))}
                                      for f, v in top.items()],
                "sum_other_features": float(contrib.sum() - top.sum())}
SHAPINFO = {"model": BEST, "base_value": float(base[0]), "max_abs_reconstruction_error": float(np.max(np.abs(recon - pbest))),
            "local": loc, "time_s": time.time() - t}
dump(SHAPINFO, "shap_info.json")
log(f"SHAP top 15:\n{glob.head(15).round(4).to_string()}")

# ---------------------------------------------------------------- save state
TIMES["total_s"] = time.time() - T0
dump(TIMES, "timings.json")
pd.to_pickle({"POP": POP, "IS_DEV": IS_DEV, "test_idx": test_idx, "dev_idx": dev_idx, "y": y, "PRED": PRED,
              "SMEAR": SMEAR, "BEST": BEST, "BEST_CV": BEST_CV, "OOF": OOF, "Xbest_cols": list(Xbest.columns),
              "Xbest_test": Xbest.iloc[test_idx], "phi": phi, "CV": CV, "PARAMS": PARAMS, "W_ipw": W,
              "ABL_PRED": ABL_PRED}, OUT / "state.pkl")
log(f"done. total {TIMES['total_s']/60:.1f} min")
