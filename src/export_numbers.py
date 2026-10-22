"""Flatten every result file in outputs/ into outputs/all_numbers.json (key -> {value, source}).
Used for cross-checking numbers cited in the article."""
import json

import pandas as pd

from . import config as C

OUT = C.OUT_DIR
res = {}


def put(prefix, obj, src):
    if isinstance(obj, dict):
        for k, v in obj.items():
            put(f"{prefix}.{k}" if prefix else str(k), v, src)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for i, v in enumerate(obj):
            put(f"{prefix}[{i}]", v, src)
    else:
        res[prefix] = {"value": obj, "source": src}


for f in sorted(OUT.glob("*.json")):
    if f.name == "all_numbers.json":
        continue
    put(f.stem, json.load(open(f)), f"outputs/{f.name}")

for f, key in [("metrics_test.csv", "model"), ("stat_tests.csv", "other"), ("ipw_metrics.csv", "model"),
               ("split70_metrics.csv", "model"), ("split70_tests.csv", "other")]:
    df = pd.read_csv(OUT / f).set_index(key)
    put(f.replace(".csv", ""), json.loads(df.to_json(orient="index")), f"outputs/{f}")
df = pd.read_csv(OUT / "ablation.csv")
df["k"] = df.variant + " | " + df.model
put("ablation", json.loads(df.set_index("k").to_json(orient="index")), "outputs/ablation.csv")
df = pd.read_csv(OUT / "seed_runs.csv").groupby("model")[["log_RMSE", "log_MAE", "log_R2"]].agg(["mean", "std"])
df.columns = [f"{a}_{b}" for a, b in df.columns]
put("seed_variability", json.loads(df.to_json(orient="index")), "outputs/seed_runs.csv")
df = pd.read_csv(OUT / "group_errors.csv")
df["k"] = df.grouping + " | " + df.group.astype(str) + " | " + df.model
put("group_errors", json.loads(df.set_index("k").to_json(orient="index")), "outputs/group_errors.csv")
for f in ["worst_under.csv", "worst_over.csv"]:
    df = pd.read_csv(OUT / f)
    put(f.replace(".csv", ""), json.loads(df.to_json(orient="records")), f"outputs/{f}")
df = pd.read_csv(OUT / "shap_global.csv").head(20)
put("shap_global", json.loads(df.set_index("feature").to_json(orient="index")), "outputs/shap_global.csv")
# SHAP importance aggregated over feature groups (sum of mean |SHAP| of member features)
from .features import HIST_COLS, TIMING_COLS, BUDGET_COLS  # noqa: E402


def group_of(f):
    if f in BUDGET_COLS:
        return "Budget"
    if f in HIST_COLS:
        return "Past box office (director, cast, companies)"
    if f in TIMING_COLS:
        return "Release timing"
    if f.startswith("genre_"):
        return "Genres"
    if f.startswith("company_"):
        return "Production-company indicators"
    if f.startswith("country_"):
        return "Production-country indicators"
    if f in ("language",) or f.startswith("lang_"):
        return "Original language"
    if f == "runtime":
        return "Runtime"
    return "Counts of companies/countries"


g = pd.read_csv(OUT / "shap_global.csv")
g["group"] = g.feature.apply(group_of)
gg = g.groupby("group").agg(mean_abs_shap=("mean_abs_shap", "sum"), share=("share", "sum"),
                            n_features=("feature", "size")).sort_values("mean_abs_shap", ascending=False)
gg.to_csv(OUT / "shap_groups.csv")
put("shap_groups", json.loads(gg.to_json(orient="index")), "outputs/shap_groups.csv")
for f in ["error_by_actual_band.csv", "calibration_deciles.csv"]:
    df = pd.read_csv(OUT / f)
    put(f.replace(".csv", ""), json.loads(df.to_json(orient="records")), f"outputs/{f}")
# extra descriptive numbers used in the text (computed from saved predictions / SHAP values)
import numpy as np  # noqa: E402

pr = pd.read_csv(OUT / "predictions_test.csv")
st = pd.read_pickle(OUT / "state.pkl")
phi, X = st["phi"], st["Xbest_test"]
cols = list(X.columns)
extra = {}
big = pr.revenue_usd > 1e7
extra["n_test_actual_above_10M"] = int(big.sum())
for m in ["CatBoost", "Linear Regression"]:
    e = pr.y_log - pr["pred_log_" + m]
    extra[f"share_within_factor_e_above_10M_{m}"] = float((np.abs(e[big]) < 1).mean())
sh = lambda f: phi[:, cols.index(f)]  # noqa: E731
ry = sh("release_year")
extra.update({"shap_year_min": float(ry.min()), "shap_year_max": float(ry.max()), "shap_year_mean": float(ry.mean()),
              "shap_year_share_negative": float((ry < 0).mean())})
lb, b = X.log_budget.to_numpy(float), sh("log_budget")
ok = ~np.isnan(lb)
extra.update({"shap_budget_min": float(b.min()), "shap_budget_max": float(b.max()),
              "shap_budget_corr_value": float(np.corrcoef(lb[ok], b[ok])[0, 1]), "shap_budget_mean_missing": float(b[~ok].mean())})
c = sh("comp_prior_logrev_max")
extra.update({"shap_comp_logrev_min": float(c.min()), "shap_comp_logrev_max": float(c.max())})
d = sh("genre_Drama")
extra.update({"shap_drama_mean_if_drama": float(d[X.genre_Drama == 1].mean()), "shap_drama_mean_if_not": float(d[X.genre_Drama == 0].mean())})
w = sh("release_dow")
extra["shap_weekday_mean_by_day_mon0"] = [float(w[X.release_dow.to_numpy() == k].mean()) for k in range(7)]
extra["n_by_weekday_mon0"] = [int((X.release_dow.to_numpy() == k).sum()) for k in range(7)]
cn, v = sh("cast_prior_n_total"), X.cast_prior_n_total.to_numpy(float)
q90 = float(np.percentile(v, 90))
extra.update({"shap_castn_corr_value": float(np.corrcoef(v, cn)[0, 1]), "cast_prior_n_total_p90": q90,
              "shap_castn_mean_above_p90": float(cn[v > q90].mean()), "n_above_p90": int((v > q90).sum())})
cv = json.load(open(OUT / "cv_results.json"))
extra["cv_fold_winner"] = [min(cv, key=lambda m: cv[m]["fold_log_rmse"][k]) for k in range(5)]
json.dump(extra, open(OUT / "extra_numbers.json", "w"), indent=1)
put("extra_numbers", extra, "outputs/extra_numbers.json")
json.dump(res, open(OUT / "all_numbers.json", "w"), indent=1)
print(len(res), "numbers exported")
