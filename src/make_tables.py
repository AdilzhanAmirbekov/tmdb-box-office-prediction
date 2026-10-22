"""LaTeX tables (booktabs, IEEE style: caption above) generated from outputs/."""
import json

import numpy as np
import pandas as pd

from . import config as C

OUT, TAB = C.OUT_DIR, C.TABLE_DIR
S = pd.read_pickle(OUT / "state.pkl")
BEST = S["BEST"]
NAIVE_COLS = True
TABSEP = 2.6
ORDER = ["Naive median", "Budget-only OLS", "Linear Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]


def J(name):
    return json.load(open(OUT / name))


def esc(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def f(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    x = 0.0 if round(x, d) == 0 else x
    s = f"{x:,.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi, d=3):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def note(text):
    return "\\par\\vspace{2pt}{\\footnotesize\\raggedright Note: " + text + "\\par}"


def write(name, body):
    (TAB / name).write_text(body)


# ------------------------------------------------------------------ Table: data actually used
def table_data():
    d = J("data_summary.json")
    td, tt = d["target_dev"], d["target_test"]
    folds = d["cv_folds"]
    pop = S["POP"]
    dev = pop[S["IS_DEV"]]
    test = pop[~S["IS_DEV"]]
    rows = [
        ("Source", r"TMDB 5000 Movie Dataset \cite{tmdb5000dataset}"),
        ("Raw records", f"{d['n_movies_raw']:,} films; credits joined on \\texttt{{id}}"),
        ("Removed: no release date", f"{d['n_missing_release_date']}"),
        ("Removed: revenue $=0$ (not recorded)", f"{d['n_revenue_zero']:,}"),
        ("Removed: $0<$ revenue $<$ USD 1,000 (entry error)", f"{d['n_revenue_tiny']}"),
        ("Modelling sample", f"{d['n_model_population']:,} films released {d['dev_date_min']} to {d['test_date_max']}"),
        ("Budget set to missing ($0$ or $<$ USD 1,000)", f"{d['n_budget_missing_in_pop']} ({d['budget_missing_dev']} dev., {d['budget_missing_test']} test)"),
        ("Development set (earliest 80\\%)", f"{d['n_dev']:,} films, {d['dev_date_min']} to {d['dev_date_max']}"),
        ("Tuning CV (5 expanding folds)", f"train {folds[0]['train_n']}--{folds[-1]['train_n']:,}; valid. {folds[0]['valid_n']} each"),
        ("Test set (latest 20\\%)", f"{d['n_test']:,} films, {d['test_date_min']} to {d['test_date_max']}"),
        ("Revenue, dev.: median / mean (USD M)", f"{f(td['usd_median'],1)} / {f(td['usd_mean'],1)}"),
        ("Revenue, test: median / mean (USD M)", f"{f(tt['usd_median'],1)} / {f(tt['usd_mean'],1)}"),
        ("Revenue, test: min / max", f"USD {tt['usd_min']*1e6:,.0f} / USD {f(tt['usd_max'],1)} M"),
        ("$\\log(1+y)$, dev. / test: mean (SD)", f"{f(td['log_mean'],2)} ({f(td['log_sd'],2)}) / {f(tt['log_mean'],2)} ({f(tt['log_sd'],2)})"),
        ("Films below USD 1 M, dev. / test", f"{100*td['share_below_1M']:.1f}\\% / {100*tt['share_below_1M']:.1f}\\%"),
        ("Predictors (one-hot / CatBoost)", f"{d['n_features_tree']} / {d['n_features_catboost']}"),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    write("tab_data.tex", rf"""\begin{{table}}[!t]
\centering
\caption{{Data Actually Used in the Experiment}}
\label{{tab:data}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.50\columnwidth}}>{{\raggedright\arraybackslash}}p{{0.47\columnwidth}}@{{}}}}
\toprule
Item & Value \\
\midrule
{body}
\bottomrule
\end{{tabular}}
{note("Revenue is worldwide nominal revenue in U.S. dollars; dev.\\ = development set; M = million.")}
\end{{table}}
""")


# ------------------------------------------------------------------ Table: main results
def table_main():
    m = pd.read_csv(OUT / "metrics_test.csv").set_index("model").loc[ORDER]
    lines = []
    for name, r in m.iterrows():
        nm = name + (r"$^{\dagger}$" if name == BEST else "")
        top = [nm, f"{f(r.cv_mean)} $\\pm$ {f(r.cv_std)}",
               f(r.log_MAE), f(r.log_RMSE), f(r.log_R2),
               f(r.usd_MAE_smear, 1), f(r.usd_RMSE_smear, 1), f(r.usd_R2_smear), f(r.usd_MedAE_smear, 1),
               f(r.smear, 2)] + ([f(r.usd_MAE_naive, 1), f(r.usd_R2_naive)] if NAIVE_COLS else [])
        bot = ["", "", ci(r.log_MAE_lo, r.log_MAE_hi), ci(r.log_RMSE_lo, r.log_RMSE_hi), ci(r.log_R2_lo, r.log_R2_hi),
               ci(r.usd_MAE_smear_lo, r.usd_MAE_smear_hi, 1), ci(r.usd_RMSE_smear_lo, r.usd_RMSE_smear_hi, 1),
               ci(r.usd_R2_smear_lo, r.usd_R2_smear_hi), ci(r.usd_MedAE_smear_lo, r.usd_MedAE_smear_hi, 1),
               ""] + (["", ""] if NAIVE_COLS else [])
        lines.append(" & ".join(top) + r" \\")
        lines.append(" & ".join(b for b in bot) + r" \\[2pt]")
        if name in ("Linear Regression",):
            lines.append(r"\midrule")
    body = "\n".join(lines)
    d = J('data_summary.json')
    ncol = 10 + (2 if NAIVE_COLS else 0)
    naive_head = r" & \multicolumn{2}{c}{Naive $e^{\hat y^*}-1$}" if NAIVE_COLS else ""
    naive_rule = r"\cmidrule(lr){11-12}" if NAIVE_COLS else ""
    naive_cols = r" & MAE & $R^2$" if NAIVE_COLS else ""
    naive_note = (" Naive: dollar MAE and $R^2$ with the back-transformation $e^{\\hat y^*}-1$ (no smearing)."
                  if NAIVE_COLS else
                  " Dollar metrics for the naive back-transformation $e^{\\hat y^*}-1$ are given in the text and in the repository.")
    write("tab_main.tex", rf"""\begin{{table*}}[!t]
\centering
\caption{{Test-Set Performance of All Models on the Same {len(S['test_idx'])} Chronologically Held-Out Films}}
\label{{tab:main}}
\scriptsize
\setlength{{\tabcolsep}}{{{TABSEP}pt}}
\begin{{tabular}}{{@{{}}l{'c' * (ncol - 1)}@{{}}}}
\toprule
 & & \multicolumn{{3}}{{c}}{{Log scale}} & \multicolumn{{4}}{{c}}{{USD million, smearing back-transform}} & {naive_head} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-9}}{naive_rule}
Model & CV log RMSE & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ & MedAE & $S${naive_cols} \\
\midrule
{{body}}
\bottomrule
\end{{tabular}}
{note(f"Test films released {d['test_date_min']} to {d['test_date_max']}. Log-scale metrics refer to $y^*=\\log(1+y)$; test log RMSE is the primary ranking criterion. Brackets: 95\\% percentile bootstrap confidence intervals ($B={C.BOOTSTRAP_B}$, resampling test films). CV: mean $\\pm$ standard deviation of log RMSE over the five expanding-window validation folds of the development set. MedAE: median absolute error. $S$: Duan smearing factor from out-of-fold development residuals." + naive_note + " $^{\\dagger}$Lowest test log RMSE among the tree-based ensembles.")}
\end{{table*}}
""".replace("{body}", body))


# ------------------------------------------------------------------ Table: tests, seeds, robustness
def table_uncertainty():
    t = pd.read_csv(OUT / "stat_tests.csv").set_index("other")
    seeds = pd.read_csv(OUT / "seed_runs.csv").groupby("model").log_RMSE.agg(["mean", "std"])
    ipw = pd.read_csv(OUT / "ipw_metrics.csv").set_index("model")
    s70 = pd.read_csv(OUT / "split70_metrics.csv").set_index("model")
    lines = []
    for name in ORDER:
        if name == BEST:
            cmp_ = [r"\multicolumn{4}{c}{reference model}"]
        else:
            r = t.loc[name]
            cmp_ = [f"{f(r.d_log_RMSE)} {ci(r.d_log_RMSE_lo, r.d_log_RMSE_hi)}",
                    f"{f(r.d_log_MAE)} {ci(r.d_log_MAE_lo, r.d_log_MAE_hi)}",
                    pv(r.wilcoxon_p), pv(r.holm_p) + ("*" if r.holm_reject else "")]
        sd = f"{f(seeds.loc[name,'mean'])} $\\pm$ {f(seeds.loc[name,'std'])}" if name in seeds.index else "--"
        i = ipw.loc[name]
        s = s70.loc[name]
        lines.append(" & ".join([name] + cmp_ + [sd, f(i.unw_log_RMSE), f(i.ipw_log_RMSE), f(i.ipw_log_MAE),
                                                f(s.log_RMSE), f(s.log_MAE)]) + r" \\")
    info = J("ipw_info.json")
    i70 = J("split70_info.json")
    overlap = i70["n_test"] - len(S["test_idx"])  # 70/30 test films that were in the 80/20 development (tuning) data
    write("tab_uncertainty.tex", rf"""\begin{{table*}}[!t]
\centering
\caption{{Paired Comparison With {BEST}, Seed Variability and Selection and Split Sensitivity}}
\label{{tab:uncertainty}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lcccccccccc@{{}}}}
\toprule
 & \multicolumn{{4}}{{c}}{{Paired comparison with {BEST} (80/20 test set)}} & Seeds & \multicolumn{{3}}{{c}}{{Selection sensitivity (log)}} & \multicolumn{{2}}{{c}}{{70/30 split (log)}} \\
\cmidrule(lr){{2-5}}\cmidrule(lr){{6-6}}\cmidrule(lr){{7-9}}\cmidrule(lr){{10-11}}
Model & $\Delta$ log RMSE [95\% CI] & $\Delta$ log MAE [95\% CI] & $p_W$ & $p_{{\mathrm{{Holm}}}}$ & log RMSE & RMSE & IPW RMSE & IPW MAE & RMSE & MAE \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
{note(f"$\\Delta$: metric of {BEST} minus metric of the other model on the same test films (negative = lower error for {BEST}), with paired bootstrap 95\\% CI. $p_W$: two-sided Wilcoxon signed-rank test on per-film absolute log errors; $p_{{\\mathrm{{Holm}}}}$: Holm-adjusted over the six comparisons (* = $p_{{\\mathrm{{Holm}}}}<0.05$). Seeds: mean $\\pm$ SD of test log RMSE over seeds 42--46 at fixed hyperparameters. IPW: test metrics weighted by the inverse estimated probability that revenue is recorded (effective sample size {info['effective_sample_size']:.0f} of {len(S['test_idx'])}); no intervals or tests were computed for them. 70/30: models refitted with the hyperparameters tuned on the 80\\% development set, trained on the earliest 70\\% of films and tested on the latest 30\\% ($n={i70['n_test']:,}$); {overlap} of these films were in the tuning data of the 80/20 analysis, so this split is a stability check rather than an independent test.")}
\end{{table*}}
""")


# ------------------------------------------------------------------ Table: ablation + leakage
def table_ablation():
    a = pd.read_csv(OUT / "ablation.csv")
    b = a[a.model == BEST].set_index("variant")
    lr = a[a.model == "Linear Regression"].set_index("variant")
    labels = {
        "Full model": "Full feature set",
        "Without budget": "Without budget",
        "Without past box-office features": "Without past box office",
        "A2 frequency encoding instead of past box office": r"A2 $\geq$10-film encoding",
        "Without release timing": "Without release timing",
        "With post-release variables (leakage demo)": r"Plus post-release vars.",
    }
    lines = []
    for v, lab in labels.items():
        r = b.loc[v]
        if v == "Full model":
            d = "--"
            p = "--"
        else:
            d = f"{f(r.d_log_RMSE_vs_full)} {ci(r.d_lo, r.d_hi)}"
            p = pv(r.holm_p) if not np.isnan(r.get("holm_p", np.nan)) else pv(r.wilcoxon_p_vs_full) + r"$^{\ddagger}$"
        lines.append(" & ".join([lab, f(r.log_RMSE), f(r.log_MAE), d, p, f(lr.loc[v].log_RMSE)]) + r" \\")
        if v == "Without release timing":
            lines.append(r"\midrule")
    write("tab_ablation.tex", rf"""\begin{{table}}[!t]
\centering
\caption{{Ablation and Leakage Check for {BEST}}}
\label{{tab:ablation}}
\scriptsize
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
Feature set & RMSE & MAE & $\Delta$ RMSE [95\% CI] & $p$ & LR \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
{note(f"Fixed hyperparameters, same 671 test films. RMSE and MAE: test log-scale errors. $\\Delta$: change in test log RMSE relative to the full feature set with paired bootstrap 95\\% CI; $p$: Holm-adjusted Wilcoxon signed-rank $p$ over the four ablations. The last row adds post-release variables that are excluded from the study and is shown only to quantify leakage. LR: Linear Regression log RMSE with the same feature set. $^{{\\ddagger}}$Unadjusted.")}
\end{{table}}
""")


# ------------------------------------------------------------------ Table: worst cases
def table_worst(k=5):
    u = pd.read_csv(OUT / "worst_under.csv").head(k)
    o = pd.read_csv(OUT / "worst_over.csv").head(k)
    lines = [r"\multicolumn{5}{@{}l}{\emph{Largest under-predictions (actual $>$ predicted)}} \\"]
    for _, r in u.iterrows():
        lines.append(f"{esc(r.title)} ({str(r.date)[:4]}) & {f(r.actual_usd_m, 3)} & {f(r.pred_usd_m, 1)} & "
                     f"{f(r.budget_usd_m, 1)} & {f(r.log_error, 2)} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{Largest over-predictions (actual $<$ predicted)}} \\")
    for _, r in o.iterrows():
        lines.append(f"{esc(r.title)} ({str(r.date)[:4]}) & {f(r.actual_usd_m, 3)} & {f(r.pred_usd_m, 1)} & "
                     f"{f(r.budget_usd_m, 1)} & {f(r.log_error, 2)} \\\\")
    write("tab_worst.tex", rf"""\begin{{table}}[!t]
\centering
\caption{{Test Films With the Largest {BEST} Errors on the Log Scale}}
\label{{tab:worst}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.44\columnwidth}}rrrr@{{}}}}
\toprule
Film (year) & Actual & Predicted & Budget & Error \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
{note("Actual revenue, predicted revenue and budget in USD million; predicted revenue uses the smearing back-transform; error $=y^*-\\hat y^*$; -- = budget not recorded.")}
\end{{table}}
""")


# ------------------------------------------------------------------ Table (Methods): search space + selected values
def table_hyper():
    P = J("hyperparameters.json")
    cv = J("cv_results.json")
    space = {
        "Random Forest": [("n\\_estimators", "\\{200, 300, 500\\}", "n_estimators", 0),
                          ("max\\_depth", "\\{None, 8, 12, 16, 20\\}", "max_depth", 0),
                          ("min\\_samples\\_split", "\\{2, 5, 10, 20\\}", "min_samples_split", 0),
                          ("min\\_samples\\_leaf", "\\{1, 2, 4, 8\\}", "min_samples_leaf", 0),
                          ("max\\_features", "\\{0.2, 0.33, 0.5, 0.7, 1.0\\}", "max_features", 2)],
        "XGBoost": [("n\\_estimators", "200--1000", "n_estimators", 0), ("learning\\_rate", "0.01--0.2 (log)", "learning_rate", 3),
                    ("max\\_depth", "2--8", "max_depth", 0), ("min\\_child\\_weight", "\\{1, 3, 5, 10\\}", "min_child_weight", 0),
                    ("subsample", "0.6--1.0", "subsample", 2), ("colsample\\_bytree", "0.5--1.0", "colsample_bytree", 2),
                    ("reg\\_lambda", "0.1--10 (log)", "reg_lambda", 3), ("reg\\_alpha", "0.001--1 (log)", "reg_alpha", 3)],
        "LightGBM": [("n\\_estimators", "200--1000", "n_estimators", 0), ("learning\\_rate", "0.01--0.2 (log)", "learning_rate", 3),
                     ("num\\_leaves", "\\{7, 15, 31, 63\\}", "num_leaves", 0), ("max\\_depth", "\\{$-$1, 4, 6, 8\\}", "max_depth", 0),
                     ("min\\_child\\_samples", "\\{5, 10, 20, 40\\}", "min_child_samples", 0),
                     ("subsample", "0.6--1.0", "subsample", 2), ("colsample\\_bytree", "0.5--1.0", "colsample_bytree", 2),
                     ("reg\\_lambda", "0.001--10 (log)", "reg_lambda", 3), ("reg\\_alpha", "0.001--1 (log)", "reg_alpha", 3)],
        "CatBoost": [("iterations", "\\{300, 500, 800, 1000\\}", "iterations", 0), ("learning\\_rate", "0.02--0.2 (log)", "learning_rate", 3),
                     ("depth", "4--8", "depth", 0), ("l2\\_leaf\\_reg", "1--10 (log)", "l2_leaf_reg", 2),
                     ("subsample", "0.6--1.0", "subsample", 2), ("rsm", "0.5--1.0", "rsm", 2),
                     ("random\\_strength", "0.1--10 (log)", "random_strength", 2)],
    }
    lines = []
    for m, rows in space.items():
        lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{m}}} ({C.N_ITER[m]} candidates, tuning time {cv[m]['tuning_time_s']/60:.1f} min)}} \\")
        for lab, rng, key, d in rows:
            v = P[m][key]
            vs = "None" if v is None else (f"{v:.{d}f}" if d else str(int(v)))
            lines.append(f"\\quad \\texttt{{{lab}}} & {rng} & {vs} \\\\")
    write("tab_hyper.tex", rf"""\begin{{table}}[!t]
\centering
\caption{{Randomised Search Spaces and Selected Hyperparameters}}
\label{{tab:hyper}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}lll@{{}}}}
\toprule
Hyperparameter & Search space & Selected \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
{note(f"Selection by expanding-window CV on the development set, objective log RMSE, seed {C.SEED}. Ranges marked (log) were sampled log-uniformly, other ranges uniformly. XGBoost and LightGBM share three sampled values (\\texttt{{n\\_estimators}}, \\texttt{{learning\\_rate}}, \\texttt{{colsample\\_bytree}}) because both searches used the same seed and the same distributions for these parameters.")}
\end{{table}}
""")


if __name__ == "__main__":
    table_hyper()
    table_data()
    table_main()
    table_uncertainty()
    table_ablation()
    table_worst()
    print("tables written to", TAB)
