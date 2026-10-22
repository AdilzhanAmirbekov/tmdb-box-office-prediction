"""Vector figures for the Results section (IEEE single-column width, 8 pt serif)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from . import config as C  # noqa: E402

OUT, FIG = C.OUT_DIR, C.FIG_DIR
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})
# Okabe-Ito colour-blind safe colours; paired with different markers for greyscale printing
BLUE, VERM, GREY, GREEN, ORANGE = "#0072B2", "#D55E00", "#7f7f7f", "#009E73", "#E69F00"
COL = 3.5

S = pd.read_pickle(OUT / "state.pkl")
BEST = S["BEST"]
y = S["y"][S["test_idx"]]

NICE = {
    "log_budget": "Budget (log USD)", "budget_missing": "Budget missing", "runtime": "Runtime (min)",
    "release_year": "Release year", "release_month": "Release month", "release_dow": "Release weekday",
    "n_companies": "No. of production companies", "n_countries": "No. of production countries",
    "dir_prior_n": "Director: no. of prior films", "dir_prior_logrev": "Director: prior mean log revenue",
    "dir_hist_missing": "Director: no revenue history", "cast_prior_n_total": "Top-3 cast: no. of prior films",
    "cast_prior_logrev_mean": "Top-3 cast: prior mean log rev. (mean)",
    "cast_prior_logrev_max": "Top-3 cast: prior mean log rev. (max)", "cast_hist_missing": "Top-3 cast: no history",
    "comp_prior_n_max": "Company: no. of prior films (max)", "comp_prior_logrev_max": "Company: prior mean log rev. (max)",
    "comp_hist_missing": "Company: no revenue history", "language": "Original language",
}


def nice(f):
    if f in NICE:
        return NICE[f]
    for pre, lab in (("genre_", "Genre: "), ("company_", "Company: "), ("country_", "Country: "), ("lang_", "Language: ")):
        if f.startswith(pre):
            return lab + f[len(pre):].replace("_", " ")
    return f


def usd_axis(ax, which="both"):
    ticks = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    lab = {1e3: "$10^3$", 1e4: "$10^4$", 1e5: "$10^5$", 1e6: "$10^6$", 1e7: "$10^7$", 1e8: "$10^8$", 1e9: "$10^9$", 1e10: "$10^{10}$"}
    fmt = FuncFormatter(lambda v, p: lab.get(v, ""))
    for a in ([ax.xaxis, ax.yaxis] if which == "both" else [ax.xaxis] if which == "x" else [ax.yaxis]):
        a.set_major_locator(FixedLocator(ticks))
        a.set_major_formatter(fmt)
        a.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1 predicted vs actual
def fig_pred_actual():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.95), sharex=True, sharey=True)
    actual = np.expm1(y)
    lo, hi = 1e3, 5e9
    for ax, name, c, mk, tag in ((axes[0], BEST, BLUE, "o", "(a)"), (axes[1], "Linear Regression", VERM, "s", "(b)")):
        p = np.expm1(S["PRED"][name])
        ax.scatter(actual, p, s=4, c=c, marker=mk, alpha=0.45, linewidths=0, rasterized=False)
        xx = np.array([lo, hi])
        ax.plot(xx, xx, color="black", lw=0.7)
        ax.plot(xx, xx * np.e, color="black", lw=0.5, ls="--")
        ax.plot(xx, xx / np.e, color="black", lw=0.5, ls="--")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
        usd_axis(ax)
        ax.set_aspect("equal")
        rmse = np.sqrt(np.mean((y - S["PRED"][name]) ** 2))
        ax.text(0.04, 0.96, f"{tag} {name}\nlog RMSE = {rmse:.3f}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("Actual revenue (USD, log scale)")
    axes[0].set_ylabel("Predicted revenue (USD, log scale)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_pred_vs_actual")


# ------------------------------------------------------------------ Fig. 2 errors by group
def fig_groups():
    G = pd.read_csv(OUT / "group_errors.csv")
    E = pd.read_csv(OUT / "errors_best_model.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    g = G[G.grouping == "budget_group"]
    order = ["Budget missing", "Q1", "Q2", "Q3", "Q4", "Q5"]
    labels = ["Miss.", "Q1", "Q2", "Q3", "Q4", "Q5"]
    xs = np.arange(len(order))
    for k, (name, c, mk, dx) in enumerate(((BEST, BLUE, "o", -0.18), ("Linear Regression", VERM, "s", 0.0),
                                            ("Naive median", GREY, "^", 0.18))):
        sub = g[g.model == name].set_index("group").reindex(order)
        ax.plot(xs + dx, sub.log_MAE, ls="none", marker=mk, ms=3.5, color=c, label=name)
    n = g[g.model == BEST].set_index("group").reindex(order).n.astype(int)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n{v}" for l, v in zip(labels, n)])
    ax.set_xlabel("(a) Budget quintile ($n$ test films)")
    ax.set_ylabel("Mean absolute error (log USD)")
    ax.set_ylim(0, None)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07),
               handletextpad=0.2, columnspacing=1.0, fontsize=7)

    ax = axes[1]
    ax.scatter(np.expm1(E.y_log), E.log_error, s=4, c=BLUE, alpha=0.45, linewidths=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(1, color="black", lw=0.5, ls="--")
    ax.axhline(-1, color="black", lw=0.5, ls="--")
    # binned mean residual
    bins = np.logspace(3, 10, 15)
    idx = np.digitize(np.expm1(E.y_log), bins)
    mids, means = [], []
    for b in np.unique(idx):
        m = idx == b
        if m.sum() >= 5:
            mids.append(np.exp(np.mean(np.log(np.expm1(E.y_log[m])))))
            means.append(E.log_error[m].mean())
    ax.plot(mids, means, color="black", lw=1.0, marker="D", ms=2.5, label="Binned mean")
    ax.set_xscale("log")
    usd_axis(ax, "x")
    ax.set_xlim(1e3, 5e9)
    ax.set_xlabel("(b) Actual revenue (USD, log scale)")
    ax.set_ylabel("Residual $y^*-\\hat{y}^*$ (log)")
    ax.legend(loc="lower right", handletextpad=0.2, fontsize=6.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_errors")


# ------------------------------------------------------------------ Fig. 3 SHAP beeswarm
def fig_shap(top=15):
    phi = S["phi"]
    X = S["Xbest_test"]
    imp = np.abs(phi).mean(0)
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(COL, 3.25))
    rng = np.random.default_rng(C.SEED)
    cmap = plt.get_cmap("viridis")
    for r, j in enumerate(order[::-1]):
        v = phi[:, j]
        col = X.iloc[:, j]
        if col.dtype == object:
            codes = pd.Series(col).astype("category").cat.codes.to_numpy(dtype=float)
            vals = codes
        else:
            vals = col.to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.any():
            lo_, hi_ = np.nanpercentile(vals[finite], [5, 95])
            if hi_ == lo_:
                lo_, hi_ = np.nanmin(vals[finite]), np.nanmax(vals[finite])
            z = np.clip((vals - lo_) / (hi_ - lo_ if hi_ > lo_ else 1), 0, 1)
        else:
            z = np.zeros_like(vals)
        # simple beeswarm-like jitter scaled by local density
        hist, edges = np.histogram(v, bins=50)
        dens = hist[np.clip(np.digitize(v, edges) - 1, 0, 49)] / hist.max()
        jit = (rng.random(len(v)) - 0.5) * 0.75 * np.sqrt(dens)
        ax.scatter(v[~finite], r + jit[~finite], s=3, c="#bdbdbd", linewidths=0, alpha=0.8)
        ax.scatter(v[finite], r + jit[finite], s=3, c=cmap(z[finite]), linewidths=0, alpha=0.85)
    ax.set_yticks(range(top))
    ax.set_yticklabels([nice(X.columns[j]) for j in order[::-1]], fontsize=6.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("SHAP value (contribution to predicted log revenue)")
    ax.set_ylim(-0.6, top - 0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.035, aspect=30)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value (grey = missing)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=1.5)
    cb.outline.set_linewidth(0.4)
    fig.tight_layout()
    save(fig, "fig_shap_beeswarm")


if __name__ == "__main__":
    fig_pred_actual()
    fig_groups()
    fig_shap()
    print("figures written to", FIG)
