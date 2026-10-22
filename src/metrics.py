"""Metrics, Duan smearing, bootstrap confidence intervals and paired tests."""
import numpy as np
from scipy.stats import wilcoxon

from . import config as C


def r2(y, p, w=None):
    w = np.ones_like(y) if w is None else w
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2)


def smearing_factor(oof_resid):
    """Duan (1983): S = mean(exp(e_i)) over out-of-fold log residuals e_i = y*_i - yhat*_i."""
    return float(np.mean(np.exp(oof_resid)))


def to_dollars(pred_log, smear=1.0):
    """yhat = exp(yhat*) * S - 1 (S = 1 gives the naive back-transformation)."""
    return np.exp(pred_log) * smear - 1.0


def all_metrics(y_log, p_log, smear, w=None):
    """Log-scale and dollar-scale metrics (dollar values in USD million)."""
    w = np.ones_like(y_log) if w is None else w / np.mean(w)
    e = y_log - p_log
    y_usd = np.expm1(y_log) / 1e6
    out = {
        "log_MAE": np.average(np.abs(e), weights=w),
        "log_RMSE": np.sqrt(np.average(e ** 2, weights=w)),
        "log_R2": r2(y_log, p_log, w),
    }
    for tag, s in (("smear", smear), ("naive", 1.0)):
        p_usd = to_dollars(p_log, s) / 1e6
        d = y_usd - p_usd
        out[f"usd_MAE_{tag}"] = np.average(np.abs(d), weights=w)
        out[f"usd_RMSE_{tag}"] = np.sqrt(np.average(d ** 2, weights=w))
        out[f"usd_R2_{tag}"] = r2(y_usd, p_usd, w)
        out[f"usd_MedAE_{tag}"] = float(np.median(np.abs(d)))
    return {k: float(v) for k, v in out.items()}


def bootstrap_indices(n, B=C.BOOTSTRAP_B, seed=C.SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))


def _vec_metrics(Y, P, Yu, Pu, Pn):
    """Vectorised metrics over bootstrap replicates (rows)."""
    e = Y - P
    sst = np.sum((Y - Y.mean(1, keepdims=True)) ** 2, 1)
    d = Yu - Pu
    sstu = np.sum((Yu - Yu.mean(1, keepdims=True)) ** 2, 1)
    dn = Yu - Pn
    return {
        "log_MAE": np.mean(np.abs(e), 1),
        "log_RMSE": np.sqrt(np.mean(e ** 2, 1)),
        "log_R2": 1 - np.sum(e ** 2, 1) / sst,
        "usd_MAE_smear": np.mean(np.abs(d), 1),
        "usd_RMSE_smear": np.sqrt(np.mean(d ** 2, 1)),
        "usd_R2_smear": 1 - np.sum(d ** 2, 1) / sstu,
        "usd_MedAE_smear": np.median(np.abs(d), 1),
        "usd_MAE_naive": np.mean(np.abs(dn), 1),
        "usd_RMSE_naive": np.sqrt(np.mean(dn ** 2, 1)),
        "usd_R2_naive": 1 - np.sum(dn ** 2, 1) / sstu,
        "usd_MedAE_naive": np.median(np.abs(dn), 1),
    }


def bootstrap_ci(y_log, p_log, smear, idx):
    Y, P = y_log[idx], p_log[idx]
    Yu = np.expm1(Y) / 1e6
    Pu = to_dollars(P, smear) / 1e6
    Pn = to_dollars(P, 1.0) / 1e6
    reps = _vec_metrics(Y, P, Yu, Pu, Pn)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in reps.items()}


def paired_comparison(y_log, p_best, p_other, idx):
    """Difference best - other in log RMSE and log MAE (negative = best has lower error):
    point estimate, paired bootstrap 95% CI, and two-sided Wilcoxon signed-rank test on the
    per-film absolute log errors."""
    ab, ao = np.abs(y_log - p_best), np.abs(y_log - p_other)
    sb, so = (y_log - p_best) ** 2, (y_log - p_other) ** 2
    d_rmse = np.sqrt(sb.mean()) - np.sqrt(so.mean())
    d_mae = ab.mean() - ao.mean()
    rep_rmse = np.sqrt(sb[idx].mean(1)) - np.sqrt(so[idx].mean(1))
    rep_mae = ab[idx].mean(1) - ao[idx].mean(1)
    stat, p = wilcoxon(ab, ao, alternative="two-sided", zero_method="wilcox")
    return {
        "d_log_RMSE": float(d_rmse),
        "d_log_RMSE_lo": float(np.percentile(rep_rmse, 2.5)),
        "d_log_RMSE_hi": float(np.percentile(rep_rmse, 97.5)),
        "d_log_MAE": float(d_mae),
        "d_log_MAE_lo": float(np.percentile(rep_mae, 2.5)),
        "d_log_MAE_hi": float(np.percentile(rep_mae, 97.5)),
        "wilcoxon_W": float(stat),
        "wilcoxon_p": float(p),
        "median_abs_err_diff": float(np.median(ab - ao)),
    }


def holm(pvals, alpha=C.ALPHA):
    """Holm (1979) step-down adjusted p-values and rejection decisions."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj, adj < alpha
