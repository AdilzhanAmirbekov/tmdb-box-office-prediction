"""Pre-release feature engineering.

Two encodings of creative talent are available:
  * 'history' (A3, default): time-aware past box-office features of the director, the top-3
    billed cast and the production companies, computed only from films released at least
    HISTORY_GAP_DAYS before the target film;
  * 'old' (A2 plan): director and top-3 cast as categories, names with < 10 development
    films grouped into 'Other'.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import config as C

HIST_COLS = ["dir_prior_n", "dir_prior_logrev", "dir_hist_missing",
             "cast_prior_n_total", "cast_prior_logrev_mean", "cast_prior_logrev_max", "cast_hist_missing",
             "comp_prior_n_max", "comp_prior_logrev_max", "comp_hist_missing"]
TIMING_COLS = ["release_year", "release_month", "release_dow"]
BUDGET_COLS = ["log_budget", "budget_missing"]
LEAK_COLS = ["vote_count", "popularity", "vote_average"]


# --------------------------------------------------------------------------- history features
class _History:
    """For every entity: sorted release dates, cumulative film counts and cumulative sums of
    recorded log revenue, so that 'history before date t' is a binary search."""

    def __init__(self, dates, entity_lists, logrev):
        bucket = defaultdict(list)
        for d, ents, lr in zip(dates, entity_lists, logrev):
            for e in set(ents):
                if e == e:  # skip NaN
                    bucket[e].append((d, lr))
        self.tab = {}
        for e, rows in bucket.items():
            rows.sort(key=lambda r: r[0])
            ds = np.array([r[0] for r in rows], dtype="datetime64[ns]")
            lr = np.array([r[1] for r in rows], dtype=float)
            ok = ~np.isnan(lr)
            self.tab[e] = (ds, np.cumsum(np.ones(len(lr))), np.cumsum(np.where(ok, lr, 0.0)), np.cumsum(ok))

    def query(self, e, cutoff):
        """Number of earlier films and mean recorded log revenue for entity e before cutoff."""
        if e not in self.tab:
            return 0, np.nan
        ds, cn, cs, ck = self.tab[e]
        k = np.searchsorted(ds, cutoff, side="right")
        if k == 0:
            return 0, np.nan
        n_rec = ck[k - 1]
        return int(cn[k - 1]), (cs[k - 1] / n_rec if n_rec > 0 else np.nan)


def add_history_features(all_df):
    """all_df: every dated film (also films without recorded revenue, which add to counts only)."""
    dates = all_df["release_date"].to_numpy(dtype="datetime64[ns]")
    lr = all_df["log_revenue"].to_numpy(dtype=float)
    h_dir = _History(dates, [[d] for d in all_df["director_id"]], lr)
    h_cast = _History(dates, all_df["cast_ids"], lr)
    h_comp = _History(dates, all_df["companies_id"], lr)
    gap = np.timedelta64(C.HISTORY_GAP_DAYS, "D")
    out = {c: [] for c in HIST_COLS}
    for d, dir_id, top3, comps in zip(dates, all_df["director_id"], all_df["top3_ids"], all_df["companies_id"]):
        cut = d - gap
        n, m = h_dir.query(dir_id, cut) if dir_id == dir_id else (0, np.nan)
        out["dir_prior_n"].append(n)
        out["dir_prior_logrev"].append(m)
        out["dir_hist_missing"].append(int(np.isnan(m)))
        q = [h_cast.query(a, cut) for a in top3]
        means = [x[1] for x in q if not np.isnan(x[1])]
        out["cast_prior_n_total"].append(sum(x[0] for x in q))
        out["cast_prior_logrev_mean"].append(np.mean(means) if means else np.nan)
        out["cast_prior_logrev_max"].append(np.max(means) if means else np.nan)
        out["cast_hist_missing"].append(int(not means))
        q = [h_comp.query(c, cut) for c in comps]
        means = [x[1] for x in q if not np.isnan(x[1])]
        out["comp_prior_n_max"].append(max([x[0] for x in q], default=0))
        out["comp_prior_logrev_max"].append(np.max(means) if means else np.nan)
        out["comp_hist_missing"].append(int(not means))
    for c in HIST_COLS:
        all_df[c] = out[c]
    return all_df


# --------------------------------------------------------------------------- vocabulary (dev only)
def fit_vocab(dev):
    k = C.CAT_MIN_COUNT
    lang = Counter(dev["original_language"])
    comp = Counter(c for l in dev["companies_l"] for c in set(l))
    ctry = Counter(c for l in dev["countries_l"] for c in set(l))
    genres = sorted({g for l in dev["genres_l"] for g in l})
    dirs = Counter(dev["director_name"].dropna())
    acts = Counter(a for l in dev["top3_names"] for a in set(l))
    return {
        "languages": sorted(x for x, n in lang.items() if n >= k),
        "companies": sorted(x for x, n in comp.items() if n >= k),
        "countries": sorted(x for x, n in ctry.items() if n >= k),
        "genres": genres,
        "directors": sorted(x for x, n in dirs.items() if n >= k),
        "actors": sorted(x for x, n in acts.items() if n >= k),
    }


def old_encoding_other_shares(dev, vocab):
    """Share of development films whose director / k-th billed actor falls into 'Other'
    under the A2 rule (>= 10 development films)."""
    D, A = set(vocab["directors"]), set(vocab["actors"])
    res = {"dir_other_share": float(np.mean([d not in D for d in dev["director_name"]]))}
    for k in range(3):
        res[f"cast{k+1}_other_share"] = float(np.mean([(l[k] if len(l) > k else None) not in A
                                                       for l in dev["top3_names"]]))
    res["all_top3_other_share"] = float(np.mean([all(a not in A for a in l) for l in dev["top3_names"]]))
    res["dir_and_all_cast_other_share"] = float(np.mean(
        [(d not in D) and all(a not in A for a in l) for d, l in zip(dev["director_name"], dev["top3_names"])]))
    res["n_directors_kept"] = len(D)
    res["n_actors_kept"] = len(A)
    res["n_distinct_directors_dev"] = int(dev["director_name"].nunique())
    res["n_distinct_top3_actors_dev"] = len({a for l in dev["top3_names"] for a in l})
    return res


def _slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))[:40]


# --------------------------------------------------------------------------- design matrices
def transform(df, vocab, encoding="history", leak=False, drop=()):
    """Return (X_tree, X_cb, cat_cols).

    X_tree: all-numeric matrix (language one-hot), NaN kept for tree models.
    X_cb:   same numeric part with single-valued categoricals as strings (CatBoost native).
    """
    num = {}
    num["log_budget"] = np.log(df["budget_clean"])
    num["budget_missing"] = df["budget_clean"].isna().astype(int)
    num["runtime"] = df["runtime_clean"]
    num["release_year"] = df["release_date"].dt.year
    num["release_month"] = df["release_date"].dt.month
    num["release_dow"] = df["release_date"].dt.dayofweek
    num["n_companies"] = df["companies_l"].apply(len)
    num["n_countries"] = df["countries_l"].apply(len)
    for g in vocab["genres"]:
        num["genre_" + _slug(g)] = df["genres_l"].apply(lambda l, g=g: int(g in l))
    for c in vocab["companies"]:
        num["company_" + _slug(c)] = df["companies_l"].apply(lambda l, c=c: int(c in l))
    for c in vocab["countries"]:
        num["country_" + c] = df["countries_l"].apply(lambda l, c=c: int(c in l))
    if encoding == "history":
        for c in HIST_COLS:
            num[c] = df[c].astype(float)
    if leak:
        for c in LEAK_COLS:
            num[c] = df[c].astype(float)
    num = pd.DataFrame(num, index=df.index)

    lang = df["original_language"].where(df["original_language"].isin(vocab["languages"]), "Other")
    cats = pd.DataFrame({"language": lang}, index=df.index)
    if encoding == "old":
        D, A = set(vocab["directors"]), set(vocab["actors"])
        cats["director"] = df["director_name"].apply(lambda d: d if d in D else "Other")
        for k in range(3):
            cats[f"cast{k+1}"] = df["top3_names"].apply(
                lambda l, k=k: (l[k] if len(l) > k and l[k] in A else "Other"))

    dropcols = [c for c in num.columns if any(c == d or c.startswith(d) for d in drop)]
    num = num.drop(columns=dropcols)

    # one-hot version for LR / RF / XGB / LGBM
    oh = [pd.get_dummies(cats["language"], prefix="lang").reindex(
        columns=["lang_" + l for l in vocab["languages"] + ["Other"]], fill_value=0)]
    if encoding == "old":
        oh.append(pd.DataFrame({"director_" + _slug(d): (cats["director"] == d).astype(int)
                                for d in vocab["directors"]}, index=df.index))
        oh.append(pd.DataFrame({"actor_" + _slug(a): df["top3_names"].apply(lambda l, a=a: int(a in l))
                                for a in vocab["actors"]}, index=df.index))
    X_tree = pd.concat([num] + oh, axis=1).astype(float)
    X_cb = pd.concat([num.astype(float), cats], axis=1)
    return X_tree, X_cb, list(cats.columns)
