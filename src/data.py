"""Loading, cleaning and chronological splitting of the TMDB 5000 data."""
import json

import numpy as np
import pandas as pd

from . import config as C


def _names(s, key="name"):
    try:
        return [d[key] for d in json.loads(s)]
    except (TypeError, ValueError):
        return []


def load_raw():
    movies = pd.read_csv(C.DATA_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(C.DATA_DIR / "tmdb_5000_credits.csv")
    return movies, credits


def load_joined():
    """Join movies and credits, parse JSON fields. Returns (df, log) where df holds ALL films
    with a valid release date (needed for history features and selection analysis)."""
    movies, credits = load_raw()
    log = {"n_movies_raw": int(len(movies)), "n_credits_raw": int(len(credits))}
    credits = credits.rename(columns={"movie_id": "id", "title": "title_credits"})
    df = movies.merge(credits, on="id", how="inner")
    log["n_joined"] = int(len(df))
    log["n_title_mismatch_after_join"] = int((df["title"] != df["title_credits"]).sum())
    n0 = len(df)
    df = df.drop_duplicates(subset="id")
    log["n_duplicate_ids_removed"] = int(n0 - len(df))
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    no_date = df[df["release_date"].isna()]
    log["n_missing_release_date"] = int(len(no_date))
    excluded = [no_date.assign(reason="missing release date")]
    df = df[df["release_date"].notna()].copy()

    df["genres_l"] = df["genres"].apply(_names)
    df["companies_l"] = df["production_companies"].apply(_names)
    df["companies_id"] = df["production_companies"].apply(lambda s: _names(s, "id"))
    df["countries_l"] = df["production_countries"].apply(lambda s: _names(s, "iso_3166_1"))

    def cast_ids(s):
        cast = sorted(json.loads(s), key=lambda d: d["order"])
        return [d["id"] for d in cast]

    def cast_names(s):
        cast = sorted(json.loads(s), key=lambda d: d["order"])
        return [d["name"] for d in cast[:3]]

    def director(s):
        dirs = [d for d in json.loads(s) if d.get("job") == "Director"]
        return (dirs[0]["id"], dirs[0]["name"]) if dirs else (np.nan, None)

    df["cast_ids"] = df["cast"].apply(cast_ids)
    df["top3_ids"] = df["cast_ids"].apply(lambda l: l[:3])
    df["top3_names"] = df["cast"].apply(cast_names)
    d = df["crew"].apply(director)
    df["director_id"] = d.apply(lambda t: t[0])
    df["director_name"] = d.apply(lambda t: t[1])

    # Monetary fields: 0 = not recorded; values below MIN_VALID_USD are entry errors.
    df["revenue_ok"] = df["revenue"] >= C.MIN_VALID_USD
    df["budget_clean"] = df["budget"].where(df["budget"] >= C.MIN_VALID_USD, np.nan).astype(float)
    df["runtime_clean"] = df["runtime"].where(df["runtime"] > 0, np.nan).astype(float)
    df["log_revenue"] = np.where(df["revenue_ok"], np.log1p(df["revenue"].clip(lower=0)), np.nan)
    df = df.sort_values(["release_date", "id"]).reset_index(drop=True)
    return df, log, excluded


def modelling_population(df, log, excluded):
    """Apply the target rules and return the modelling sample with an exclusion log."""
    zero = df[df["revenue"] == 0]
    tiny = df[(df["revenue"] > 0) & (df["revenue"] < C.MIN_VALID_USD)]
    excluded = excluded + [zero.assign(reason="revenue = 0 (not recorded)"),
                           tiny.assign(reason=f"0 < revenue < {C.MIN_VALID_USD} USD (entry error)")]
    log["n_revenue_zero"] = int(len(zero))
    log["n_revenue_tiny"] = int(len(tiny))
    log["n_budget_zero_in_all_dated"] = int((df["budget"] == 0).sum())
    pop = df[df["revenue_ok"]].copy().reset_index(drop=True)
    log["n_model_population"] = int(len(pop))
    log["n_budget_tiny_set_missing_in_pop"] = int(((pop["budget"] > 0) & (pop["budget"] < C.MIN_VALID_USD)).sum())
    log["n_budget_zero_in_pop"] = int((pop["budget"] == 0).sum())
    log["n_budget_missing_in_pop"] = int(pop["budget_clean"].isna().sum())
    log["n_runtime_missing_in_pop"] = int(pop["runtime_clean"].isna().sum())
    ex = pd.concat(excluded)[["id", "title", "release_date", "revenue", "budget", "status", "reason"]]
    return pop, log, ex


def chronological_split(pop):
    """Rows of pop are already sorted by release date (ties by id)."""
    return split_at_fraction(pop, 1 - C.TEST_FRACTION)


def split_at_fraction(pop, dev_fraction):
    """Earliest dev_fraction of films = development. Films sharing the boundary release date are
    kept together in development so that every test film is released strictly later."""
    n = len(pop)
    n_dev = n - int(round((1 - dev_fraction) * n))
    last_dev_date = pop["release_date"].iloc[n_dev - 1]
    is_dev = (pop["release_date"] <= last_dev_date).to_numpy()
    return is_dev
