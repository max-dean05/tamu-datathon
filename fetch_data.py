"""
Pull college football games + betting totals from CollegeFootballData.com and
build one clean row per FBS-vs-FBS game.

Usage:
    export CFBD_API_KEY='your-key'          # from https://collegefootballdata.com/key
    python fetch_data.py --start 2014 --end 2025

    python fetch_data.py --synthetic         # fake data to test the pipeline before your key arrives

API budget: the free tier allows 1,000 calls/month. This script makes 4 calls per
season (games + lines, regular + postseason) and caches raw JSON in data/raw/, so
12 seasons costs ~48 calls once, and re-runs cost zero.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/cfb_totals.csv")

POWER_CONFS = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
# Programs that draw outsized recreational ("brand") money. Edit freely.
BRAND_TEAMS = {
    "Alabama", "Ohio State", "Texas", "Texas A&M", "Michigan", "Notre Dame",
    "Georgia", "USC", "LSU", "Oklahoma", "Florida", "Penn State", "Clemson",
    "Tennessee", "Oregon", "Auburn", "Florida State", "Miami",
}


def g(d, *keys, default=None):
    """Read a field that may be camelCase or snake_case depending on client version."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def fetch_season(api_games, api_lines, year, season_type):
    """Fetch (or load cached) games and lines for one season/season type."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, call in [
        ("games", lambda: api_games.get_games(year=year, season_type=season_type)),
        ("lines", lambda: api_lines.get_lines(year=year, season_type=season_type)),
    ]:
        path = RAW_DIR / f"{name}_{year}_{season_type}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
            continue
        records = [r.to_dict() for r in call()]
        path.write_text(json.dumps(records, default=str))
        print(f"  fetched {name} {year} {season_type}: {len(records)} rows")
        out[name] = records
    return out["games"], out["lines"]


def build_rows(games, lines):
    game_meta = {g(x, "id"): x for x in games}
    rows = []
    for gl in lines:
        gid = g(gl, "id")
        meta = game_meta.get(gid, {})
        totals = [g(l, "overUnder", "over_under") for l in g(gl, "lines", default=[])]
        opens = [g(l, "overUnderOpen", "over_under_open") for l in g(gl, "lines", default=[])]
        totals = [float(t) for t in totals if t is not None]
        opens = [float(t) for t in opens if t is not None]
        home_score = g(gl, "homeScore", "home_score")
        away_score = g(gl, "awayScore", "away_score")
        if not totals or home_score is None or away_score is None:
            continue
        rows.append({
            "game_id": gid,
            "season": g(gl, "season"),
            "week": g(gl, "week"),
            "season_type": str(g(gl, "seasonType", "season_type", default="")),
            "start_date": g(meta, "startDate", "start_date", default=g(gl, "startDate", "start_date")),
            "home_team": g(gl, "homeTeam", "home_team"),
            "away_team": g(gl, "awayTeam", "away_team"),
            "home_conf": g(gl, "homeConference", "home_conference"),
            "away_conf": g(gl, "awayConference", "away_conference"),
            "home_class": str(g(meta, "homeClassification", "home_classification",
                                default=g(gl, "homeClassification", "home_classification", default=""))).lower(),
            "away_class": str(g(meta, "awayClassification", "away_classification",
                                default=g(gl, "awayClassification", "away_classification", default=""))).lower(),
            "neutral_site": bool(g(meta, "neutralSite", "neutral_site", default=False)),
            "fair_total": float(np.median(totals)),     # consensus closing total across books
            "open_total": float(np.median(opens)) if opens else np.nan,
            "actual_total": int(home_score) + int(away_score),
        })
    return rows


def add_market_profile(df):
    """Features describing how much (and what kind of) money a game attracts."""
    df["power_home"] = df["home_conf"].isin(POWER_CONFS).astype(int)
    df["power_away"] = df["away_conf"].isin(POWER_CONFS).astype(int)
    df["brand"] = (df["home_team"].isin(BRAND_TEAMS) | df["away_team"].isin(BRAND_TEAMS)).astype(int)
    kickoff = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    try:
        kickoff = kickoff.dt.tz_convert("America/New_York")
    except Exception:                       # no tz database installed: approximate Eastern time
        kickoff = kickoff - pd.Timedelta(hours=5)
    df["weekday"] = kickoff.dt.dayofweek.isin([1, 2, 3]).astype(int)     # Tue/Wed/Thu ("MACtion")
    df["primetime"] = ((kickoff.dt.dayofweek == 5) & (kickoff.dt.hour >= 19)).astype(int)
    df["postseason"] = df["season_type"].str.contains("post", case=False).astype(int)
    score = (0.25 * df["power_home"] + 0.25 * df["power_away"] + 0.3 * df["brand"]
             + 0.2 * df["primetime"] + 0.2 * df["postseason"] - 0.2 * df["weekday"])
    df["profile"] = score.clip(0, 1).round(3)
    return df


def make_synthetic(n_seasons=8, games_per_season=700, seed=0):
    """Fake games with roughly realistic CFB totals, for testing before the key arrives."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = [f"Team{i}" for i in range(130)] + sorted(BRAND_TEAMS)
    confs = list(POWER_CONFS) + ["MAC", "Sun Belt", "Mountain West", "AAC", "C-USA"]
    for s in range(2015, 2015 + n_seasons):
        for i in range(games_per_season):
            fair = float(np.round(rng.normal(54, 8) * 2) / 2)
            rows.append({
                "game_id": s * 10000 + i, "season": s, "week": int(rng.integers(1, 15)),
                "season_type": "regular",
                "start_date": pd.Timestamp(f"{s}-09-01", tz="UTC")
                              + pd.Timedelta(days=int(rng.integers(0, 90)), hours=int(rng.integers(16, 28))),
                "home_team": rng.choice(teams), "away_team": rng.choice(teams),
                "home_conf": rng.choice(confs), "away_conf": rng.choice(confs),
                "home_class": "fbs", "away_class": "fbs", "neutral_site": False,
                "fair_total": fair,
                "open_total": float(np.round((fair + rng.normal(0, 2)) * 2) / 2),
                "actual_total": int(max(0, round(rng.normal(fair, 15)))),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2014)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.synthetic:
        df = make_synthetic()
    else:
        import cfbd
        key = os.environ.get("CFBD_API_KEY")
        if not key:
            raise SystemExit("Set CFBD_API_KEY first: export CFBD_API_KEY='your-key'")
        config = cfbd.Configuration(access_token=key)
        rows = []
        with cfbd.ApiClient(config) as client:
            api_games, api_lines = cfbd.GamesApi(client), cfbd.BettingApi(client)
            for year in range(args.start, args.end + 1):
                for season_type in ["regular", "postseason"]:
                    games, lines = fetch_season(api_games, api_lines, year, season_type)
                    rows += build_rows(games, lines)
        df = pd.DataFrame(rows)
        before = len(df)
        df = df[(df["home_class"] == "fbs") & (df["away_class"] == "fbs")]
        print(f"Kept {len(df)} FBS-vs-FBS games with totals (dropped {before - len(df)})")

    df = add_market_profile(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {len(df)} games, seasons {df.season.min()}-{df.season.max()}")
    print(f"Share with an opening line: {df.open_total.notna().mean():.0%}")
    print(f"SD of (actual - closing total): {(df.actual_total - df.fair_total).std():.1f} points")


if __name__ == "__main__":
    main()
