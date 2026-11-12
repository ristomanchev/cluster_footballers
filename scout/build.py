"""Build the player dataset for one season.

    python -m scout.build                     # 2025/2026, all 10 leagues
    python -m scout.build --season 2026/2027  # current season so far
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd

from . import fotmob
from .config import (DEFAULT_SEASON, LEAGUES, MIN_PROFILE_MINUTES, PER90_STATS,
                     position_group, PROCESSED_DIR, SUB_IS_TOTAL, SUB_PCT, TOTAL_STATS)

NAN_WHEN_MISSING = {"rating", "phys_sprints_per_90", "phys_tdc_per_90", "phys_ts"}
QUALIFIER_STATS = ["accurate_pass", "ball_recovery", "defensive_contributions"]

# Per-player stat sheet key -> (our column, which number to use)
PLAYER_STAT_MAP = {
    "non_penalty_xg": ("npxg_p90", "p90"),
    "expected_goals": ("xg_p90", "p90"),
    "shots": ("shots_p90", "p90"),
    "touches_opp_box": ("box_touches_p90", "p90"),
    "expected_assists": ("xa_p90", "p90"),
    "chances_created": ("chances_created_p90", "p90"),
    "big_chance_created_team_title": ("big_chances_created_p90", "p90"),
    "crosses_succeeeded": ("crosses_p90", "p90"),
    "dribbles_succeeded": ("dribbles_p90", "p90"),
    "won_contest_subtitle": ("dribble_pct", "pct"),
    "successful_passes": ("passes_p90", "p90"),
    "successful_passes_accuracy": ("pass_pct", "pct"),
    "long_balls_accurate": ("long_balls_p90", "p90"),
    "long_ball_succeeeded_accuracy": ("long_ball_pct", "pct"),
    "line_breaking_passes": ("lbp_p90", "p90"),
    "matchstats.headers.tackles": ("tackles_p90", "p90"),
    "interceptions": ("interceptions_p90", "p90"),
    "clearances": ("clearances_p90", "p90"),
    "blocked_shots": ("blocks_p90", "p90"),
    "recoveries": ("recoveries_p90", "p90"),
    "poss_won_att_3rd_team_title": ("poss_won_att3rd_p90", "p90"),
    "aerials_won": ("aerials_p90", "p90"),
    "aerials_won_percent": ("aerial_pct", "pct"),
    "duel_won_percent": ("duel_pct", "pct"),
    "touches": ("touches_p90", "p90"),
    "defensive_actions": ("def_actions_p90", "p90"),
    "fouls": ("fouls_p90", "p90"),
    "goals": ("goals_p90", "p90"),
    "assists": ("assists_p90", "p90"),
}


def league_frame(league_id: int, season: str) -> pd.DataFrame:
    name, country, tier = LEAGUES[league_id]
    base = fotmob.stat_list(league_id, season, "mins_played")
    if not base:
        raise RuntimeError(f"No minutes data for {name} {season}")
    df = pd.DataFrame({
        "player_id": [r["ParticiantId"] for r in base],
        "name": [r["ParticipantName"] for r in base],
        "team": [r["TeamName"] for r in base],
        "minutes": [r["MinutesPlayed"] for r in base],
        "matches": [r["MatchesPlayed"] for r in base],
    }).set_index("player_id")
    df["league"], df["league_country"], df["tier"] = name, country, tier
    df["league_id"] = league_id

    for stat in TOTAL_STATS:
        if stat == "mins_played":
            continue
        rows = fotmob.stat_list(league_id, season, stat) or []
        vals = {r["ParticiantId"]: r["StatValue"] for r in rows}
        df[stat] = df.index.map(vals).astype(float)
        df[stat] = df[stat].fillna(0.0)

    # FotMob only lists "qualified" players in per-90 tables (based on share of team
    # matches, so long-term injured players / January movers can be missing). Every
    # qualified outfielder makes passes and recoveries, so those lists define the set.
    qualified = set()
    for stat in QUALIFIER_STATS:
        qualified |= {r["ParticiantId"] for r in fotmob.stat_list(league_id, season, stat) or []}
    df["per90_qualified"] = df.index.isin(qualified)

    for stat in PER90_STATS:
        rows = fotmob.stat_list(league_id, season, stat)
        if rows is None:
            df[stat] = np.nan
            continue
        vals, subs = {}, {}
        for r in rows:
            v, sub, mins = r["StatValue"], r.get("SubStatValue"), r["MinutesPlayed"]
            if stat in SUB_IS_TOTAL and sub is not None and mins:
                precise = sub / mins * 90
                if abs(precise - v) <= 0.051:  # sub really is the season total
                    v = precise
            vals[r["ParticiantId"]] = v
            subs[r["ParticiantId"]] = sub
        col = df.index.map(vals).astype(float)
        if stat not in NAN_WHEN_MISSING:
            # Qualified but absent from a per-90 list => zero of that action.
            col = np.where(np.isnan(col) & df["per90_qualified"], 0.0, col)
        df[stat] = col
        if stat in SUB_PCT:
            df[SUB_PCT[stat]] = df.index.map(subs).astype(float)
    return df


def fetch_profiles(ids: list[int], workers: int = 4) -> pd.DataFrame:
    out, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fotmob.player_profile, pid): pid for pid in ids}
        for fut in as_completed(futs):
            done += 1
            if done % 250 == 0 or done == len(ids):
                print(f"  profiles {done}/{len(ids)}", flush=True)
            try:
                prof = fut.result()
            except Exception as e:  # keep going on individual failures
                print(f"  ! profile {futs[fut]} failed: {e}")
                continue
            if prof:
                out.append(prof)
    return pd.DataFrame(out).rename(columns={"id": "player_id", "name": "profile_name"})


def _player_row(pid: int, entries, season: str, league_id: int) -> dict | None:
    entries = entries if isinstance(entries, list) else []
    entry = next((e["entry_id"] for e in entries
                  if isinstance(e, dict) and e["season"] == season and e["tournament_id"] == league_id and e.get("deep")), None)
    if entry is None:
        return None
    stats = fotmob.player_season_stats(pid, entry, season, league_id)
    # A sheet with only the headline card (no detailed sections) has no style data.
    if not stats or not any(not k.startswith("top:") for k in stats):
        return None
    row = {"player_id": pid}
    for key, (col, kind) in PLAYER_STAT_MAP.items():
        total, p90 = stats.get(key, [None, None])
        if kind == "pct":
            row[col] = total  # absent => no attempts => NaN
        else:
            row[col] = p90 if p90 is not None else 0.0
    rating = stats.get("top:rating", [None, None])[1]
    row["rating"] = rating if rating else np.nan
    npxg, shots = stats.get("non_penalty_xg", [0, 0])[0] or 0, stats.get("shots", [0, 0])[0] or 0
    row["xg_per_shot"] = npxg / shots if shots >= 3 else np.nan
    return row


def fetch_player_stats(df: pd.DataFrame, season: str, workers: int = 4) -> pd.DataFrame:
    out, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_player_row, r.player_id, r.stat_entries, season, r.league_id): r.player_id
                for r in df.itertuples()}
        for fut in as_completed(futs):
            done += 1
            if done % 250 == 0 or done == len(futs):
                print(f"  player stats {done}/{len(futs)}", flush=True)
            try:
                row = fut.result()
            except Exception as e:
                print(f"  ! stats {futs[fut]} failed: {e}")
                continue
            if row:
                out.append(row)
    return pd.DataFrame(out).set_index("player_id") if out else pd.DataFrame()


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Table-based features (fallback when a player's own stat sheet is unavailable)."""
    m90 = df["minutes"].clip(lower=1) / 90
    df["goals_p90"] = df["goals"] / m90
    df["assists_p90"] = df["goal_assist"] / m90
    df["xg_p90"] = df["expected_goals"] / m90
    df["xa_p90"] = df["expected_assists"] / m90
    df["chances_created_p90"] = df["total_att_assist"] / m90
    df["big_chances_created_p90"] = df["big_chance_created"] / m90
    df["shots_p90"] = df["total_scoring_att"]
    shots_total = df["shots_p90"] * m90
    df["xg_per_shot"] = np.where(shots_total >= 3, df["expected_goals"] / shots_total.clip(lower=1), np.nan)
    df["passes_p90"] = df["accurate_pass"]
    df["long_balls_p90"] = df["accurate_long_balls"]
    df["lbp_p90"] = df["lbp_total"]
    df["dribbles_p90"] = df["won_contest"]
    df["tackles_p90"] = df["total_tackle"]
    df["interceptions_p90"] = df["interception"]
    df["clearances_p90"] = df["effective_clearance"]
    df["blocks_p90"] = df["outfielder_block"]
    df["recoveries_p90"] = df["ball_recovery"]
    df["poss_won_att3rd_p90"] = df["poss_won_att_3rd"]
    df["def_actions_p90"] = df["defensive_contributions"]
    df["fouls_p90"] = df["fouls"]
    df["sprints_p90"] = df["phys_sprints_per_90"]
    df["distance_km_p90"] = df["phys_tdc_per_90"] / 1000
    df["top_speed_kmh"] = df["phys_ts"]
    return df


def build(season: str, workers: int = 4) -> pd.DataFrame:
    frames = []
    for lid, (name, _, _) in LEAGUES.items():
        print(f"[{name}] fetching stats ...", flush=True)
        frames.append(league_frame(lid, season))
    df = pd.concat(frames).reset_index()

    # Players who switched league mid-season: keep their main league row.
    df["leagues_played"] = df.groupby("player_id")["league"].transform(lambda s: " / ".join(s))
    df = df.sort_values("minutes", ascending=False).drop_duplicates("player_id")
    df = df[df["minutes"] >= MIN_PROFILE_MINUTES].copy()

    print(f"Fetching {len(df)} player profiles (cached after first run) ...", flush=True)
    prof = fetch_profiles(df["player_id"].tolist(), workers=workers)
    df = df.merge(prof, on="player_id", how="left")

    today = pd.Timestamp(date.today())
    bd = pd.to_datetime(df["birth_date"], errors="coerce")
    df["age"] = ((today - bd).dt.days / 365.25).round(1)
    df["pos_group"] = df["primary_position"].map(position_group)
    df["season"] = season
    df = derive_features(df)
    df["npxg_p90"] = np.nan  # only available from player stat sheets
    for c in ("box_touches_p90", "crosses_p90", "aerials_p90", "aerial_pct", "duel_pct", "touches_p90"):
        df[c] = np.nan

    print(f"Fetching per-player season stat sheets ...", flush=True)
    ps = fetch_player_stats(df, season, workers=workers)
    df = df.set_index("player_id")
    df["has_player_stats"] = df.index.isin(ps.index)
    if len(ps):
        df.loc[ps.index, ps.columns] = ps  # overwrite table values with the richer sheet
    df = df.reset_index().drop(columns=["stat_entries"])
    print(f"  stat sheets: {df['has_player_stats'].sum()}/{len(df)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"players_{season.replace('/', '-')}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} players -> {out}")
    print(df.groupby("league")["player_id"].count().to_string())
    print("Position groups:\n" + df["pos_group"].value_counts(dropna=False).to_string())

    from .model import save_embeddings  # local import: model depends on the saved CSV
    print("Precomputing archetype map (t-SNE) ...", flush=True)
    save_embeddings(season)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    build(args.season, workers=args.workers)


if __name__ == "__main__":
    main()
