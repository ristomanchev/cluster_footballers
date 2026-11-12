"""Small FotMob client with on-disk caching and polite rate limiting."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import requests

from .config import RAW_DIR

BASE = "https://www.fotmob.com/api/data"
STATS_BASE = "https://data.fotmob.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/json",
}
MIN_INTERVAL = 0.25  # seconds between requests (shared across threads)

_session = requests.Session()
_session.headers.update(HEADERS)
_lock = threading.Lock()
_last = [0.0]


class NotFound(Exception):
    pass

def _get_json(url: str, retries: int = 3):
    for attempt in range(retries):
        with _lock:
            wait = MIN_INTERVAL - (time.monotonic() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
        try:
            r = _session.get(url, timeout=20)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (403, 404):
            raise NotFound(url)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if data is None:  # FotMob occasionally serves an empty body; retry
            time.sleep(2 * (attempt + 1))
            continue
        return data
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def _cached(path: Path, url: str, refresh: bool = False):
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = _get_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def season_stats_dir(league_id: int, season: str, refresh: bool = False) -> str:
    """Return the relative stats directory for a league season, e.g. 'stats/47/season/27110'."""
    data = _cached(RAW_DIR / "leagues" / f"{league_id}.json",
                   f"{BASE}/leagues?id={league_id}", refresh=refresh)
    for link in data["stats"]["seasonStatLinks"]:
        if link["Name"] == season:
            return link["RelativePath"].rsplit("/", 1)[0]
    available = [l["Name"] for l in data["stats"]["seasonStatLinks"]]
    raise ValueError(f"Season {season} not available for league {league_id}: {available}")


def stat_list(league_id: int, season: str, stat: str, refresh: bool = False) -> list[dict] | None:
    """All player rows for one stat in one league season (None if FotMob has no such file)."""
    rel = season_stats_dir(league_id, season)
    path = RAW_DIR / "stats" / season.replace("/", "-") / str(league_id) / f"{stat}.json"
    if path.exists() and not refresh:
        data = json.loads(path.read_text())
    else:
        try:
            data = _get_json(f"{STATS_BASE}/{rel}/{stat}.json")
        except NotFound:
            data = {"TopLists": []}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    if not data.get("TopLists"):
        return None
    return data["TopLists"][0]["StatList"]


def player_profile(player_id: int, refresh: bool = False) -> dict | None:
    """Compact player profile (birth date, position, height, foot, market value...)."""
    path = RAW_DIR / "profiles" / f"{player_id}.json"
    if path.exists() and not refresh:
        prof = json.loads(path.read_text())
        if "stat_entries" in prof:
            return prof
    try:
        d = _get_json(f"{BASE}/playerData?id={player_id}")
    except NotFound:
        return None
    info = {x.get("title"): x.get("value", {}) for x in d.get("playerInformation", [])}
    pos = d.get("positionDescription") or {}
    team = d.get("primaryTeam") or {}
    prof = {
        "id": d.get("id"),
        "name": d.get("name"),
        "birth_date": ((d.get("birthDate") or {}).get("utcTime") or "")[:10] or None,
        "primary_position": (pos.get("primaryPosition") or {}).get("key"),
        "primary_position_label": (pos.get("primaryPosition") or {}).get("label"),
        "other_positions": [p.get("label") for p in pos.get("nonPrimaryPositions") or []],
        "height_cm": (info.get("Height") or {}).get("numberValue"),
        "foot": (info.get("Preferred foot") or {}).get("fallback"),
        "nationality": (info.get("Country") or {}).get("fallback"),
        "market_value_eur": (info.get("Market value") or {}).get("numberValue"),
        "contract_end": (info.get("Contract end") or {}).get("dateValue"),
        "current_team": team.get("teamName"),
        "on_loan": team.get("onLoan"),
        # (season, tournament) -> entry id used by the per-season stats endpoint
        "stat_entries": [
            {"season": s.get("seasonName"), "tournament_id": t.get("tournamentId"),
             "entry_id": t.get("entryId"), "deep": t.get("hasDeepStats")}
            for s in d.get("statSeasons") or [] for t in s.get("tournaments") or []
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prof))
    return prof


def player_season_stats(player_id: int, entry_id: str, season: str,
                        league_id: int, refresh: bool = False) -> dict | None:
    """Full stat sheet for one player in one competition-season: {stat_key: [total, per90]}."""
    path = RAW_DIR / "player_stats" / season.replace("/", "-") / str(league_id) / f"{player_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    try:
        d = _get_json(f"{BASE}/playerStats?playerId={player_id}&seasonId={entry_id}")
    except NotFound:
        return None

    def num(x):
        try:
            return float(str(x).replace(",", ""))
        except (TypeError, ValueError):
            return None

    out = {}
    for item in (d.get("topStatCard") or {}).get("items") or []:
        out["top:" + item.get("localizedTitleId", "")] = [num(item.get("statValue")), item.get("per90")]
    for sec in (d.get("statsSection") or {}).get("items") or []:
        for item in sec.get("items") or []:
            out[item.get("localizedTitleId", "")] = [num(item.get("statValue")), item.get("per90")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out
