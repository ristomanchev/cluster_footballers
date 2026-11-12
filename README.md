# Prospect Comps ⚽

A scouting app for young footballers in Europe's top five leagues **and their second tiers**:
it clusters players into playing-style archetypes and answers *"who does this prospect play like
in today's game?"*

| Tier 1 | Tier 2 |
|---|---|
| Premier League | Championship |
| LaLiga | LaLiga2 |
| Bundesliga | 2. Bundesliga |
| Serie A | Serie B |
| Ligue 1 | Ligue 2 |

## Quick start

```bash
pip install -r requirements.txt
python -m scout.build                      # fetch + build 2025/26 (first run ~40 min, then cached)
python -m scout.build --season 2026/2027   # optional: current season so far
streamlit run app.py
```

## What's in the app

- **Prospect board**: every U21 (adjustable) player with enough minutes, their archetype and their
  three closest comps among established players. Click a row for the full comparison.
- **Player comps**: pick anyone and get their 10 most similar players from the comparison pool
  (default: age 24+, top-five leagues), with a percentile radar and raw numbers side by side.
- **Archetype map**: a t-SNE map of every player's style profile, with archetypes highlighted and
  U21 prospects marked, plus a table of archetypes and their prototypical players.

## How it works

1. **Data** (`scout/fotmob.py`, `scout/build.py`): FotMob league tables give the roster, minutes
   and club for every league-season. Then, for each player with 300+ minutes, the builder fetches
   their profile (birth date, primary position, foot, height, market value, current club) and their
   full season stat sheet for that league. The league tables only list players who appeared in
   about half their team's matches; the per-player sheets cover everyone else (injury-hit regulars,
   January signings, late breakthroughs).
   Everything is cached under `data/raw/`; the tidy output is `data/processed/players_<season>.csv`.
2. **Style profile** (`scout/model.py`): 23 per-90 / rate metrics in six groups:
   - Shooting: non-penalty xG, shots, npxG per shot, touches in the opposition box
   - Creation: xA, chances created, big chances created, successful crosses
   - Carrying: successful dribbles, dribble success %
   - Passing: accurate passes, pass %, accurate long balls, long-ball %, line-breaking passes
   - Defending: tackles, interceptions, clearances, blocks, recoveries, possession won in the final third
   - Aerial: aerial duels won, aerial win %

   Success percentages (dribbles, long balls, aerials) are shrunk toward the position median for
   low-volume players.
3. **League adjustment**: attacking output is multiplied by a league-strength factor
   (`LEAGUE_STRENGTH` in `scout/config.py`, hand-set estimates, tune them).
4. **Similarity**: z-scores (fit on players with ≥ 900 min, clipped at ±3), then cosine
   similarity. Feature-group weights are adjustable in the sidebar.
5. **Archetypes**: K-means within each position group (CB, FB, DM/CM, AM/W, ST). Each cluster
   is named after the scouting role it fits best (e.g. "Ball-playing defender", "Target forward"),
   defined by signature stats in `ROLES` in `scout/config.py`.

## Project layout

```
app.py              Streamlit app
scout/config.py     leagues, stats, features, league factors, cluster counts
scout/fotmob.py     cached FotMob client
scout/build.py      dataset builder (CLI)
scout/model.py      features, archetypes, similarity, t-SNE map
data/processed/     built datasets (one CSV per season)
```

## Limitations and ideas

- Goalkeepers are not modelled yet (FotMob only publishes keeper stats for ~1,800+ minute keepers).
- Physical data (sprints, distance, top speed) is collected but only exists for the Premier League,
  so it's not part of the similarity profile.
- Event data misses off-ball work; team style shapes individual numbers. Treat comps as a prompt
  for video scouting.
- Ideas: multi-season profiles for prospects, possession-adjusted defensive stats, a
  "trajectory" view (similarity to what stars looked like at the same age), goalkeeper model.

Data comes from FotMob's public endpoints; it's for personal and educational use, so respect their
terms and keep the request rate low (the client waits 0.25 s between requests and caches everything).
