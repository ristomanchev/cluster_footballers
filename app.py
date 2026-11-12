"""Prospect Comps - a scouting app for young players in Europe's top two tiers.

    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scout.config import FEATURE_GROUPS, FEATURES, LEAGUES, OUTFIELD_GROUPS, SHORT_LABELS
from scout.model import (BASE_MIN_MINUTES, FEATURE_COLS, ScoutModel, available_seasons, build_model,
                         embedding_path, load_players)

# Reference palette (dataviz skill): categorical slots 1-2, muted context gray.
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
TOP_LEAGUES = [v[0] for v in LEAGUES.values() if v[2] == 1]
ALL_LEAGUES = [v[0] for v in LEAGUES.values()]

st.set_page_config(page_title="Prospect Comps", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------- data/model
# The model is built once per (season, league adjustment) on every player with 300+ minutes;
# the minutes slider only filters on top of it, so moving it costs no CPU.
@st.cache_resource(show_spinner="Building model ...")
def get_model(season: str, league_adjust: bool) -> ScoutModel:
    return build_model(load_players(season), min_minutes=BASE_MIN_MINUTES, league_adjust=league_adjust)


@st.cache_data(show_spinner="Loading player map ...")
def get_embedding(season: str, league_adjust: bool) -> np.ndarray:
    """Precomputed t-SNE coordinates aligned to the model's players (computed here only as a fallback)."""
    P = get_model(season, league_adjust).players
    path = embedding_path(season, league_adjust)
    if path.exists():
        xy = pd.read_csv(path).set_index("player_id").reindex(P["player_id"])
        if xy.notna().all().all():
            return xy[["x", "y"]].to_numpy()
    return get_model(season, league_adjust).compute_embedding()


@st.cache_data
def get_labels(season: str, league_adjust: bool) -> dict[int, str]:
    P = get_model(season, league_adjust).players
    age = P["age"].round(0).astype("Int64").astype(str).replace("<NA>", "?")
    return dict(zip(P["player_id"], P["name"] + " · " + P["team"] + " · " + age))


seasons = available_seasons()
if not seasons:
    st.error("No data yet. Run `python -m scout.build` first.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚽ Prospect Comps")
    season = st.selectbox("Season", seasons)
    min_minutes = st.slider("Minimum minutes played", 300, 2000, 600, step=50,
                            help="Players below this are excluded everywhere.")
    league_adjust = st.toggle("League-strength adjustment", value=True,
                              help="Scale attacking output (xG, xA, shots, chances, dribbles, "
                                   "line-breaking passes) by a league-quality factor, so a "
                                   "Championship winger isn't credited like a Premier League one.")
    st.subheader("Comparison pool")
    st.caption("Who prospects are compared against: established players in today's game.")
    pool_min_age = st.slider("Minimum age", 18, 32, 24)
    pool_leagues = st.multiselect("Leagues", ALL_LEAGUES, default=TOP_LEAGUES)
    with st.expander("Similarity weights"):
        st.caption("Emphasise parts of the game when matching players.")
        weights = {g: st.slider(g, 0.0, 3.0, 1.0, 0.25, key=f"w_{g}") for g in FEATURE_GROUPS}

model = get_model(season, league_adjust)
P = model.players
LABELS = get_labels(season, league_adjust)
active = (P["minutes"] >= min_minutes).to_numpy()
pool_mask = active & ((P["age"] >= pool_min_age) & P["league"].isin(pool_leagues)).to_numpy()


def fmt_value(v) -> str:
    if pd.isna(v):
        return "–"
    return f"€{v / 1e6:.1f}m" if v >= 1e6 else f"€{v / 1e3:.0f}k"


# ---------------------------------------------------------------- charts
def radar(pid_a: int, pid_b: int | None) -> go.Figure:
    labels = [SHORT_LABELS.get(c, FEATURES[c][0]) for c in FEATURE_COLS]
    full = [FEATURES[c][0] for c in FEATURE_COLS]
    fig = go.Figure()
    for pid, color in ((pid_a, BLUE), (pid_b, ORANGE)):
        if pid is None:
            continue
        i = P.index[P["player_id"] == pid][0]
        vals = model.percentiles.loc[i, FEATURE_COLS].astype(float).round(0).tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=labels + labels[:1], customdata=full + full[:1],
            name=P.at[i, "name"], line=dict(color=color, width=2), fill="toself", opacity=0.75,
            marker=dict(size=8),
            hovertemplate="%{customdata}: %{r:.0f}th percentile<extra>" + str(P.at[i, "name"]) + "</extra>"))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75], showline=False,
                                   tickfont=dict(size=9, color=MUTED),
                                   gridcolor="rgba(137,135,129,0.3)"),
                   angularaxis=dict(tickfont=dict(size=11), gridcolor="rgba(137,135,129,0.3)")),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0.5, xanchor="center"),
        margin=dict(l=90, r=90, t=60, b=40), height=560)
    return fig


def player_card(r):
    st.markdown(f"### {r['name']}")
    st.markdown(f"**{r.team}** · {r.league} · {r.get('primary_position_label') or r.pos_group} "
                f"· *{r.archetype}*")
    c = st.columns(4)
    c[0].metric("Age", f"{r.age:.0f}" if pd.notna(r.age) else "–")
    c[1].metric("Minutes", f"{r.minutes:,.0f}")
    c[2].metric("FotMob rating", f"{r.rating:.2f}" if pd.notna(r.rating) else "–")
    c[3].metric("Market value", fmt_value(r.market_value_eur))
    extra = []
    if pd.notna(r.get("current_team")) and r.current_team != r.team:
        extra.append(f"Now at **{r.current_team}**" + (" (on loan)" if r.get("on_loan") is True else ""))
    for key, lbl in (("nationality", ""), ("foot", "Foot: "), ("height_cm", "Height: ")):
        v = r.get(key)
        if pd.notna(v) and v != "":
            extra.append(f"{lbl}{v:.0f} cm" if key == "height_cm" else f"{lbl}{v}")
    if isinstance(r.get("leagues_played"), str) and " / " in r.leagues_played:
        extra.append(f"Played in: {r.leagues_played}")
    if extra:
        st.caption(" · ".join(extra))


def player_detail(pid: int, key: str):
    r = P[P["player_id"] == pid].iloc[0]
    player_card(r)

    sims = model.similar(pid, pool_mask=pool_mask, n=10, weights=weights)
    if sims.empty:
        st.info("The comparison pool is empty – loosen the pool filters in the sidebar.")
        return
    # Row 1: similar players, full width so no column gets squeezed.
    st.markdown("**Most similar players in the comparison pool**")
    table = sims[["similarity", "name", "age", "team", "league", "archetype", "minutes"]].rename(
        columns={"similarity": "Match %", "name": "Player", "age": "Age", "team": "Club",
                 "league": "League", "archetype": "Archetype", "minutes": "Mins"})
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Match %": st.column_config.ProgressColumn(
                     "Match %", min_value=0, max_value=100, format="%.0f"),
                     "Age": st.column_config.NumberColumn(format="%.0f")})

    # Row 2: pick a comp, then radar (wide) next to the raw numbers.
    pick, _ = st.columns([2, 3])
    opts = sims["player_id"].tolist()
    comp = pick.selectbox("Compare against", opts, key=f"comp_{key}", format_func=LABELS.get)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(radar(pid, comp), use_container_width=True, key=f"radar_{key}")
        st.caption("Percentiles vs. players in the same position group (≥ 900 min). "
                   "Hover a point for the full metric name. "
                   + ("Attacking output is league-adjusted." if league_adjust else ""))
    with right:
        a, b = r, P[P["player_id"] == comp].iloc[0]
        raw = pd.DataFrame({
            "Metric": [FEATURES[c][0] for c in FEATURE_COLS],
            a["name"]: [round(a[c], 2) for c in FEATURE_COLS],
            b["name"] + " ": [round(b[c], 2) for c in FEATURE_COLS],
        })
        st.markdown("**Raw numbers**")
        st.dataframe(raw, hide_index=True, use_container_width=True, height=560)


# ---------------------------------------------------------------- tabs
tab_board, tab_comps, tab_map, tab_about = st.tabs(
    ["Prospect board", "Player comps", "Archetype map", "How it works"])

with tab_board:
    f1, f2, f3, f4 = st.columns([1, 2, 2, 1])
    max_age = f1.slider("Max age", 16, 25, 21)
    leagues = f2.multiselect("Leagues", ALL_LEAGUES, default=ALL_LEAGUES, key="board_leagues")
    groups = f3.multiselect("Positions", OUTFIELD_GROUPS, default=OUTFIELD_GROUPS)
    min_mins_board = f4.number_input("Min minutes", min_minutes, 4000, max(min_minutes, 600), step=100)

    q_mask = ((P["age"] < max_age + 1) & P["league"].isin(leagues) & P["pos_group"].isin(groups)
              & (P["minutes"] >= min_mins_board)).to_numpy()
    comps = model.top_comps(q_mask, pool_mask, k=3, weights=weights)
    board = P[q_mask].join(comps).sort_values("rating", ascending=False)
    st.caption(f"{len(board)} prospects · compared against {pool_mask.sum()} established players "
               "· select a row to see the full comparison")
    view = pd.DataFrame({
        "Player": board["name"], "Age": board["age"],
        "Pos": board["primary_position_label"].fillna(board["pos_group"]),
        "Club": board["team"], "League": board["league"], "Mins": board["minutes"],
        "Rating": board["rating"], "Value": board["market_value_eur"].map(fmt_value),
        "Archetype": board["archetype"], "Plays like": board.get("comps"),
    })
    event = st.dataframe(
        view, hide_index=True, use_container_width=True, height=460,
        on_select="rerun", selection_mode="single-row",
        column_config={"Age": st.column_config.NumberColumn(format="%.1f"),
                       "Rating": st.column_config.NumberColumn(format="%.2f"),
                       "Plays like": st.column_config.TextColumn(width="large")})
    rows = event.selection.rows if event and event.selection else []
    if rows:
        st.divider()
        player_detail(int(board.iloc[rows[0]]["player_id"]), key="board")

with tab_comps:
    order = P[active].sort_values(["age", "minutes"], ascending=[True, False])
    options = order["player_id"].tolist()
    pid = st.selectbox("Player (any age, any of the 10 leagues)", options, format_func=LABELS.get,
                       index=None, placeholder="Type a name ...")
    if pid is not None:
        player_detail(pid, key="comps")
    else:
        st.info("Pick any player to find who they most resemble in the comparison pool.")

with tab_map:
    emb = get_embedding(season, league_adjust)
    arch = model.archetypes
    c1, c2 = st.columns([1, 2])
    group = c1.selectbox("Position group", OUTFIELD_GROUPS, index=3)
    names = arch.loc[arch["pos_group"] == group, "archetype"].tolist()
    focus = c2.selectbox("Highlight archetype", names)

    in_focus = (P["archetype"] == focus).to_numpy() & active
    young = (P["age"] < 22).to_numpy()
    hover = (P["name"] + "<br>" + P["team"] + " · " + P["league"] + "<br>Age "
             + P["age"].round(0).astype("Int64").astype(str) + " · " + P["archetype"]).to_numpy()
    fig = go.Figure()
    layers = [
        (active & ~in_focus, "Other players", MUTED, 5, 0.25),
        (in_focus & ~young, f"{focus} (22+)", BLUE, 8, 0.85),
        (in_focus & young, f"{focus} (U21 prospects)", ORANGE, 10, 0.95),
    ]
    for m, name, color, size, op in layers:
        fig.add_trace(go.Scattergl(
            x=emb[m, 0], y=emb[m, 1], mode="markers", name=name, text=hover[m],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(color=color, size=size, opacity=op, line=dict(width=0))))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("t-SNE projection of every player's style profile: nearby dots play alike. "
               "Distances between far-apart clusters are not meaningful.")

    st.markdown("**Archetypes**")
    st.dataframe(arch.rename(columns={"pos_group": "Group", "archetype": "Archetype",
                                      "players": "Players", "key_traits": "Stands out for",
                                      "prototypes": "Prototypical top-flight players"}),
                 hide_index=True, use_container_width=True)

with tab_about:
    st.markdown(f"""
**Data.** FotMob season stats for the Premier League, LaLiga, Bundesliga, Serie A, Ligue 1 and
their second tiers (Championship, LaLiga2, 2. Bundesliga, Serie B, Ligue 2), plus player profiles
(age, position, foot, height, market value). Season shown: **{season}**. Ages are as of today.

**Style profile.** Each outfield player becomes a vector of {len(FEATURE_COLS)} per-90 / rate
metrics across {len(FEATURE_GROUPS)} groups: {", ".join(FEATURE_GROUPS)}. Success percentages
(dribbles, long balls, aerials) are shrunk toward the position median for low-volume players so one lucky take-on ≠ 100%.

**League adjustment.** Attacking output is multiplied by a league-strength factor
(Premier League 1.00 → Ligue 2 0.70, editable in `scout/config.py`). Defensive volume is not
adjusted.

**Similarity.** Metrics are standardised (z-scores fitted on players with ≥ 900 min, clipped at ±3)
and compared with cosine similarity, which matches the *shape* of a profile, i.e. how a player
plays more than how much. The sidebar weights multiply each group's contribution.

**Archetypes.** K-means inside each position group (CB, FB, DM/CM, AM/W, ST) on the same
standardised profile. Names are generated from each cluster's two most distinctive traits.

**Caveats.** Event data can't see off-ball movement, scanning or pressing structure; team style
shapes individual numbers; young players' samples are small. Use comps as a starting point for
video scouting, not a verdict. Goalkeepers are not modelled yet.
""")
