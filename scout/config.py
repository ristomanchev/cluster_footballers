"""Central configuration: leagues, stats to pull, model features."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DEFAULT_SEASON = "2025/2026"

# FotMob league id -> (name, country, tier)
LEAGUES = {
    47: ("Premier League", "ENG", 1),
    87: ("LaLiga", "ESP", 1),
    54: ("Bundesliga", "GER", 1),
    55: ("Serie A", "ITA", 1),
    53: ("Ligue 1", "FRA", 1),
    48: ("Championship", "ENG", 2),
    140: ("LaLiga2", "ESP", 2),
    146: ("2. Bundesliga", "GER", 2),
    86: ("Serie B", "ITA", 2),
    110: ("Ligue 2", "FRA", 2),
}

# Rough league-strength multipliers (Premier League = 1.0) applied to attacking
# output when "league adjustment" is on. Hand-set estimates - tune to taste.
LEAGUE_STRENGTH = {
    "Premier League": 1.00,
    "LaLiga": 0.95,
    "Bundesliga": 0.93,
    "Serie A": 0.93,
    "Ligue 1": 0.88,
    "Championship": 0.78,
    "2. Bundesliga": 0.74,
    "LaLiga2": 0.72,
    "Serie B": 0.71,
    "Ligue 2": 0.70,
}

# --- FotMob stat files -------------------------------------------------------
# Season totals: value is a raw count for the season.
TOTAL_STATS = [
    "mins_played",
    "goals",
    "goal_assist",
    "expected_goals",
    "expected_assists",
    "total_att_assist",       # chances created
    "big_chance_created",
    "penalty_won",
    "yellow_card",
]
# Per-90 stats (value already per 90). Sub-value meaning differs per stat.
PER90_STATS = [
    "total_scoring_att",       # shots
    "ontarget_scoring_att",    # shots on target (sub = accuracy %)
    "accurate_pass",           # sub = pass completion %
    "accurate_long_balls",     # sub = long-ball accuracy %
    "lbp_total",               # line-breaking passes
    "won_contest",             # successful dribbles (sub = success %)
    "total_tackle",
    "interception",
    "effective_clearance",
    "outfielder_block",
    "ball_recovery",
    "poss_won_att_3rd",
    "defensive_contributions",
    "fouls",
    "rating",                  # average FotMob match rating
    "phys_sprints_per_90",     # physical data: Premier League only
    "phys_tdc_per_90",
    "phys_ts",
]
# Per-90 stats whose sub-value is the season total -> recompute precise per-90.
SUB_IS_TOTAL = {
    "total_tackle", "interception", "effective_clearance", "outfielder_block",
    "ball_recovery", "lbp_total", "defensive_contributions",
}
# Per-90 stats whose sub-value is a success percentage we want to keep.
SUB_PCT = {
    "accurate_pass": "pass_pct",
    "accurate_long_balls": "long_ball_pct",
    "won_contest": "dribble_pct",
    "ontarget_scoring_att": "shot_accuracy_pct",
}

# --- Model features ----------------------------------------------------------
# column -> (label, group)
FEATURES = {
    "npxg_p90": ("Non-penalty xG /90", "Shooting"),
    "shots_p90": ("Shots /90", "Shooting"),
    "xg_per_shot": ("npxG per shot", "Shooting"),
    "box_touches_p90": ("Touches in opp. box /90", "Shooting"),
    "xa_p90": ("xA /90", "Creation"),
    "chances_created_p90": ("Chances created /90", "Creation"),
    "big_chances_created_p90": ("Big chances created /90", "Creation"),
    "crosses_p90": ("Successful crosses /90", "Creation"),
    "dribbles_p90": ("Successful dribbles /90", "Carrying"),
    "dribble_pct": ("Dribble success %", "Carrying"),
    "passes_p90": ("Accurate passes /90", "Passing"),
    "pass_pct": ("Pass completion %", "Passing"),
    "long_balls_p90": ("Accurate long balls /90", "Passing"),
    "long_ball_pct": ("Long-ball accuracy %", "Passing"),
    "lbp_p90": ("Line-breaking passes /90", "Passing"),
    "tackles_p90": ("Tackles /90", "Defending"),
    "interceptions_p90": ("Interceptions /90", "Defending"),
    "clearances_p90": ("Clearances /90", "Defending"),
    "blocks_p90": ("Blocks /90", "Defending"),
    "recoveries_p90": ("Recoveries /90", "Defending"),
    "poss_won_att3rd_p90": ("Possession won final third /90", "Defending"),
    "aerials_p90": ("Aerial duels won /90", "Aerial"),
    "aerial_pct": ("Aerial duel win %", "Aerial"),
}
FEATURE_GROUPS = ["Shooting", "Creation", "Carrying", "Passing", "Defending", "Aerial"]

# Compact labels for crowded charts (radar spokes); full labels stay in FEATURES.
SHORT_LABELS = {
    "npxg_p90": "npxG", "shots_p90": "Shots", "xg_per_shot": "npxG/shot",
    "box_touches_p90": "Box touches", "xa_p90": "xA", "chances_created_p90": "Chances",
    "big_chances_created_p90": "Big chances", "crosses_p90": "Crosses",
    "dribbles_p90": "Dribbles", "dribble_pct": "Dribble %", "passes_p90": "Passes",
    "pass_pct": "Pass %", "long_balls_p90": "Long balls", "long_ball_pct": "Long ball %",
    "lbp_p90": "Line-breaking", "tackles_p90": "Tackles", "interceptions_p90": "Interceptions",
    "clearances_p90": "Clearances", "blocks_p90": "Blocks", "recoveries_p90": "Recoveries",
    "poss_won_att3rd_p90": "High regains", "aerials_p90": "Aerials won", "aerial_pct": "Aerial %",
}

# Volume of attacking output scales with league quality; these get multiplied
# by LEAGUE_STRENGTH when league adjustment is on.
LEAGUE_ADJUSTED = [
    "npxg_p90", "shots_p90", "box_touches_p90", "xa_p90", "chances_created_p90",
    "big_chances_created_p90", "crosses_p90", "dribbles_p90", "lbp_p90",
]

# Short trait names used to auto-label archetype clusters.
TRAIT_NAMES = {
    "npxg_p90": "Goal threat",
    "shots_p90": "Shot-happy",
    "xg_per_shot": "Shot selector",
    "box_touches_p90": "Box presence",
    "xa_p90": "Chance creator",
    "chances_created_p90": "Playmaker",
    "big_chances_created_p90": "Final-ball specialist",
    "crosses_p90": "Crosser",
    "aerials_p90": "Aerial presence",
    "aerial_pct": "Aerial dominance",
    "dribbles_p90": "Dribbler",
    "dribble_pct": "Press-resistant",
    "passes_p90": "Ball circulator",
    "pass_pct": "Secure passer",
    "long_balls_p90": "Long-range distributor",
    "long_ball_pct": "Accurate switcher",
    "lbp_p90": "Line-breaker",
    "tackles_p90": "Tackler",
    "interceptions_p90": "Interceptor",
    "clearances_p90": "Box defender",
    "blocks_p90": "Blocker",
    "recoveries_p90": "Ball winner",
    "poss_won_att3rd_p90": "High presser",
}

# FotMob primary-position key (e.g. "leftwinger", "centerdefensivemidfielder") -> group.
# Checked in order; first substring match wins.
_POSITION_RULES = [
    ("keeper", "GK"),
    ("centerback", "CB"), ("centreback", "CB"),
    ("wingback", "FB"), ("rightback", "FB"), ("leftback", "FB"),
    ("defensivemidfielder", "DM/CM"), ("centralmidfielder", "DM/CM"), ("centermidfielder", "DM/CM"),
    ("attackingmidfielder", "AM/W"), ("winger", "AM/W"),
    ("rightmidfielder", "AM/W"), ("leftmidfielder", "AM/W"),
    ("striker", "ST"), ("forward", "ST"),
]


def position_group(key) -> str | None:
    if not isinstance(key, str):
        return None
    k = key.lower().replace("_", "")  # keys come as "leftwingback" or "left_wing_back"
    return next((g for pat, g in _POSITION_RULES if pat in k), None)
OUTFIELD_GROUPS = ["CB", "FB", "DM/CM", "AM/W", "ST"]

# Scouting roles per position group, defined by their signature features. Each
# K-means cluster is named after the role whose signature its centroid fits best.
ROLES = {
    "CB": {
        "Ball-playing defender": ["passes_p90", "lbp_p90", "long_balls_p90", "pass_pct"],
        "Aerial stopper": ["aerials_p90", "aerial_pct", "clearances_p90", "blocks_p90"],
        "Front-foot defender": ["tackles_p90", "interceptions_p90", "poss_won_att3rd_p90", "dribbles_p90"],
    },
    "FB": {
        "Attacking crosser": ["crosses_p90", "xa_p90", "chances_created_p90", "big_chances_created_p90"],
        "Defensive full-back": ["tackles_p90", "clearances_p90", "blocks_p90", "interceptions_p90"],
        "Build-up full-back": ["passes_p90", "lbp_p90", "pass_pct", "long_balls_p90"],
        "Dribbling wing-back": ["dribbles_p90", "box_touches_p90", "npxg_p90"],
    },
    "DM/CM": {
        "Deep-lying playmaker": ["passes_p90", "long_balls_p90", "lbp_p90", "long_ball_pct"],
        "Box-to-box runner": ["box_touches_p90", "shots_p90", "npxg_p90", "dribbles_p90"],
        "Ball-winning anchor": ["interceptions_p90", "tackles_p90", "aerials_p90", "blocks_p90", "recoveries_p90"],
        "Creative eight": ["crosses_p90", "chances_created_p90", "xa_p90", "big_chances_created_p90"],
    },
    "AM/W": {
        "Wide creator": ["crosses_p90", "chances_created_p90", "xa_p90", "big_chances_created_p90"],
        "Dribbling winger": ["dribbles_p90", "dribble_pct", "box_touches_p90"],
        "Inside forward": ["npxg_p90", "shots_p90", "box_touches_p90", "xg_per_shot"],
        "Two-way wide player": ["tackles_p90", "interceptions_p90", "recoveries_p90", "aerial_pct"],
        "Advanced playmaker": ["passes_p90", "lbp_p90", "chances_created_p90", "pass_pct"],
    },
    "ST": {
        "Target forward": ["aerials_p90", "aerial_pct", "clearances_p90"],
        "Poacher": ["npxg_p90", "xg_per_shot", "box_touches_p90"],
        "Link-up forward": ["xa_p90", "chances_created_p90", "passes_p90", "dribbles_p90"],
        "Pressing forward": ["poss_won_att3rd_p90", "tackles_p90", "recoveries_p90"],
    },
}
ARCHETYPES_PER_GROUP = {"CB": 3, "FB": 3, "DM/CM": 4, "AM/W": 4, "ST": 3}

MIN_PROFILE_MINUTES = 300   # only fetch profiles for players with this many minutes
RELIABLE_MINUTES = 900      # players used to fit clusters / scalers
