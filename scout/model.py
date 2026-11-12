"""Feature preparation, archetype clustering and similarity search."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import (ARCHETYPES_PER_GROUP, FEATURES, LEAGUE_ADJUSTED, LEAGUE_STRENGTH,
                     OUTFIELD_GROUPS, PROCESSED_DIR, RELIABLE_MINUTES, ROLES, TRAIT_NAMES)

FEATURE_COLS = list(FEATURES)
Z_CLIP = 3.0
SHRINK_ATTEMPTS = 10  # prior strength (attempts) for success-% shrinkage


def available_seasons() -> list[str]:
    files = sorted(PROCESSED_DIR.glob("players_*.csv"), reverse=True)
    return [f.stem.replace("players_", "").replace("-", "/") for f in files]


def load_players(season: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"players_{season.replace('/', '-')}.csv")


def _shrink_pct(df: pd.DataFrame, pct_col: str, p90_col: str) -> pd.Series:
    """Shrink success % toward the position-group median for low-volume players."""
    m90 = df["minutes"] / 90
    succ = df[p90_col] * m90
    pct = df[pct_col] / 100
    att = np.where(pct > 0, succ / pct.where(pct > 0), np.nan)
    prior = df.groupby("pos_group")[pct_col].transform("median") / 100
    shrunk = (succ + SHRINK_ATTEMPTS * prior) / (att + SHRINK_ATTEMPTS)
    return (shrunk * 100).where(~np.isnan(att), df[pct_col])


@dataclass
class ScoutModel:
    players: pd.DataFrame            # outfield players in the model, index aligned with Z
    Z: np.ndarray                    # standardized (clipped) feature matrix
    percentiles: pd.DataFrame        # feature percentiles within position group
    archetypes: pd.DataFrame         # one row per archetype with description
    league_adjust: bool
    embedding: np.ndarray | None = field(default=None, repr=False)

    # ---- similarity -------------------------------------------------------
    def _weighted(self, weights: dict[str, float] | None) -> np.ndarray:
        if not weights:
            return self.Z
        w = np.array([weights.get(FEATURES[c][1], 1.0) for c in FEATURE_COLS])
        return self.Z * np.sqrt(w)

    def similar(self, player_id: int, pool_mask: np.ndarray | None = None, n: int = 10,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
        Zw = self._weighted(weights)
        i = self.players.index[self.players["player_id"] == player_id][0]
        sims = _cosine(Zw[i:i + 1], Zw)[0]
        mask = np.ones(len(sims), bool) if pool_mask is None else pool_mask.copy()
        mask[i] = False
        order = np.argsort(-np.where(mask, sims, -np.inf))[:n]
        out = self.players.iloc[order].copy()
        out.insert(0, "similarity", (sims[order] * 100).round(1))
        return out

    def top_comps(self, query_mask: np.ndarray, pool_mask: np.ndarray, k: int = 3,
                  weights: dict[str, float] | None = None) -> pd.DataFrame:
        """For every query player, the k most similar players from the pool."""
        Zw = self._weighted(weights)
        q_idx, p_idx = np.where(query_mask)[0], np.where(pool_mask)[0]
        if len(q_idx) == 0 or len(p_idx) == 0:
            return pd.DataFrame(index=q_idx)
        S = _cosine(Zw[q_idx], Zw[p_idx])
        S[q_idx[:, None] == p_idx[None, :]] = -np.inf  # never match self
        top = np.argsort(-S, axis=1)[:, :k]
        names = self.players["name"].to_numpy()
        rows = []
        for r, qi in enumerate(q_idx):
            cols = top[r]
            rows.append({
                "comps": ", ".join(f"{names[p_idx[c]]} ({S[r, c] * 100:.0f}%)" for c in cols),
                "best_comp": names[p_idx[cols[0]]],
                "best_similarity": round(float(S[r, cols[0]] * 100), 1),
            })
        return pd.DataFrame(rows, index=q_idx)

    # ---- map ----------------------------------------------------------------
    def compute_embedding(self, seed: int = 42) -> np.ndarray:
        if self.embedding is None:
            perplexity = min(40, max(5, len(self.Z) // 50))
            self.embedding = TSNE(n_components=2, perplexity=perplexity, init="pca",
                                  random_state=seed).fit_transform(self.Z)
        return self.embedding


def _cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-9)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-9)
    return An @ Bn.T


def prepare_features(df: pd.DataFrame, min_minutes: int, league_adjust: bool) -> pd.DataFrame:
    df = df[df["pos_group"].isin(OUTFIELD_GROUPS) & (df["minutes"] >= min_minutes)
            & (df["has_player_stats"].astype(bool) | df["per90_qualified"].astype(bool))].copy()
    df = df.reset_index(drop=True)
    df["dribble_pct"] = _shrink_pct(df, "dribble_pct", "dribbles_p90")
    df["long_ball_pct"] = _shrink_pct(df, "long_ball_pct", "long_balls_p90")
    df["aerial_pct"] = _shrink_pct(df, "aerial_pct", "aerials_p90")
    if league_adjust:
        factor = df["league"].map(LEAGUE_STRENGTH).fillna(1.0)
        for c in LEAGUE_ADJUSTED:
            df[c] = df[c] * factor
    # Impute gaps (e.g. xG/shot for players who barely shoot) with group medians.
    for c in FEATURE_COLS:
        df[c] = df[c].fillna(df.groupby("pos_group")[c].transform("median")).fillna(df[c].median())
    return df


def _name_clusters(group: str, diffs: list[pd.Series]) -> dict[int, str]:
    """Assign each cluster the best-fitting scouting role (greedy, one role per cluster).

    Falls back to the cluster's strongest trait if a group has more clusters than roles.
    """
    roles = ROLES.get(group, {})
    scores = [(diffs[c][feats].mean(), c, role) for c in range(len(diffs)) for role, feats in roles.items()]
    names: dict[int, str] = {}
    for _, c, role in sorted(scores, reverse=True):
        if c not in names and role not in names.values():
            names[c] = role
    for c in range(len(diffs)):
        if c not in names:
            names[c] = TRAIT_NAMES[diffs[c].idxmax()]
    return names


def build_model(df: pd.DataFrame, min_minutes: int = 600, league_adjust: bool = True,
                seed: int = 42) -> ScoutModel:
    players = prepare_features(df, min_minutes, league_adjust)
    reliable = (players["minutes"] >= max(RELIABLE_MINUTES, min_minutes)).to_numpy()

    scaler = StandardScaler().fit(players.loc[reliable, FEATURE_COLS])
    Z = np.clip(scaler.transform(players[FEATURE_COLS]), -Z_CLIP, Z_CLIP)

    # Percentiles within position group, relative to reliable players of that group.
    pct = pd.DataFrame(index=players.index, columns=FEATURE_COLS, dtype=float)
    for g, idx in players.groupby("pos_group").groups.items():
        ref = players.loc[idx][reliable[idx]] if reliable[idx].sum() >= 20 else players.loc[idx]
        for c in FEATURE_COLS:
            ref_vals = np.sort(ref[c].to_numpy())
            pct.loc[idx, c] = np.searchsorted(ref_vals, players.loc[idx, c].to_numpy(), side="right") / len(ref_vals) * 100

    # Archetypes: KMeans inside each position group.
    players["archetype"] = ""
    arche_rows = []
    for g in OUTFIELD_GROUPS:
        gmask = (players["pos_group"] == g).to_numpy()
        fit_mask = gmask & reliable
        if fit_mask.sum() < 10:
            continue
        k = ARCHETYPES_PER_GROUP[g]
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z[fit_mask])
        labels = km.predict(Z[gmask])
        group_mean, group_std = Z[fit_mask].mean(axis=0), Z[fit_mask].std(axis=0).clip(min=1e-6)
        # Centroid vs. group average, in within-group standard deviations.
        diffs = [pd.Series((km.cluster_centers_[c] - group_mean) / group_std, index=FEATURE_COLS)
                 for c in range(k)]
        names = _name_clusters(g, diffs)
        for c in range(k):
            diff = diffs[c]
            members = players.loc[gmask].iloc[np.where(labels == c)[0]]
            # Reference players: established top-flight players closest to the centroid.
            cand = members[(members["tier"] == 1) & (members["minutes"] >= RELIABLE_MINUTES)]
            if len(cand):
                d = np.linalg.norm(Z[cand.index] - km.cluster_centers_[c], axis=1)
                examples = ", ".join(cand.iloc[np.argsort(d)[:5]]["name"])
            else:
                examples = ""
            top_traits = diff.sort_values(ascending=False).head(4)
            arche_rows.append({
                "pos_group": g,
                "archetype": f"{g} · {names[c]}",
                "players": len(members),
                "key_traits": ", ".join(FEATURES[t][0] for t in top_traits.index if np.isfinite(top_traits[t])),
                "prototypes": examples,
            })
        players.loc[gmask, "archetype"] = [f"{g} · {names[l]}" for l in labels]

    return ScoutModel(players=players, Z=Z, percentiles=pct,
                      archetypes=pd.DataFrame(arche_rows), league_adjust=league_adjust)
