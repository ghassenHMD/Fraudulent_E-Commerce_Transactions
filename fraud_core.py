"""
fraud_core.py : noyau commun à l'API (api.py) et à l'application (app.py).

Une seule source de vérité pour :
  - le chargement de l'artefact (artifacts.joblib)
  - la construction des variables à partir d'une transaction brute
  - le score, l'explication (SHAP natif LightGBM) et les codes motifs
  - les calculs de coût et de métriques

Vérification : `python fraud_core.py` compare les variables/scores reconstruits
avec ceux du notebook (échantillon de parité stocké dans l'artefact).
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", str(Path(__file__).with_name("artifacts.joblib")))

CAT_COLS = ["payment_method", "product_category", "device_used"]
RAW_COLS = ["transaction_date", "transaction_amount", "payment_method", "product_category",
            "device_used", "quantity", "customer_age", "account_age_days"]
DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

GROUP_LABELS = {
    "hour": "Heure de la transaction",
    "day": "Jour de la semaine",
    "amount": "Montant",
    "address": "Adresses livraison / facturation",
    "payment_method": "Moyen de paiement",
    "product_category": "Catégorie de produit",
    "device_used": "Appareil utilisé",
    "quantity": "Quantité",
    "customer_age": "Âge du client",
    "account_age_days": "Ancienneté du compte",
}


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_artifacts(path: str | None = None) -> dict:
    return joblib.load(path or ARTIFACTS_PATH)


# --------------------------------------------------------------------------- #
# Variables (doit reproduire EXACTEMENT la préparation du notebook)
# --------------------------------------------------------------------------- #
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (out.columns.astype(str).str.strip().str.lower()
                   .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    return out


def missing_columns(raw: pd.DataFrame) -> list[str]:
    cols = set(normalize_columns(raw).columns)
    return [c for c in RAW_COLS if c not in cols]


def unknown_categories(raw: pd.DataFrame, art: dict) -> dict[str, list[str]]:
    """Valeurs catégorielles absentes des modalités vues à l'entraînement."""
    raw = normalize_columns(raw)
    out = {}
    for c in CAT_COLS:
        known = {l.lower() for l in art["cat_levels"][c]}
        vals = raw[c].astype(str).str.strip().str.lower()
        bad = sorted(set(raw.loc[~vals.isin(known), c].astype(str)))
        if bad:
            out[c] = bad
    return out


def make_features(raw: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Transactions brutes -> matrice de variables dans l'ordre exact du modèle."""
    raw = normalize_columns(raw)
    miss = [c for c in RAW_COLS if c not in raw.columns]
    if miss:
        raise ValueError(f"Colonnes manquantes : {miss}")

    dt = pd.to_datetime(raw["transaction_date"], errors="coerce")
    if dt.isna().any():
        raise ValueError("transaction_date : valeurs de date invalides")

    def num(col):
        s = pd.to_numeric(raw[col], errors="coerce")
        if s.isna().any():
            raise ValueError(f"{col} : valeurs manquantes ou non numériques")
        return s

    amount = num("transaction_amount")
    if (amount < 0).any():
        raise ValueError("transaction_amount : montant négatif")

    age = pd.to_numeric(raw["customer_age"], errors="coerce")
    age = age.where(age > 0, art["age_median"]).fillna(art["age_median"])   # âges <= 0 ou absents

    hour = dt.dt.hour
    X = pd.DataFrame(index=raw.index)
    X["quantity"] = num("quantity")
    X["customer_age"] = age
    X["account_age_days"] = num("account_age_days")
    if "addr_match" in raw.columns:
        X["addr_match"] = raw["addr_match"].astype(int)
    elif {"shipping_address", "billing_address"} <= set(raw.columns):
        X["addr_match"] = (raw["shipping_address"] == raw["billing_address"]).astype(int)
    else:
        X["addr_match"] = 1
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow"] = dt.dt.dayofweek
    X["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    X["is_night"] = hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    X["log_amount"] = np.log1p(amount)

    for c in CAT_COLS:
        levels = art["cat_levels"][c]
        lookup = {l.lower(): l for l in levels}
        vals = raw[c].astype(str).str.strip().str.lower().map(lookup)
        for l in levels:
            X[f"{c}_{l}"] = (vals == l).astype(int)

    return X.reindex(columns=art["feature_cols"], fill_value=0).astype(float)


# --------------------------------------------------------------------------- #
# Score et explication
# --------------------------------------------------------------------------- #
def score(X: pd.DataFrame, art: dict) -> np.ndarray:
    return art["model"].predict_proba(X)[:, 1]


def risk_level(s: float, thr: float) -> str:
    if s >= thr:
        return "Élevé"
    if s >= 0.6 * thr:
        return "Moyen"
    return "Faible"


def contributions(X: pd.DataFrame, art: dict) -> tuple[np.ndarray, np.ndarray]:
    """Valeurs SHAP exactes (TreeSHAP natif LightGBM), en log-odds.
    Retourne (contributions n x F, valeur de base n)."""
    c = art["model"].predict(X, pred_contrib=True)
    return c[:, :-1], c[:, -1]


def group_of(feature: str) -> str:
    """Regroupe les variables techniques en variables métier (l'additivité de SHAP est conservée)."""
    if feature in ("hour_sin", "hour_cos", "is_night"):
        return "hour"
    if feature in ("dow", "is_weekend"):
        return "day"
    if feature == "log_amount":
        return "amount"
    if feature == "addr_match":
        return "address"
    for c in CAT_COLS:
        if feature.startswith(c + "_"):
            return c
    return feature


def group_matrix(contrib: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    groups = [group_of(f) for f in feature_cols]
    names = list(dict.fromkeys(groups))
    M = np.zeros((len(feature_cols), len(names)))
    for j, g in enumerate(groups):
        M[j, names.index(g)] = 1.0
    return contrib @ M, names


def describe_value(group: str, x: np.ndarray, feature_cols: list[str]) -> str:
    pos = {f: i for i, f in enumerate(feature_cols)}
    if group == "hour":
        ang = np.arctan2(x[pos["hour_sin"]], x[pos["hour_cos"]]) % (2 * np.pi)
        return f"{int(round(ang * 24 / (2 * np.pi))) % 24:02d}h"
    if group == "day":
        return DAYS_FR[int(x[pos["dow"]])]
    if group == "amount":
        return f"{np.expm1(x[pos['log_amount']]):,.2f}"
    if group == "address":
        return "identiques" if x[pos["addr_match"]] >= 0.5 else "différentes"
    if group in CAT_COLS:
        for f in feature_cols:
            if f.startswith(group + "_") and x[pos[f]] >= 0.5:
                return f[len(group) + 1:]
        return "inconnu"
    if group == "customer_age":
        return f"{int(x[pos['customer_age']])} ans"
    if group == "account_age_days":
        return f"{int(x[pos['account_age_days']])} jours"
    return f"{int(x[pos[group]])}"


def explain_rows(X: pd.DataFrame, art: dict, n_reasons: int = 4) -> list[dict]:
    """Explication de chaque ligne : contributions par variable métier + codes motifs
    (les n variables qui poussent le plus vers l'alerte)."""
    cols = art["feature_cols"]
    contrib, base = contributions(X, art)
    G, names = group_matrix(contrib, cols)
    Xv = X.to_numpy()
    out = []
    for i in range(len(X)):
        items = [{
            "group": g,
            "label": GROUP_LABELS.get(g, g),
            "value": describe_value(g, Xv[i], cols),
            "contribution": float(G[i, j]),
        } for j, g in enumerate(names)]
        items.sort(key=lambda d: -abs(d["contribution"]))
        reasons = [f"{d['label']} : {d['value']}"
                   for d in sorted(items, key=lambda d: -d["contribution"])[:n_reasons]
                   if d["contribution"] > 0]
        out.append({"base_value": float(base[i]),
                    "margin": float(base[i] + G[i].sum()),
                    "contributions": items,
                    "reason_codes": reasons})
    return out


def reason_texts(X: pd.DataFrame, art: dict, n_reasons: int = 4) -> list[str]:
    """Version rapide pour de grands lots : une chaîne de codes motifs par ligne."""
    cols = art["feature_cols"]
    contrib, _ = contributions(X, art)
    G, names = group_matrix(contrib, cols)
    Xv = X.to_numpy()
    order = np.argsort(-G, axis=1)[:, :n_reasons]
    out = []
    for i in range(len(X)):
        parts = [f"{GROUP_LABELS.get(names[j], names[j])} : {describe_value(names[j], Xv[i], cols)}"
                 for j in order[i] if G[i, j] > 0]
        out.append(" · ".join(parts))
    return out


# --------------------------------------------------------------------------- #
# Coût métier et métriques (df avec colonnes score, y, amount)
# --------------------------------------------------------------------------- #
def best_threshold(df: pd.DataFrame, c: float) -> float:
    """Seuil minimisant : montants des fraudes manquées + c x nombre d'alertes."""
    s, y, a = df["score"].to_numpy(), df["y"].to_numpy(), df["amount"].to_numpy()
    order = np.argsort(-s)
    fa = (a * y)[order]
    cost = (fa.sum() - np.concatenate([[0.0], np.cumsum(fa)])) + c * np.arange(len(s) + 1)
    j = int(cost.argmin())
    return float("inf") if j == 0 else float(s[order][j - 1])


def threshold_for_budget(scores: np.ndarray, k: int) -> float:
    """Seuil qui laisse passer les k transactions les plus risquées."""
    scores = np.asarray(scores)
    if k <= 0:
        return float("inf")
    if k >= len(scores):
        return float(scores.min())
    return float(np.partition(scores, -k)[-k])


def metrics_at(df: pd.DataFrame, thr: float, c: float) -> dict:
    s, y, a = df["score"].to_numpy(), df["y"].to_numpy(), df["amount"].to_numpy()
    alert = s >= thr
    tp = int((alert & (y == 1)).sum())
    n_alert = int(alert.sum())
    fp = n_alert - tp
    fn = int(((~alert) & (y == 1)).sum())
    tn = len(y) - tp - fp - fn
    prec = tp / n_alert if n_alert else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {
        "n": len(y), "alerts": n_alert, "pct_alerts": n_alert / len(y),
        "precision": prec, "recall": rec, "f2": f2, "mcc": float(mcc),
        "fraud_caught": tp, "fraud_missed": fn,
        "amount_missed": float(a[(y == 1) & ~alert].sum()),
        "cost": float(a[(y == 1) & ~alert].sum() + c * n_alert),
        "cost_no_model": float(a[y == 1].sum()),
        "cost_alert_all": float(c * len(y)),
    }


def curves(df: pd.DataFrame, c: float, npts: int = 400) -> pd.DataFrame:
    """Précision, rappel et coût en fonction du nombre d'alertes (classées par score)."""
    s, y, a = df["score"].to_numpy(), df["y"].to_numpy(), df["amount"].to_numpy()
    order = np.argsort(-s)
    ys, fa = y[order], (a * y)[order]
    tp, caught = np.cumsum(ys), np.cumsum(fa)
    N = len(s)
    ks = np.unique(np.concatenate([np.geomspace(1, N, npts), np.linspace(1, N, npts)]).astype(int))
    out = pd.DataFrame({
        "k": ks, "pct_alerts": ks / N,
        "precision": tp[ks - 1] / ks, "recall": tp[ks - 1] / max(ys.sum(), 1),
        "cost": (fa.sum() - caught[ks - 1]) + c * ks,
    })
    start = pd.DataFrame({"k": [0], "pct_alerts": [0.0], "precision": [np.nan],
                          "recall": [0.0], "cost": [fa.sum()]})
    return pd.concat([start, out], ignore_index=True)


# --------------------------------------------------------------------------- #
# Auto-contrôle : `python fraud_core.py`
# --------------------------------------------------------------------------- #
def selfcheck(path: str | None = None) -> None:
    art = load_artifacts(path)
    p = art["parity"]
    X = make_features(p["raw"], art)
    assert list(X.columns) == list(p["X"].columns), "ordre des colonnes différent"
    err_x = float(np.abs(X.to_numpy() - p["X"].to_numpy()).max())
    err_s = float(np.abs(score(X, art) - p["score"]).max())
    contrib, base = contributions(X, art)
    err_c = float(np.abs((contrib.sum(1) + base) - art["model"].predict(X, raw_score=True)).max())
    print(f"Variables : écart max = {err_x:.2e}")
    print(f"Scores    : écart max = {err_s:.2e}")
    print(f"SHAP      : écart additivité = {err_c:.2e}")
    assert err_x < 1e-9 and err_s < 1e-9 and err_c < 1e-6, "ÉCART : l'app ne reproduit pas le notebook"
    print(f"OK : modèle={art['model_name']} | seuil={art['threshold']:.4f} | "
          f"lightgbm notebook={art.get('versions', {}).get('lightgbm')}")


if __name__ == "__main__":
    selfcheck()