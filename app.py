"""
Application de détection de fraude : Streamlit.
Lancer :  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fraud_core as fc

CURRENCY = "$"                      # devise d'affichage des montants
RED, GREEN, BLUE, GREY = "#E45756", "#54A24B", "#4C78A8", "#9AA0A6"
R_COST, R_CAP, R_MAN = "Recommandée (coût minimal)", "Selon ma capacité de traitement", "Seuil de risque manuel"
MAX_ROWS = 3000                     # alertes détaillées affichées (les plus risquées)

st.set_page_config(page_title="Détection de fraude", page_icon="🛡️", layout="wide")
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_art():
    return fc.load_artifacts()


try:
    art = get_art()
except FileNotFoundError:
    st.error("Fichier `artifacts.joblib` introuvable : placez-le à côté de `app.py` "
             "(ou définissez la variable d'environnement ARTIFACTS_PATH).")
    st.stop()

VAL, TEST, TEST_DAYS = art["val"], art["test"], art["test_days"]


@st.cache_data
def get_demo():
    d = art["demo"].copy()
    d["day"] = pd.to_datetime(d["transaction_date"]).dt.date
    return d


@st.cache_data
def optimal_threshold(c: float) -> float:
    return art["threshold"] if abs(c - art["c_alert"]) < 1e-9 else fc.best_threshold(VAL, c)


@st.cache_data
def get_curves(c: float) -> pd.DataFrame:
    return fc.curves(TEST, c)


@st.cache_data(show_spinner="Analyse des transactions…")
def score_raw(raw: pd.DataFrame):
    X = fc.make_features(raw, art)
    return X, fc.score(X, art)


def money(x: float) -> str:
    return f"{x:,.0f} {CURRENCY}".replace(",", " ")


# --------------------------------------------------------------------------- #
# Barre latérale : règle d'alerte
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛡️ Détection de fraude")
    st.caption("Transactions e-commerce")
    st.divider()
    st.subheader("Règle d'alerte")
    rule = st.radio("Déclenchement des alertes", [R_COST, R_CAP, R_MAN], label_visibility="collapsed")
    cost_alert = st.number_input(
        f"Coût d'analyse d'un dossier ({CURRENCY})", min_value=1.0, max_value=1000.0,
        value=float(art["c_alert"]), step=5.0,
        help="Temps d'analyste et gêne client pour chaque alerte traitée. "
             "Le seuil recommandé équilibre ce coût et les montants de fraude manqués.")
    capacity = st.slider("Capacité de l'équipe (dossiers / jour)", 50, 3000,
                         int(art["capacity_per_day"]), step=50)
    opt_thr = optimal_threshold(cost_alert)
    if rule == R_MAN:
        manual_thr = st.slider("Seuil de risque", 0.0, 1.0, float(round(opt_thr, 2)), 0.01,
                               help="Une transaction est en alerte si son score dépasse ce seuil.")
    elif rule == R_COST:
        st.caption(f"Seuil recommandé : **{opt_thr:.2f}**")
    else:
        st.caption(f"Les {capacity} dossiers les plus risqués chaque jour sont traités.")


def decide(scores: np.ndarray, n_days: int) -> tuple[np.ndarray, float]:
    """Règle active -> (masque d'alertes, seuil effectif)."""
    if rule == R_COST:
        thr = opt_thr
    elif rule == R_CAP:
        thr = fc.threshold_for_budget(scores, int(capacity * n_days))
    else:
        thr = manual_thr
    return scores >= thr, thr


def reference_threshold() -> float:
    """Seuil équivalent pour juger une transaction isolée."""
    return decide(TEST["score"].to_numpy(), TEST_DAYS)[1]


# --------------------------------------------------------------------------- #
# Graphique d'explication (waterfall)
# --------------------------------------------------------------------------- #
def waterfall_fig(e: dict, score_value: float, top: int = 8) -> go.Figure:
    items = e["contributions"][:top]
    rest = sum(c["contribution"] for c in e["contributions"][top:])
    labels = ["Risque moyen"] + [f"{c['label']} : {c['value']}" for c in items]
    values = [e["base_value"]] + [c["contribution"] for c in items]
    measures = ["absolute"] + ["relative"] * len(items)
    text = [""] + [f"{v:+.2f}" for v in values[1:]]
    if abs(rest) > 1e-9:
        labels.append("Autres facteurs"); values.append(rest); measures.append("relative"); text.append(f"{rest:+.2f}")
    labels.append("Cette transaction"); values.append(e["margin"]); measures.append("total")
    text.append(f"score {score_value * 100:.0f}/100")
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measures, text=text, textposition="outside",
        increasing={"marker": {"color": RED}}, decreasing={"marker": {"color": GREEN}},
        totals={"marker": {"color": BLUE}}, connector={"line": {"color": GREY, "width": 1}}))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 38 * len(labels), margin=dict(l=0, r=30, t=10, b=40), showlegend=False,
                      xaxis_title="Niveau de risque  (rouge : augmente le risque · vert : le réduit)")
    return fig


def show_explanation(e: dict, s: float, thr: float):
    c1, c2 = st.columns([1, 2])
    with c1:
        lvl = fc.risk_level(s, thr)
        icon = {"Élevé": "🔴", "Moyen": "🟠", "Faible": "🟢"}[lvl]
        st.metric("Score de risque", f"{s * 100:.0f} / 100", help="Sert à classer les transactions ; ce n'est pas une probabilité.")
        st.markdown(f"**Niveau : {icon} {lvl}**  \nSeuil d'alerte : {thr * 100:.0f}")
        st.markdown("**Motifs principaux de l'alerte**")
        if e["reason_codes"]:
            for r in e["reason_codes"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("_Aucun facteur ne pousse vers l'alerte._")
    with c2:
        st.plotly_chart(waterfall_fig(e, s), width="stretch")


# --------------------------------------------------------------------------- #
# Onglets
# --------------------------------------------------------------------------- #
tab_alerts, tab_one, tab_perf, tab_trends, tab_model = st.tabs(
    ["📋 Alertes", "🔎 Analyser une transaction", "📈 Performance et coûts", "📊 Tendances de fraude", "🧠 Comprendre le modèle"])

# ============================== 1. ALERTES ================================== #
with tab_alerts:
    st.header("Alertes à traiter")
    source = st.radio("Transactions à analyser", ["Journée de démonstration", "Importer un fichier"],
                      horizontal=True, label_visibility="collapsed")
    raw = None
    if source == "Journée de démonstration":
        demo = get_demo()
        days = sorted(demo["day"].unique())
        day = st.selectbox("Journée", days, index=len(days) - 1, format_func=lambda d: d.strftime("%d/%m/%Y"))
        raw = demo[demo["day"] == day].drop(columns="day").reset_index(drop=True)
    else:
        up = st.file_uploader("Fichier CSV de transactions", type=["csv"])
        with st.expander("Format du fichier attendu"):
            st.write("Une ligne par transaction, avec au minimum : `transaction_date`, `transaction_amount`, "
                     "`payment_method`, `product_category`, `device_used`, `quantity`, `customer_age`, "
                     "`account_age_days`. Optionnel : `transaction_id`, `addr_match` (1 si adresses identiques) "
                     "ou `shipping_address` + `billing_address`.")
            st.write("Modalités connues : " + " · ".join(
                f"**{fc.GROUP_LABELS[c]}** : {', '.join(v)}" for c, v in art["cat_levels"].items()))
            st.download_button("Télécharger un fichier exemple", get_demo().drop(columns="day").head(20).to_csv(index=False),
                               "exemple_transactions.csv", "text/csv")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                miss = fc.missing_columns(raw)
                if miss:
                    st.error(f"Colonnes manquantes : {', '.join(miss)}")
                    raw = None
            except Exception as ex:
                st.error(f"Fichier illisible : {ex}")
                raw = None
        if raw is not None:
            unk = fc.unknown_categories(raw, art)
            if unk:
                st.warning("Modalités inconnues du modèle (traitées comme neutres) : "
                           + " ; ".join(f"{k} → {', '.join(v)}" for k, v in unk.items()))

    if raw is None:
        st.info("Importez un fichier pour lancer l'analyse.")
    else:
        try:
            X, scores = score_raw(raw)
        except ValueError as ex:
            st.error(f"Données invalides : {ex}")
            st.stop()
        nr = fc.normalize_columns(raw)
        dates = pd.to_datetime(nr["transaction_date"])
        n_days = max(1, (dates.max().normalize() - dates.min().normalize()).days + 1)
        mask, thr = decide(scores, n_days)
        amounts = pd.to_numeric(nr["transaction_amount"]).to_numpy()
        ids = nr["transaction_id"].astype(str).to_numpy() if "transaction_id" in nr else (np.arange(len(nr)) + 1).astype(str)

        n_alerts = int(mask.sum())
        per_day = n_alerts / n_days
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transactions analysées", f"{len(scores):,}".replace(",", " "))
        k2.metric("Alertes", f"{n_alerts:,}".replace(",", " "), f"{n_alerts / len(scores):.1%} du volume", delta_color="off")
        k3.metric("Montant sous alerte", money(amounts[mask].sum()))
        k4.metric("Charge par jour", f"{per_day:,.0f}".replace(",", " "), f"{per_day - capacity:+,.0f} vs capacité".replace(",", " "),
                  delta_color="inverse")
        if per_day > capacity * 1.05:
            st.warning(f"Cette règle génère environ {per_day:,.0f} dossiers par jour, au-dessus de la capacité de "
                       f"l'équipe ({capacity}). Choisissez « {R_CAP} » pour ne traiter que les plus risqués.".replace(",", " "))

        if n_alerts == 0:
            st.success("Aucune alerte avec la règle actuelle.")
        else:
            order = np.where(mask)[0]
            order = order[np.argsort(-scores[order])][:MAX_ROWS]
            reasons = fc.reason_texts(X.iloc[order], art, n_reasons=4)
            view = pd.DataFrame({
                "Rang": np.arange(1, len(order) + 1),
                "Transaction": ids[order],
                "Date": dates.iloc[order].to_numpy(),
                "Montant": amounts[order],
                "Risque": np.round(scores[order] * 100),
                "Niveau": [fc.risk_level(s, thr) for s in scores[order]],
                "Motifs principaux": reasons,
            })
            st.caption(f"{len(view)} alertes les plus risquées affichées. Cliquez sur une ligne pour voir l'explication détaillée.")
            event = st.dataframe(
                view, hide_index=True, width="stretch", height=380,
                on_select="rerun", selection_mode="single-row",
                column_config={
                    "Montant": st.column_config.NumberColumn(f"Montant ({CURRENCY})", format="%.2f"),
                    "Risque": st.column_config.ProgressColumn("Risque", min_value=0, max_value=100, format="%d"),
                    "Date": st.column_config.DatetimeColumn("Date", format="DD/MM HH:mm"),
                    "Motifs principaux": st.column_config.TextColumn(width="large"),
                })
            st.download_button("⬇️ Exporter les alertes (CSV)", view.to_csv(index=False).encode("utf-8"),
                               "alertes.csv", "text/csv")
            sel = event.selection.rows
            if sel:
                i = int(order[sel[0]])
                st.divider()
                st.subheader(f"Transaction {ids[i]}")
                st.caption(f"Montant {money(amounts[i])} · {dates.iloc[i]:%d/%m/%Y %H:%M}")
                show_explanation(fc.explain_rows(X.iloc[[i]], art)[0], float(scores[i]), thr)

# ====================== 2. ANALYSER UNE TRANSACTION ========================= #
with tab_one:
    st.header("Analyser une transaction")
    st.caption("Saisissez une transaction pour obtenir son niveau de risque et les raisons de la décision.")
    lv = art["cat_levels"]
    with st.form("tx_form"):
        c1, c2, c3 = st.columns(3)
        d_in = c1.date_input("Date", dt.date.today())
        t_in = c1.time_input("Heure", dt.time(14, 0))
        amount_in = c1.number_input(f"Montant ({CURRENCY})", min_value=0.01, value=150.0, step=10.0)
        pay_in = c2.selectbox("Moyen de paiement", lv["payment_method"])
        cat_in = c2.selectbox("Catégorie de produit", lv["product_category"])
        dev_in = c2.selectbox("Appareil", lv["device_used"])
        qty_in = c3.number_input("Quantité", min_value=1, value=1)
        age_in = c3.number_input("Âge du client", min_value=18, max_value=100, value=35)
        acc_in = c3.number_input("Ancienneté du compte (jours)", min_value=0, value=200)
        addr_in = st.checkbox("Adresse de livraison identique à l'adresse de facturation", value=True)
        go_btn = st.form_submit_button("Évaluer la transaction", type="primary")
    if go_btn:
        st.session_state["tx_single"] = {
            "transaction_date": dt.datetime.combine(d_in, t_in), "transaction_amount": amount_in,
            "payment_method": pay_in, "product_category": cat_in, "device_used": dev_in,
            "quantity": qty_in, "customer_age": age_in, "account_age_days": acc_in, "addr_match": int(addr_in)}
    if "tx_single" in st.session_state:
        Xs = fc.make_features(pd.DataFrame([st.session_state["tx_single"]]), art)
        s1 = float(fc.score(Xs, art)[0])
        thr1 = reference_threshold()
        if s1 >= thr1:
            st.error("🔴 **Alerte** : cette transaction doit être examinée par un analyste.")
        elif s1 >= 0.6 * thr1:
            st.warning("🟠 **À surveiller** : risque modéré, sous le seuil d'alerte.")
        else:
            st.success("🟢 **Aucune alerte** : risque faible.")
        show_explanation(fc.explain_rows(Xs, art)[0], s1, thr1)

# =========================== 3. PERFORMANCE ================================= #
with tab_perf:
    st.header("Performance et coûts")
    st.caption(f"Mesures sur {TEST_DAYS} jours de transactions réelles du jeu, non utilisés pour construire le modèle "
               f"({len(TEST):,} transactions, {TEST['y'].mean():.1%} de fraudes).".replace(",", " "))
    thr_t = decide(TEST["score"].to_numpy(), TEST_DAYS)[1]
    m = fc.metrics_at(TEST, thr_t, cost_alert)

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Alertes par jour", f"{m['alerts'] / TEST_DAYS:,.0f}".replace(",", " "), f"{m['pct_alerts']:.1%} du volume", delta_color="off")
    a2.metric("Précision", f"{m['precision']:.1%}", help="Part des alertes qui sont de vraies fraudes.")
    a3.metric("Rappel", f"{m['recall']:.1%}", help="Part des fraudes détectées.")
    a4.metric("F2", f"{m['f2']:.2f}", help="Compromis précision/rappel favorisant le rappel.")
    a5.metric("MCC", f"{m['mcc']:.2f}", help="Qualité globale de la décision (de -1 à 1, 0 = hasard).")

    naive = min(m["cost_no_model"], m["cost_alert_all"])
    saving = naive - m["cost"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Coût avec le modèle", money(m["cost"]), help="Fraudes manquées + coût d'analyse des alertes.")
    b2.metric("Sans aucun contrôle", money(m["cost_no_model"]), help="Toutes les fraudes passent.")
    b3.metric("En alertant tout", money(m["cost_alert_all"]))
    b4.metric("Économie", money(saving), f"{saving / naive:.0%} vs la meilleure alternative simple", delta_color="off")
    if m["alerts"] / TEST_DAYS > capacity * 1.05:
        st.warning(f"Avec cette règle : {m['alerts'] / TEST_DAYS:,.0f} alertes/jour pour une capacité de {capacity}.".replace(",", " "))

    cv = get_curves(cost_alert)
    left, right = st.columns(2)
    with left:
        f = go.Figure(go.Scatter(x=cv["pct_alerts"] * 100, y=cv["cost"], mode="lines", line=dict(color=BLUE)))
        f.add_trace(go.Scatter(x=[m["pct_alerts"] * 100], y=[m["cost"]], mode="markers", marker=dict(size=12, color=RED),
                               name="Règle actuelle"))
        f.update_layout(title="Coût selon la part de transactions alertées", height=340, showlegend=False,
                        xaxis_title="% de transactions alertées", yaxis_title=f"Coût ({CURRENCY})",
                        margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(f, width="stretch")
    with right:
        c2_ = cv.dropna(subset=["precision"])
        f = go.Figure(go.Scatter(x=c2_["recall"], y=c2_["precision"], mode="lines", line=dict(color=BLUE)))
        f.add_hline(y=TEST["y"].mean(), line_dash="dash", line_color=GREY, annotation_text="sans modèle")
        f.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", marker=dict(size=12, color=RED)))
        f.update_layout(title="Précision et rappel", height=340, showlegend=False, xaxis_title="Rappel (fraudes détectées)",
                        yaxis_title="Précision (alertes justifiées)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(f, width="stretch")

    if rule == R_COST:
        thr_oracle = fc.best_threshold(TEST, cost_alert)
        c_or = fc.metrics_at(TEST, thr_oracle, cost_alert)["cost"]
        st.caption(f"Fiabilité du seuil recommandé : choisi sur une période antérieure, il donne un coût de {money(m['cost'])} "
                   f"contre {money(c_or)} avec le meilleur seuil possible a posteriori (écart {m['cost'] / c_or - 1:.1%}).")

    with st.expander("Pourquoi ce modèle ? Comparaison avec d'autres méthodes"):
        comp = art["comparison"].copy()
        st.dataframe(comp.style.format("{:.3f}", subset=[c for c in comp.columns if "Coût" not in c])
                     .format("{:,.0f}", subset=[c for c in comp.columns if "Coût" in c]), width="stretch")
        st.caption("Les méthodes non supervisées (Isolation Forest, LOF) n'utilisent aucune étiquette ; "
                   "les méthodes supervisées apprennent des fraudes passées et sont nettement plus précises ici.")
        f = go.Figure()
        for name, pc in art["pr_curves"].items():
            f.add_trace(go.Scatter(x=pc["recall"], y=pc["precision"], mode="lines", name=name))
        f.add_hline(y=TEST["y"].mean(), line_dash="dash", line_color=GREY)
        f.update_layout(height=360, xaxis_title="Rappel", yaxis_title="Précision", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f, width="stretch")

# ============================== 4. TENDANCES ================================ #
with tab_trends:
    eda = art["eda"]
    k = eda["kpi"]
    st.header("Tendances de fraude")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Transactions", f"{k['n']:,}".replace(",", " "))
    t2.metric("Taux de fraude", f"{k['fraud_rate']:.2%}")
    t3.metric("Montant moyen · fraude", money(k["mean_amount_fraud"]), f"légitime : {money(k['mean_amount_legit'])}", delta_color="off")
    t4.metric("Période", f"{pd.Timestamp(k['start']):%d/%m/%y} → {pd.Timestamp(k['end']):%d/%m/%y}")

    def bar(df, x, title, xlabel):
        fig = go.Figure(go.Bar(x=df[x], y=df["rate"] * 100, marker_color=BLUE,
                               hovertemplate="%{x}<br>%{y:.2f} %<extra></extra>"))
        fig.add_hline(y=k["fraud_rate"] * 100, line_dash="dash", line_color=GREY)
        fig.update_layout(title=title, height=320, xaxis_title=xlabel, yaxis_title="Taux de fraude (%)",
                          margin=dict(l=0, r=0, t=40, b=0))
        return fig

    wk = eda["weekly"][eda["weekly"]["n"] > 0.2 * eda["weekly"]["n"].max()]
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure(go.Scatter(x=wk["transaction_date"], y=wk["rate"] * 100, mode="lines+markers", line=dict(color=BLUE)))
        f.add_hline(y=k["fraud_rate"] * 100, line_dash="dash", line_color=GREY)
        f.update_layout(title="Évolution hebdomadaire", height=320, yaxis_title="Taux de fraude (%)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(f, width="stretch")
    with c2:
        st.plotly_chart(bar(eda["hourly"], "hour", "Selon l'heure de la journée", "Heure"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        dim = st.selectbox("Taux de fraude selon", ["product_category", "payment_method", "device_used"],
                           format_func=lambda c: fc.GROUP_LABELS[c])
        st.plotly_chart(bar(eda["by_cat"][dim], dim, "", fc.GROUP_LABELS[dim]), width="stretch")
    with c4:
        st.markdown("&nbsp;")
        st.plotly_chart(bar(eda["account_age"], "bucket", "Selon l'ancienneté du compte", "Ancienneté"), width="stretch")

    ah = eda["amount_hist"]
    f = go.Figure()
    f.add_trace(go.Scatter(x=ah["amount"], y=ah["Légitime"], name="Légitime", fill="tozeroy", line=dict(color=BLUE)))
    f.add_trace(go.Scatter(x=ah["amount"], y=ah["Fraude"], name="Fraude", fill="tozeroy", line=dict(color=RED)))
    f.update_layout(title="Distribution des montants", height=320, xaxis_type="log", xaxis_title=f"Montant ({CURRENCY}, échelle log)",
                    yaxis_title="Densité", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(f, width="stretch")

# ========================== 5. COMPRENDRE LE MODÈLE ========================= #
with tab_model:
    st.header("Comprendre le modèle")
    st.write("Le modèle attribue à chaque transaction un score de risque à partir de ses caractéristiques. "
             "Le graphique ci-dessous indique **quels critères pèsent le plus** dans ses décisions, en moyenne.")
    sv = art["shap_values_sample"]
    G, names = fc.group_matrix(sv.to_numpy(), list(sv.columns))
    imp = pd.DataFrame({"Critère": [fc.GROUP_LABELS.get(n, n) for n in names],
                        "Poids": np.abs(G).mean(axis=0)}).sort_values("Poids")
    f = go.Figure(go.Bar(x=imp["Poids"], y=imp["Critère"], orientation="h", marker_color=BLUE))
    f.update_layout(height=380, xaxis_title="Influence moyenne sur le score", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f, width="stretch")
    with st.expander("Voir le détail : sens de l'effet de chaque variable (SHAP)"):
        st.image(art["shap_beeswarm_png"])
        st.caption("Chaque point est une transaction. À droite : la variable augmente le risque. "
                   "Rouge : valeur élevée de la variable ; bleu : valeur faible.")
    with st.expander("Comment utiliser le score"):
        st.markdown(
            "- Le score **classe** les transactions du plus au moins risqué ; ce n'est pas une probabilité de fraude.\n"
            "- Une alerte est une **aide à la décision** : elle déclenche un examen par un analyste, pas un blocage automatique.\n"
            "- Le modèle a été entraîné sur des données **synthétiques** : les performances en production devront être revérifiées.\n"
            "- Le modèle utilise l'âge du client : son effet sur les alertes doit être surveillé pour éviter toute discrimination.")